import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from xgboost import DMatrix
import numpy as np
import optuna
import sys

#  This script demonstrate how to do ranking with xgboost.train
x_train, y_train = load_svmlight_file("ltrc_yahoo/set1.train")
x_valid, y_valid = load_svmlight_file("ltrc_yahoo/set1.valid")
x_test, y_test = load_svmlight_file("ltrc_yahoo/set1.test")

group_train = []
with open("ltrc_yahoo/set1.train.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_train.append(int(line.split("\n")[0]))

group_valid = []
with open("ltrc_yahoo/set1.valid.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_valid.append(int(line.split("\n")[0]))

group_test = []
with open("ltrc_yahoo/set1.test.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_test.append(int(line.split("\n")[0]))

xgb_train = DMatrix(x_train, y_train)
xgb_vali = DMatrix(x_valid, y_valid)
xgb_test = DMatrix(x_test)

xgb_train.set_group(group_train)
xgb_vali.set_group(group_valid)
xgb_test.set_group(group_test)


def multiple_cutoff_rankings(scores, cutoff, return_full_rankings):
    n_samples = scores.shape[0]
    n_docs = scores.shape[1]
    cutoff = min(n_docs, cutoff)

    ind = np.arange(n_samples)
    partition = np.argpartition(scores, cutoff - 1)
    sorted_partition = np.argsort(scores[ind[:, None], partition[:, :cutoff]])
    rankings = partition[ind[:, None], sorted_partition]

    if return_full_rankings:
        partition[:, :cutoff] = rankings
        rankings = partition

    return rankings

def gumbel_sample_rankings(
    predict_scores, n_samples, cutoff=None, return_full_rankings=False
):
    n_docs = len(predict_scores)
    ind = np.arange(n_samples)

    if cutoff:
        ranking_len = min(n_docs, cutoff)
    else:
        ranking_len = n_docs

    gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
    gumbel_scores = predict_scores + gumbel_samples

    rankings = multiple_cutoff_rankings(
        -gumbel_scores, ranking_len, return_full_rankings
    )

    return rankings


def PL_rank_3_grad(rank_weights, labels, scores, n_samples):
    n_docs = labels.shape[0]
    cutoff = min(rank_weights.shape[0], n_docs)

    sampled_rankings = gumbel_sample_rankings(
        scores, n_samples, cutoff=cutoff, return_full_rankings=True
    )

    cutoff_sampled_rankings = sampled_rankings[:, :cutoff]

    scores = scores - np.max(scores) + 10.0

    srange = np.arange(n_samples)
    crange = np.arange(cutoff)

    weighted_labels = labels[cutoff_sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = np.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    # first order
    result1 = np.zeros(n_docs, dtype=np.float64)
    np.add.at(result1, cutoff_sampled_rankings[:, :-1], cumsum_labels[:, 1:])
    result1 /= n_samples

    exp_scores = np.exp(scores).astype(np.float64)
    denom_per_rank = np.cumsum(exp_scores[sampled_rankings[:, ::-1]], axis=1)[
        :, : -cutoff - 1 : -1
    ]

    # DR
    cumsum_weight_denom = np.cumsum(rank_weights[:cutoff] / denom_per_rank, axis=1)
    # RI
    cumsum_reward_denom = np.cumsum(cumsum_labels / denom_per_rank, axis=1)

    relevant_docs = np.where(np.not_equal(labels, 0))[0]
    if cutoff < n_docs:
        second_part = -exp_scores[None, :] * cumsum_reward_denom[:, -1, None]
        second_part[:, relevant_docs] += (
            labels[relevant_docs][None, :]
            * exp_scores[None, relevant_docs]
            * cumsum_weight_denom[:, -1, None]
        )
    else:
        second_part = np.empty((n_samples, n_docs), dtype=np.float64)

    sampled_direct_reward = (
        labels[cutoff_sampled_rankings]
        * exp_scores[cutoff_sampled_rankings]
        * cumsum_weight_denom
    )
    sampled_following_reward = exp_scores[cutoff_sampled_rankings] * cumsum_reward_denom
    second_part[srange[:, None], cutoff_sampled_rankings] = (
        sampled_direct_reward - sampled_following_reward
    )

    return -(result1 + np.mean(second_part, axis=0))

def PL_rank_3_hess(rank_weights, labels, scores, n_samples):
    n_docs = labels.shape[0]
    cutoff = min(rank_weights.shape[0], n_docs)
    relevant_docs = np.where(np.not_equal(labels, 0))[0]

    sampled_rankings = gumbel_sample_rankings(
        scores, n_samples, cutoff=cutoff, return_full_rankings=True
    )

    cutoff_sampled_rankings = sampled_rankings[:, :cutoff]

    scores = scores - np.max(scores) + 10.0

    srange = np.arange(n_samples)
    crange = np.arange(cutoff)

    weighted_labels = labels[cutoff_sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = np.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    exp_scores = np.exp(scores).astype(np.float64)
    denom_per_rank = np.cumsum(exp_scores[sampled_rankings[:, ::-1]], axis=1)[
        :, : -cutoff - 1 : -1
    ]

    # second order
    result2 = np.zeros(n_docs, dtype=np.float64)
    np.add.at(
        result2,
        cutoff_sampled_rankings[:, :-1],
        cumsum_labels[:, 1:]
        * (
            1
            - exp_scores[cutoff_sampled_rankings[:, :-1]]
            * np.cumsum(1 / denom_per_rank[:, :-1], axis=1)
        ),
    )
    result2 /= n_samples

    cumsum_denom = np.cumsum(1 / denom_per_rank, axis=1)
    sum_prob_per_doc = exp_scores * cumsum_denom[:, -1, None]
    sum_prob_per_doc[srange[:, None], cutoff_sampled_rankings] = (
        exp_scores[cutoff_sampled_rankings] * cumsum_denom
    )

    in_or_not = np.zeros((n_samples, n_docs), dtype=np.float64)
    in_or_not[srange[:, None], cutoff_sampled_rankings] = 1
    long_item = in_or_not + 1 - sum_prob_per_doc

    # DR
    cumsum_weight_denom = np.cumsum(rank_weights[:cutoff] / denom_per_rank, axis=1)
    # RI
    cumsum_reward_denom = np.cumsum(cumsum_labels / denom_per_rank, axis=1)

    if cutoff < n_docs:
        second_part = -exp_scores[None, :] * cumsum_reward_denom[:, -1, None]
        second_part[:, relevant_docs] += (
            labels[relevant_docs][None, :]
            * exp_scores[None, relevant_docs]
            * cumsum_weight_denom[:, -1, None]
        )
    else:
        second_part = np.empty((n_samples, n_docs), dtype=np.float64)

    sampled_direct_reward = (
        labels[cutoff_sampled_rankings]
        * exp_scores[cutoff_sampled_rankings]
        * cumsum_weight_denom
    )
    sampled_following_reward = exp_scores[cutoff_sampled_rankings] * cumsum_reward_denom
    second_part[srange[:, None], cutoff_sampled_rankings] = (
        sampled_direct_reward - sampled_following_reward
    )

    # DR
    cumsum_weight_denom_square = np.cumsum(
        rank_weights[:cutoff] / denom_per_rank ** 2, axis=1
    )
    # RI
    cumsum_reward_denom_square = np.cumsum(cumsum_labels / denom_per_rank ** 2, axis=1)

    if cutoff < n_docs:
        third_part = -exp_scores[None, :] ** 2 * cumsum_reward_denom_square[:, -1, None]
        third_part[:, relevant_docs] += (
            labels[relevant_docs][None, :]
            * exp_scores[None, relevant_docs] ** 2
            * cumsum_weight_denom_square[:, -1, None]
        )
    else:
        third_part = np.empty((n_samples, n_docs), dtype=np.float64)

    sampled_direct_reward_square = (
        labels[cutoff_sampled_rankings]
        * exp_scores[cutoff_sampled_rankings] ** 2
        * cumsum_weight_denom_square
    )
    sampled_following_reward_square = (
        exp_scores[cutoff_sampled_rankings] ** 2 * cumsum_reward_denom_square
    )
    third_part[srange[:, None], cutoff_sampled_rankings] = (
        sampled_direct_reward_square - sampled_following_reward_square
    )

    return -(
        result2 + np.mean(second_part * long_item, axis=0) - np.mean(third_part, axis=0)
    )

def plrank3obj(preds, dtrain):
    group_ptr = dtrain.get_uint_info("group_ptr")
    labels = dtrain.get_label()
    

    # number of rankings
    n_samples = 100

    grad = np.zeros(len(labels), dtype=np.float64)
    hess = np.zeros(len(labels), dtype=np.float64)

    group = np.diff(group_ptr)
    max_query_size = max(group)
    longest_metric_weights = 1.0 / np.log2(np.arange(max_query_size) + 2)

    # number of docs to display
    cutoff = int(sys.argv[1])

    max_ranking_size = np.min((cutoff, max_query_size))
    metric_weights = longest_metric_weights[:max_ranking_size]

    for q in range(len(group_ptr) - 1):
        q_l = labels[group_ptr[q] : group_ptr[q + 1]]
        scores = preds[group_ptr[q] : group_ptr[q + 1]]

        # first order
        grad[group_ptr[q] : group_ptr[q + 1]] = PL_rank_3_grad(
            metric_weights, q_l, scores.astype(np.float64), n_samples
        )
        # second order
        hess[group_ptr[q] : group_ptr[q + 1]] = PL_rank_3_hess(
            metric_weights, q_l, scores.astype(np.float64), n_samples
        )

    return grad, hess

def dcg_at_k(rel, k):
    rel = np.asfarray(rel)[:k]
    if rel.size:
        return np.sum((2**rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return 0.0

def ideal_metrics(dtrain, k):
    group_ptr = dtrain.get_uint_info("group_ptr")
    labels = dtrain.get_label()
    idcg_results = []

    for q in range(len(group_ptr) - 1):
        relevance_labels = labels[group_ptr[q] : group_ptr[q + 1]]
        
        if np.any(np.greater(relevance_labels, 0)):
            idcg_results.append(dcg_at_k(sorted(relevance_labels, reverse=True), k))

    return np.mean(np.array(idcg_results))

def ndcg_dataset(preds, dtrain):
    group_ptr = dtrain.get_uint_info("group_ptr")
    labels = dtrain.get_label()
    results = []
    k_value = int(sys.argv[1])
    idcg = ideal_metrics(dtrain, k_value)

    for q in range(len(group_ptr) - 1):
        relevance_labels = labels[group_ptr[q] : group_ptr[q + 1]]
        document_scores = preds[group_ptr[q] : group_ptr[q + 1]]
        
        if np.any(np.greater(relevance_labels, 0)):
            document_data = list(zip(document_scores, relevance_labels))
            document_data.sort(reverse=True, key=lambda x: x[0])
            sorted_relevance = [item[1] for item in document_data]
            results.append(dcg_at_k(sorted_relevance, k_value) / idcg)

    return "Minus_NDCG@{}".format(int(sys.argv[1])), -float(np.mean(np.array(results)))


# Define hyperparameter search space

def objective(trial):
    params = {
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1e-2, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }  # Search space
    bst = xgb.train(
        params,
        xgb_train,
        num_boost_round=200,
        evals=[(xgb_train, "train"), (xgb_vali, "vali")],
        obj=plrank3obj,
        custom_metric=ndcg_dataset,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    return bst.best_score


# Run hyperparameter search
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, timeout=12*60*60)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  NDCG@{}: {}".format(int(sys.argv[1]), -trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
