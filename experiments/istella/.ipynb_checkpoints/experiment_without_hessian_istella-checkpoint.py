import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from xgboost import DMatrix
import numpy as np
import optuna
import sys
import time

#  This script demonstrate how to do ranking with xgboost.train
x_train, y_train = load_svmlight_file("istella-letor/full/istella.train")
x_valid, y_valid = load_svmlight_file("istella-letor/full/istella.valid")
x_test, y_test = load_svmlight_file("istella-letor/full/istella.test")

group_train = []
with open("istella-letor/full/istella.train.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_train.append(int(line.split("\n")[0]))

group_valid = []
with open("istella-letor/full/istella.valid.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_valid.append(int(line.split("\n")[0]))

group_test = []
with open("istella-letor/full/istella.test.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_test.append(int(line.split("\n")[0]))

xgb_train = DMatrix(x_train, y_train)
xgb_vali = DMatrix(x_valid, y_valid)
xgb_test = DMatrix(x_test, y_test)

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
    # second order
    result2 = np.ones(n_docs, dtype=np.float64)

    return result2

def plrank3obj(preds, dtrain):
    group_ptr = dtrain.get_uint_info("group_ptr")
    labels = dtrain.get_label()
    

    # number of rankings
    n_samples = 200

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

    return "NDCG@{}".format(int(sys.argv[1])), float(np.mean(np.array(results)))


# Custom callback to record running time for each round
class TimingCallback(xgb.callback.TrainingCallback):
    def before_training(self, model):
        self.results = []
        return model
        
    def after_iteration(self, model, epoch, evals_log):
        elapsed_time = time.time() - start_time
        round_time = {
            'iteration': epoch + 1,
            'time': elapsed_time
        }
        self.results.append(round_time)
        return False

# plrank3
params = {
    "verbosity": 0,
    "learning_rate": 0.00023720886373434174,
    "max_depth": 8,
    "min_child_weight": 6,
    "gamma": 1.8566796123426632e-07,
    "lambda": 1.3565887579555225e-06,
    "alpha": 2.064432745653516e-06,
    "disable_default_eval_metric": 1,
}

timing_callback = TimingCallback()
ndcg_result = {}

start_time = time.time()
model = xgb.train(
    params,
    xgb_train,
    num_boost_round=400,
    evals=[(xgb_test, "test")],
    obj=plrank3obj,
    custom_metric=ndcg_dataset,
    evals_result=ndcg_result,
    verbose_eval=False,
    callbacks=[timing_callback]
)

print(ndcg_result)
print(timing_callback.results)