import json
import os.path
import sys
import time

import numpy as np
import optuna
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from xgboost import DMatrix

FOLDDATA_WRITE_VERSION = 4


def _add_zero_to_vector(vector):
    return np.concatenate([np.zeros(1, dtype=vector.dtype), vector])


def get_dataset_from_json_info(
    dataset_name, info_path,
):
    with open(info_path) as f:
        all_info = json.load(f)
    assert dataset_name in all_info, "Dataset: %s not found in info file: %s" % (
        dataset_name,
        all_info.keys(),
    )

    set_info = all_info[dataset_name]
    assert set_info["num_folds"] == len(set_info["fold_paths"]), (
        "Missing fold paths for %s" % dataset_name
    )

    num_feat = set_info["num_nonzero_feat"]

    return DataSet(
        dataset_name,
        set_info["fold_paths"],
        set_info["num_relevance_labels"],
        num_feat,
        set_info["num_nonzero_feat"],
    )


class DataSet(object):

    """
    Class designed to manage meta-data for datasets.
    """

    def __init__(
        self,
        name,
        data_paths,
        num_rel_labels,
        num_features,
        num_nonzero_feat,
        store_pickle_after_read=True,
        read_from_pickle=True,
        feature_normalization=True,
        purge_test_set=True,
    ):
        self.name = name
        self.num_rel_labels = num_rel_labels
        self.num_features = num_features
        self.data_paths = data_paths
        self.store_pickle_after_read = store_pickle_after_read
        self.read_from_pickle = read_from_pickle
        self.feature_normalization = feature_normalization
        self.purge_test_set = purge_test_set
        self._num_nonzero_feat = num_nonzero_feat

    def num_folds(self):
        return len(self.data_paths)

    def get_data_folds(self):
        return [DataFold(self, i, path) for i, path in enumerate(self.data_paths)]


class DataFold(object):
    def __init__(self, dataset, fold_num, data_path):
        self.name = dataset.name
        self.num_rel_labels = dataset.num_rel_labels
        self.num_features = dataset.num_features
        self.fold_num = fold_num
        self.data_path = data_path
        self._data_ready = False
        self.store_pickle_after_read = dataset.store_pickle_after_read
        self.read_from_pickle = dataset.read_from_pickle
        self.feature_normalization = dataset.feature_normalization
        self.purge_test_set = dataset.purge_test_set
        self._num_nonzero_feat = dataset._num_nonzero_feat

    def max_query_size(self):
        return np.amax(
            (
                self.train.max_query_size(),
                self.validation.max_query_size(),
                self.test.max_query_size(),
            ),
        )

    def data_ready(self):
        return self._data_ready

    def _read_file(self, path, feat_map, purge):
        """
        Read letor file.
        """
        queries = []
        cur_docs = []
        cur_labels = []
        current_qid = None

        for line in open(path, "r"):
            info = line[: line.find("#")].split()
            qid = info[1].split(":")[1]
            label = int(info[0])
            feat_pairs = info[2:]

            if current_qid is None:
                current_qid = qid
            elif current_qid != qid:
                stacked_documents = np.stack(cur_docs, axis=0)
                if self.feature_normalization:
                    stacked_documents -= np.amin(stacked_documents, axis=0)[None, :]
                    safe_max = np.amax(stacked_documents, axis=0)
                    safe_max[safe_max == 0] = 1.0
                    stacked_documents /= safe_max[None, :]

                np_labels = np.array(cur_labels, dtype=np.int64)
                if not purge or np.any(np.greater(np_labels, 0)):
                    queries.append(
                        {
                            "qid": current_qid,
                            "n_docs": stacked_documents.shape[0],
                            "labels": np_labels,
                            "documents": stacked_documents,
                        }
                    )
                current_qid = qid
                cur_docs = []
                cur_labels = []

            doc_feat = np.zeros(self._num_nonzero_feat)
            for pair in feat_pairs:
                feat_id, feature = pair.split(":")
                feat_id = int(feat_id)
                feat_value = float(feature)
                if feat_id not in feat_map:
                    feat_map[feat_id] = len(feat_map)
                    assert feat_map[feat_id] < self._num_nonzero_feat, (
                        "%s features found but %s expected"
                        % (feat_map[feat_id], self._num_nonzero_feat)
                    )
                doc_feat[feat_map[feat_id]] = feat_value

            cur_docs.append(doc_feat)
            cur_labels.append(label)

        stacked_documents = np.stack(cur_docs, axis=0)
        if self.feature_normalization:
            stacked_documents -= np.amin(stacked_documents, axis=0)[None, :]
            safe_max = np.amax(stacked_documents, axis=0)
            safe_max[safe_max == 0] = 1.0
            stacked_documents /= safe_max[None, :]

        np_labels = np.array(cur_labels, dtype=np.int64)
        if not purge or np.any(np.greater(np_labels, 0)):
            queries.append(
                {
                    "qid": current_qid,
                    "n_docs": stacked_documents.shape[0],
                    "labels": np_labels,
                    "documents": stacked_documents,
                }
            )

        all_docs = np.concatenate([x["documents"] for x in queries], axis=0)
        all_n_docs = np.array([x["n_docs"] for x in queries], dtype=np.int64)
        all_labels = np.concatenate([x["labels"] for x in queries], axis=0)

        query_ranges = _add_zero_to_vector(np.cumsum(all_n_docs))

        return query_ranges, all_docs, all_labels

    def read_data(self):
        """
        Reads data from a fold folder (letor format).
        """
        data_read = False
        if self.feature_normalization and self.purge_test_set:
            pickle_name = "binarized_purged_querynorm.npz"
        elif self.feature_normalization:
            pickle_name = "binarized_querynorm.npz"
        elif self.purge_test_set:
            pickle_name = "binarized_purged.npz"
        else:
            pickle_name = "binarized.npz"

        pickle_path = self.data_path + pickle_name

        train_raw_path = self.data_path + "new_train.txt"
        valid_raw_path = self.data_path + "new_valid.txt"
        test_raw_path = self.data_path + "test.txt"

        if self.read_from_pickle and os.path.isfile(pickle_path):
            loaded_data = np.load(pickle_path, allow_pickle=True)
            if loaded_data["format_version"] == FOLDDATA_WRITE_VERSION:
                feature_map = loaded_data["feature_map"].item()
                train_feature_matrix = loaded_data["train_feature_matrix"]
                train_doclist_ranges = loaded_data["train_doclist_ranges"]
                train_label_vector = loaded_data["train_label_vector"]
                valid_feature_matrix = loaded_data["valid_feature_matrix"]
                valid_doclist_ranges = loaded_data["valid_doclist_ranges"]
                valid_label_vector = loaded_data["valid_label_vector"]
                test_feature_matrix = loaded_data["test_feature_matrix"]
                test_doclist_ranges = loaded_data["test_doclist_ranges"]
                test_label_vector = loaded_data["test_label_vector"]
                data_read = True
            del loaded_data

        if not data_read:
            feature_map = {}
            (
                train_doclist_ranges,
                train_feature_matrix,
                train_label_vector,
            ) = self._read_file(train_raw_path, feature_map, False)
            (
                valid_doclist_ranges,
                valid_feature_matrix,
                valid_label_vector,
            ) = self._read_file(valid_raw_path, feature_map, False)
            (
                test_doclist_ranges,
                test_feature_matrix,
                test_label_vector,
            ) = self._read_file(test_raw_path, feature_map, self.purge_test_set)

            assert len(feature_map) == self._num_nonzero_feat, (
                "%d non-zero features found but %d expected"
                % (len(feature_map), self._num_nonzero_feat,)
            )

            # sort found features so that feature id ascends
            sorted_map = sorted(feature_map.items())
            transform_ind = np.array([x[1] for x in sorted_map])

            train_feature_matrix = train_feature_matrix[:, transform_ind]
            valid_feature_matrix = valid_feature_matrix[:, transform_ind]
            test_feature_matrix = test_feature_matrix[:, transform_ind]

            feature_map = {}
            for i, x in enumerate([x[0] for x in sorted_map]):
                feature_map[x] = i

            if self.store_pickle_after_read:
                np.savez_compressed(
                    pickle_path,
                    format_version=FOLDDATA_WRITE_VERSION,
                    feature_map=feature_map,
                    train_feature_matrix=train_feature_matrix,
                    train_doclist_ranges=train_doclist_ranges,
                    train_label_vector=train_label_vector,
                    valid_feature_matrix=valid_feature_matrix,
                    valid_doclist_ranges=valid_doclist_ranges,
                    valid_label_vector=valid_label_vector,
                    test_feature_matrix=test_feature_matrix,
                    test_doclist_ranges=test_doclist_ranges,
                    test_label_vector=test_label_vector,
                )

        n_feat = len(feature_map)
        assert n_feat == self.num_features, "%d features found but %d expected" % (
            n_feat,
            self.num_features,
        )

        self.inverse_feature_map = feature_map
        self.feature_map = [
            x[0] for x in sorted(feature_map.items(), key=lambda x: x[1])
        ]
        self.train = DataFoldSplit(
            self,
            "train",
            train_doclist_ranges,
            train_feature_matrix,
            train_label_vector,
        )
        self.validation = DataFoldSplit(
            self,
            "validation",
            valid_doclist_ranges,
            valid_feature_matrix,
            valid_label_vector,
        )
        self.test = DataFoldSplit(
            self, "test", test_doclist_ranges, test_feature_matrix, test_label_vector
        )
        self._data_ready = True


class DataFoldSplit(object):
    def __init__(self, datafold, name, doclist_ranges, feature_matrix, label_vector):
        self.datafold = datafold
        self.name = name
        self.doclist_ranges = doclist_ranges
        self.feature_matrix = feature_matrix
        self.label_vector = label_vector

    def num_queries(self):
        return self.doclist_ranges.shape[0] - 1

    def num_docs(self):
        return self.feature_matrix.shape[0]

    def query_values_from_vector(self, qid, vector):
        s_i, e_i = self.query_range(qid)
        return vector[s_i:e_i]

    def query_range(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return s_i, e_i

    def query_size(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return e_i - s_i

    def query_sizes(self):
        return self.doclist_ranges[1:] - self.doclist_ranges[:-1]

    def max_query_size(self):
        return np.amax(self.query_sizes())

    def query_labels(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return self.label_vector[s_i:e_i]

    def query_feat(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return self.feature_matrix[s_i:e_i, :]

    def doc_feat(self, query_index, doc_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        assert s_i + doc_index < self.doclist_ranges[query_index + 1]
        return self.feature_matrix[s_i + doc_index, :]

    def doc_str(self, query_index, doc_index):
        doc_feat = self.doc_feat(query_index, doc_index)
        feat_i = np.where(doc_feat)[0]
        doc_str = ""
        for f_i in feat_i:
            doc_str += "%s:%f " % (self.datafold.feature_map[f_i], doc_feat[f_i])
        return doc_str


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
    labels = 2 ** labels - 1
    

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
        document_data = list(zip(document_scores, relevance_labels))
        document_data.sort(reverse=True, key=lambda x: x[0])
        sorted_relevance = [item[1] for item in document_data]
        results.append(dcg_at_k(sorted_relevance, k_value) / idcg)

    return "Minus_NDCG@{}".format(int(sys.argv[1])), -float(np.mean(np.array(results)))




data = get_dataset_from_json_info("istella", "local_dataset_info.txt")
fold_id = (1 - 1) % data.num_folds()
data = data.get_data_folds()[fold_id]

data.read_data()

from scipy.sparse import csr_matrix

train_n_queries = data.train.num_queries()
train_array = np.concatenate(
    [data.train.query_feat(i) for i in range(train_n_queries)], axis=0
)
train_sparse_matrix = csr_matrix(train_array)
train_labels = data.train.label_vector
new_train = DMatrix(train_sparse_matrix, train_labels)
new_train.set_group([data.train.query_feat(i).shape[0] for i in range(train_n_queries)])

test_n_queries = data.test.num_queries()
test_array = np.concatenate([data.test.query_feat(i) for i in range(test_n_queries)], axis=0)
test_sparse_matrix = csr_matrix(test_array)
test_labels = data.test.label_vector
new_test = DMatrix(test_sparse_matrix, test_labels)
new_test.set_group([data.test.query_feat(i).shape[0] for i in range(test_n_queries)])

validation_n_queries = data.validation.num_queries()
validation_array = np.concatenate(
    [data.validation.query_feat(i) for i in range(validation_n_queries)], axis=0
)
validation_sparse_matrix = csr_matrix(validation_array)
validation_labels = data.validation.label_vector
new_valid = DMatrix(validation_sparse_matrix, validation_labels)
new_valid.set_group([data.validation.query_feat(i).shape[0] for i in range(validation_n_queries)])


# Define hyperparameter search space

def objective(trial):
    params = {
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }  # Search space
    bst = xgb.train(
        params,
        new_train,
        num_boost_round=30,
        evals=[(new_train, "train"), (new_valid, "vali")],
        obj=plrank3obj,
        custom_metric=ndcg_dataset,
        #early_stopping_rounds=2,
        verbose_eval=False,
    )

    return bst.best_score


# Run hyperparameter search
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, timeout=2*60*60)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  NDCG@{}: {}".format(int(sys.argv[1]), -trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
