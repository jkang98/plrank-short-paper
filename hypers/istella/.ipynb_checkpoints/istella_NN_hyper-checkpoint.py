import json
import os.path
import sys
import time

import numpy as np
import optuna
import torch
import torch.optim as optim
from torch import nn

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
    

def cutoff_ranking(scores, cutoff, invert=False):
    n_docs = scores.shape[0]
    cutoff = min(n_docs, cutoff)
    full_partition = np.argpartition(scores, cutoff - 1)
    partition = full_partition[:cutoff]
    sorted_partition = np.argsort(scores[partition])
    ranked_partition = partition[sorted_partition]
    if not invert:
        return ranked_partition
    else:
        full_partition[:cutoff] = ranked_partition
        inverted = np.empty(n_docs, dtype=ranked_partition.dtype)
        inverted[full_partition] = np.arange(n_docs)
        return ranked_partition, inverted
    

def ideal_metrics(data_split, rank_weights, labels):
    cutoff = rank_weights.size
    result = np.zeros(data_split.num_queries())
    for qid in range(data_split.num_queries()):
        q_labels = data_split.query_values_from_vector(qid, labels)
        ranking = cutoff_ranking(-q_labels, cutoff)
        result[qid] = np.sum(rank_weights[: ranking.size] * q_labels[ranking])
    return result


def compute_results(data_split, model, rank_weights, labels, ideal_metrics):
    scores = model(torch.from_numpy(data_split.feature_matrix))[:, 0].detach().numpy()

    return compute_results_from_scores(
        data_split, scores, rank_weights, labels, ideal_metrics
    )


def evaluate_max_likelihood(data_split, scores, rank_weights, labels, ideal_metrics):
    cutoff = rank_weights.size
    result = np.zeros(data_split.num_queries())
    query_normalized_result = np.zeros(data_split.num_queries())
    for qid in range(data_split.num_queries()):
        q_scores = data_split.query_values_from_vector(qid, scores)
        q_labels = data_split.query_values_from_vector(qid, labels)
        ranking = cutoff_ranking(-q_scores, cutoff)
        q_result = np.sum(rank_weights[: ranking.size] * q_labels[ranking])
        if ideal_metrics[qid] == 0:
            query_normalized_result[qid] = 0.0
        else:
            query_normalized_result[qid] = q_result / ideal_metrics[qid]
        result[qid] = q_result / np.mean(ideal_metrics)
    return float(np.mean(query_normalized_result)), float(np.mean(result))


def compute_results_from_scores(
    data_split, scores, rank_weights, labels, ideal_metrics
):
    QN_ML, N_ML = evaluate_max_likelihood(
        data_split, scores, rank_weights, labels, ideal_metrics
    )

    return {
        "query normalized maximum likelihood": QN_ML,
        "dataset normalized maximum likelihood": N_ML,
    }

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []

    in_features = 220
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features, dtype=torch.float64))
        layers.append(nn.Sigmoid())
        in_features = out_features
    layers.append(nn.Linear(in_features, 1, dtype=torch.float64))

    return nn.Sequential(*layers)



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


def PL_rank_3(rank_weights, labels, scores, n_samples):
    n_docs = labels.shape[0]
    cutoff = min(rank_weights.shape[0], n_docs)

    sampled_rankings = gumbel_sample_rankings(
        scores, n_samples, cutoff=cutoff, return_full_rankings=True
    )

    cutoff_sampled_rankings = sampled_rankings[:, :cutoff]

    scores = scores.copy() - np.amax(scores) + 10.0

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

    return result1 + np.mean(second_part, axis=0)


cutoff = int(sys.argv[1])
num_samples = 200


data = get_dataset_from_json_info("istella", "local_dataset_info.txt")
fold_id = (1 - 1) % data.num_folds()
data = data.get_data_folds()[fold_id]

data.read_data()

max_ranking_size = np.min((cutoff, data.max_query_size()))

longest_possible_metric_weights = 1.0 / np.log2(np.arange(data.max_query_size()) + 2)
metric_weights = longest_possible_metric_weights[:max_ranking_size]
train_labels = 2 ** data.train.label_vector - 1
vali_labels = 2 ** data.validation.label_vector - 1
test_labels = 2 ** data.test.label_vector - 1
ideal_train_metrics = ideal_metrics(data.train, metric_weights, train_labels)
ideal_vali_metrics = ideal_metrics(data.validation, metric_weights, vali_labels)
ideal_test_metrics = ideal_metrics(data.test, metric_weights, test_labels)

n_queries = data.train.num_queries()


def objective(trial):
    # Generate the model.
    model = define_model(trial)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    

    # Training of the model.
    n_epochs = 200
    best_score= 0
    patience = 20
    counter = 0
    for epoch_i in range(n_epochs):
        last_method_train_time = time.time()
        query_permutation = np.random.permutation(n_queries)
        model.train()
        for batch_i in range(int(np.ceil(n_queries/batch_size))):
            batch_queries = query_permutation[batch_i*batch_size:(batch_i+1)*batch_size]
            cur_batch_size = batch_queries.shape[0]
            batch_ranges = np.zeros(cur_batch_size+1, dtype=np.int64)
            batch_features = [data.train.query_feat(batch_queries[0])]
            batch_ranges[1] = batch_features[0].shape[0]
            for i in range(1, cur_batch_size):
                batch_features.append(data.train.query_feat(batch_queries[i]))
                batch_ranges[i+1] = batch_ranges[i] + batch_features[i].shape[0]
            batch_features = torch.from_numpy(np.concatenate(batch_features, axis=0))

            batch_tf_scores = model(batch_features)
            loss = 0
            batch_doc_weights = np.zeros(batch_features.shape[0], dtype=np.float64)

            for i, qid in enumerate(batch_queries):
                q_labels =  data.train.query_values_from_vector(
                                      qid, train_labels)
                q_feat = batch_features[batch_ranges[i]:batch_ranges[i+1],:]
                q_ideal_metric = ideal_train_metrics[qid]

                if q_ideal_metric != 0:
                    q_metric_weights = metric_weights  # /q_ideal_metric #uncomment for NDCG
                    q_tf_scores = model(q_feat)

                    q_np_scores = q_tf_scores.detach().numpy()[:, 0]

                    doc_weights = PL_rank_3(
                        q_metric_weights, q_labels, q_np_scores, n_samples=num_samples
                    )
                    batch_doc_weights[batch_ranges[i]:batch_ranges[i+1]] = doc_weights

            loss = -torch.sum(batch_tf_scores[:,0] * torch.from_numpy(batch_doc_weights))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        with torch.no_grad():
            vali_result = compute_results(
                data.validation, model, metric_weights, vali_labels, ideal_vali_metrics,
            )
        if best_score < vali_result["dataset normalized maximum likelihood"]:
            best_score = vali_result["dataset normalized maximum likelihood"]
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                return best_score
        
    return best_score


# Run hyperparameter search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=12*60*60)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  NDCG@{}: {}".format(int(sys.argv[1]), trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
