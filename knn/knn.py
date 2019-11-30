from loader.DataSet import DataSet
from scipy.spatial.distance import cdist
import numpy as np


class KnnClassifier(object):
    def __init__(self, data_path):
        self._data_path = data_path
        self._data_set = DataSet(self._data_path)
        self._x_train = self._data_set.x_train
        self._y_train = self._data_set.y_train
        self._x_test = self._data_set.x_test
        self._y_test = self._data_set.y_test

    def classify_test_by_train(self, k):
        only_check_for_x_first = 100
        only_use_first_x_from_test = 1000
        x_test = self._x_test[:only_check_for_x_first]
        x_train = self._x_train[:only_use_first_x_from_test]
        pairs_distances = cdist(x_test, x_train)
        # k_min_distances_inds_per_test_sample = np.argpartition(pairs_distances, k, axis=0)[:, :k]
        sorted_distances_ind = np.argsort(pairs_distances, axis=1)
        sorted_distances = pairs_distances[:, sorted_distances_ind]
        k_min_distances_per_test_sample = pairs_distances[k_min_distances_inds_per_test_sample]

        votes_per_test_sample = self._y_train[k_min_distances_inds_per_test_sample]
        classes, votes_count = np.unique(votes_per_test_sample, return_counts=True, axis=0)
        votes_count[classes].reshape(votes_per_test_sample.shape)
        hi=5


if __name__ == "__main__":
    def main():
        classifier = KnnClassifier(data_path='../mnist.pkl.gz')
        classifier.classify_test_by_train(k=5)

    main()
