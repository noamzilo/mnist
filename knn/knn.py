from loader.DataSet import DataSet
from scipy.spatial.distance import cdist
import numpy as np
from utils.take_k_along_axis import take_smallest_along_axis
from utils.take_k_along_axis import take_smallest_indices_along_axis
from utils.bincount_vectorized import bin_count_2d_vectorized
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


class KnnClassifier(object):
    def __init__(self, data_set):
        self._data_set = data_set
        train_data_size = len(self._data_set.y_train)
        test_data_size = len(self._data_set.y_test)
        # train_data_size = 10000
        # test_data_size = 1000
        self._train_subset_size = train_data_size
        self._test_subset_size = test_data_size
        self._x_train = self._data_set.x_train[:self._train_subset_size]
        self._y_train = self._data_set.y_train[:self._train_subset_size]
        self._x_test = self._data_set.x_test[:self._test_subset_size]
        self._y_test = self._data_set.y_test[:self._test_subset_size]
        self._pairs_distances = None
        self._num_classes = np.max(self._y_train) + 1 # should use len(unique) but this is faster assuming the data is good
        self._y_predict = None
        self._confusion_matrix = None
        self._report = None

    def _calculate_pairs_distances(self):
        # allows for fast consecutive calls for classification by different k
        x_test = self._x_test
        x_train = self._x_train
        if self._pairs_distances is None:
            self._pairs_distances = cdist(x_test, x_train)

    def classify_test_by_train(self, k):
        self._calculate_pairs_distances()
        pairs_distances = self._pairs_distances
        k_min_distances_inds_per_test_sample = take_smallest_indices_along_axis(pairs_distances, n=k, axis=1)
        # k_min_distances = take_smallest_along_axis(pairs_distances, n=k, axis=1)
        votes_per_test_sample = self._y_train[k_min_distances_inds_per_test_sample]
        vote_counts_per_test_sample = bin_count_2d_vectorized(votes_per_test_sample)
        self._y_predict = np.argmax(vote_counts_per_test_sample, axis=1)
        # ties are broken using np's tie breaker: select 1st.

    def calculate_performance(self):
        self._confusion_matrix = confusion_matrix(self._y_test, self._y_predict)
        self._report = classification_report(self._y_test, self._y_predict)
        n_accurate = np.trace(self._confusion_matrix)
        n_total = np.sum(self._confusion_matrix)
        accuracy_rate = n_accurate / n_total
        return accuracy_rate

    def classify_using_sklearn(self, k):
        # This is used to verify my solution. mine is better for running cdist only once for all k
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(self._x_train, self._y_train)
        self._y_predict = classifier.predict(self._x_test)

    def find_best_k(self, max_k=10):
        accuracies = []
        for k in range(1, max_k + 1):
            self.classify_test_by_train(k)
            accuracy = self.calculate_performance()
            accuracies.append(accuracy)
        plt.plot(np.array(range(1, max_k + 1)), accuracies)
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.title(f'accuracy(k), #dimensions={self._x_train.shape[1]}')
        plt.show(block=True)
        best_k = np.argmax(accuracies) + 1
        print(f"best k is {best_k}")
        return best_k

if __name__ == "__main__":
    def main():
        data_set = DataSet(path='../mnist.pkl.gz')
        classifier = KnnClassifier(data_set)
        classifier.find_best_k()




    main()
