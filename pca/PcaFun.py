from loader.DataSet import DataSet
from sklearn.decomposition import PCA
from knn.knn import KnnClassifier
from pylab import *


class PcaFun(object):
    def __init__(self, data_set):
        self._data_set = data_set
        self._shape = self._data_set.shape
        train_data_size = len(self._data_set.y_train)
        test_data_size = len(self._data_set.y_test)
        # train_data_size = 1000
        # test_data_size = 100
        self._train_subset_size = train_data_size
        self._test_subset_size = test_data_size
        self._x_train = self._data_set.x_train[:self._train_subset_size]
        self._y_train = self._data_set.y_train[:self._train_subset_size]
        self._x_test = self._data_set.x_test[:self._test_subset_size]
        self._y_test = self._data_set.y_test[:self._test_subset_size]

        self._pca = None

    def calculate_pca_for_train(self):
        self._pca = PCA()
        self._pca.fit(self._x_train)

    def _reverse_transform(self, pc_data):
        pca = self._pca
        data_original = np.dot(pc_data, pca.components_) + pca.mean_
        return data_original

    def sanity_restore_data(self):
        pca = self._pca
        train_pc = pca.transform(self._x_train)
        data_original = self._reverse_transform(train_pc)
        diff = data_original - self._x_train
        eps = 1e-5
        diff_numeric = np.abs(diff) < eps
        assert np.all(diff_numeric)

    def draw_average_digit(self):
        # doesn't matter on which coordinate system the mean is calculated.
        pca = self._pca
        train_pc = pca.transform(self._x_train)
        avg_digit_pc = np.mean(train_pc, axis=0)
        avg_digit_restored = self._reverse_transform(avg_digit_pc)
        avg_digit_original = np.mean(self._x_train, axis=0)
        abs_diff = np.abs(avg_digit_original - avg_digit_restored)
        eps = 1e-5
        diff_numeric = abs_diff < eps
        assert all(diff_numeric)
        plt.figure()
        plt.title('average component')
        plt.imshow(np.reshape(avg_digit_original, self._shape), cmap='gray', interpolation='nearest')
        plt.show(block=True)

    def draw_first_k_components(self, k, rows, cols):
        pca = self._pca
        first_k_pc_restored = pca.components_[:k]  # components are already in feature space by sklearn

        assert cols * rows == k
        fig, plots = plt.subplots(nrows=rows, ncols=cols)
        for i, row_axes in enumerate(plots):
            for j, ax in enumerate(row_axes):
                plot_index = i * cols + j
                xticks([]), yticks([])
                image = first_k_pc_restored[plot_index].reshape(self._shape)

                ax.imshow(image, cmap='gray', interpolation='nearest')
                ax.set_title(f"component #{plot_index}")
                ax.axis('off')
        plt.subplots_adjust(left=None, bottom=.1, right=None, top=0.95, wspace=None, hspace=None)
        plt.show()

    def draw_cumulative_variance_graph(self):
        pca = self._pca
        explained_variance_ratio = pca.explained_variance_ratio_
        plt.figure()
        plt.title('cumulative explained variance ratio')
        plt.xlabel('#components')
        plt.ylabel('cumulative explained_variance_ratio')
        plt.plot(np.cumsum(explained_variance_ratio))
        plt.show(block=True)

    def calculate_n_components_required(self, desired_variance_ratio):
        pca = self._pca
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
        cutoff_ind = np.searchsorted(cumulative_explained_variance_ratio, desired_variance_ratio) + 1
        return cutoff_ind

    def reduce_dimensionality(self, data, dim):
        pca = self._pca
        reduced_components = pca.components_.T[:, :dim]
        reduced_data = np.dot(data - pca.mean_, reduced_components)

        # # same result, but slower due to higher dimension
        # transformed = pca.transform(data)
        # reduced_data = transformed[:, :dim]
        # diff = reduced_data - transformed_manually
        # assert np.all(diff < 1e-5)
        return reduced_data

    def reduce_dimensionality_to_2_and_plot(self):
        train_reduced = self.reduce_dimensionality(self._x_train, dim=2)
        train_pc_x_component = train_reduced[:, 0]
        train_pc_y_component = train_reduced[:, 1]

        labels = self._y_train

        plt.figure()
        plt.title('dimensionality reduction to 2')

        for label in np.unique(labels):
            label_ind = np.where(labels == label)
            plt.scatter(train_pc_x_component[label_ind], train_pc_y_component[label_ind],
                        marker='.',
                        c=np.random.rand(3,))
        plt.show(block=True)

    def reduce_dimensionality_and_classify_by_knn(self, n_dim, max_k_neighbours):
        reduced_data_set = DataSet(None)
        reduced_data_set.x_train = self.reduce_dimensionality(self._x_train, dim=n_dim)
        reduced_data_set.x_test = self.reduce_dimensionality(self._x_test, dim=n_dim)
        reduced_data_set.x_validation = self.reduce_dimensionality(self._data_set.x_validation, dim=n_dim)
        reduced_data_set.y_train = self._y_train
        reduced_data_set.y_validation = self._data_set.y_validation
        reduced_data_set.y_test = self._y_test

        knn = KnnClassifier(reduced_data_set)
        knn.find_best_k(max_k=max_k_neighbours)


if __name__ == "__main__":
    data_set = DataSet('../mnist.pkl.gz')
    pca_fun = PcaFun(data_set)
    pca_fun.calculate_pca_for_train()
    pca_fun.sanity_restore_data()
    pca_fun.draw_average_digit()
    pca_fun.draw_first_k_components(k=6, rows=2, cols=3)
    pca_fun.draw_cumulative_variance_graph()
    cutoff_for_80_percent = pca_fun.calculate_n_components_required(desired_variance_ratio=0.8)
    cutoff_for_95_percent = pca_fun.calculate_n_components_required(desired_variance_ratio=0.95)
    print(f"cutoff_for_80_percent: {cutoff_for_80_percent}")
    print(f"cutoff_for_95_percent: {cutoff_for_95_percent}")
    pca_fun.reduce_dimensionality_to_2_and_plot()
    pca_fun.reduce_dimensionality_and_classify_by_knn(n_dim=2, max_k_neighbours=10)
