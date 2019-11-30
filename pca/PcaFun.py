from loader.DataSet import DataSet
from sklearn.decomposition import PCA
from pylab import *


class PcaFun(object):
    def __init__(self, data_path):
        self._data_path = data_path
        self._data_set = DataSet(self._data_path)
        self._shape = self._data_set.shape
        # train_data_size = len(self._data_set.y_train)
        # test_data_size = len(self._data_set.y_test)
        train_data_size = 1000
        test_data_size = 100
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
        plt.imshow(np.reshape(avg_digit_original, self._shape), cmap='gray', interpolation='nearest')
        plt.show(block=True)

    def draw_first_k_components(self, k, rows, cols):
        pca = self._pca
        first_k_pc_restored = self._reverse_transform(pca.components_[:k])

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



# explained_variance_ratio = pca.explained_variance_ratio_

if __name__ == "__main__":
    pca_fun = PcaFun(data_path='../mnist.pkl.gz')
    pca_fun.calculate_pca_for_train()
    pca_fun.sanity_restore_data()
    pca_fun.draw_average_digit()
    pca_fun.draw_first_k_components(k=6, rows=2, cols=3)