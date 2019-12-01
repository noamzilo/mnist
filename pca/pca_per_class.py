from loader.DataSet import DataSet
from sklearn.decomposition import PCA
from pylab import *


class PcaPerClass(object):
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

        self._x_pred_pc = None
        self._x_pred_restored = None
        self._classes = None
        self._n_classes = None

        self._class_pca_models = []

    def calculate_pca_for_all_classes(self):
        self._classes = np.unique(self._y_train)
        self._n_classes = len(self._classes)
        self._class_pca_models = []
        for c in self._classes:
            class_inds = np.argwhere(c == self._y_train)[:, 0]
            x_class_train = self._x_train[class_inds]

            class_pca = PCA()
            class_pca.fit(x_class_train)
            self._class_pca_models.append(class_pca)

    def draw_first_k_components(self, k, rows, cols):
        for pca in self._class_pca_models:
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

    def project_test_set_to_all_models(self, dim):
        self._x_pred_pc = np.zeros((len(self._class_pca_models), self._x_test.shape[0], dim))
        data = self._x_test
        for i, pca in enumerate(self._class_pca_models):
            class_components = pca.components_.T[:, :dim]
            self._x_pred_pc[i, :, :] = np.dot(data - pca.mean_, class_components)

    def restore_pred_to_all_models(self, dim):
        self._x_pred_restored = np.zeros((len(self._class_pca_models), self._x_test.shape[0], self._x_test.shape[1]))
        for i, pca in enumerate(self._class_pca_models):
            x_pred_pc = self._x_pred_pc[i, :, :]
            self._x_pred_restored[i, :, :] = np.dot(x_pred_pc, pca.components_[:dim, :]) + pca.mean_
        # plt.figure()
        # plt.imshow(np.reshape(self._x_pred_restored[1, 100, :], self._shape))
        # plt.show(block=True)
        # hi=5

    def draw_sample_restored_image(self, index):
        assert 0 <= index < self._x_test.shape[0]
        images = self._x_pred_restored[:, index, :]

        cols = self._n_classes // 2
        rows = 2
        fig, plots = plt.subplots(nrows=rows, ncols=cols)
        for i, row_axes in enumerate(plots):
            for j, ax in enumerate(row_axes):
                plot_index = i * cols + j
                xticks([]), yticks([])
                image = images[plot_index, :].reshape(self._shape)

                ax.imshow(image, cmap='gray', interpolation='nearest')
                ax.set_title(f"model of #{plot_index}")
                ax.axis('off')
        plt.subplots_adjust(left=None, bottom=.1, right=None, top=0.95, wspace=None, hspace=None)
        plt.show()

    def calculate_distances_to_original(self, index):
        distances = np.zeros(self._n_classes)
        for i in range(self._n_classes):
            image = self._x_pred_restored[i, index, :]
            diff = np.abs(image - self._x_test[index, :])
            diff2 = diff ** 2
            sum_diff2 = np.sum(diff2)
            distances[i] = sum_diff2
        print(distances)
        best_model_ind = np.argmin(distances)
        print(f"best_model: {best_model_ind}")

    def calculate_all_distances_to_original(self):
        distances = np.zeros((self._x_test.shape[0], self._n_classes))
        for i in range(self._n_classes):
            images = self._x_pred_restored[i, :, :]
            diff = np.abs(images - self._x_test)
            diff2 = diff ** 2
            sum_diff2 = np.sum(diff2, axis=1)
            distances[:, i] = sum_diff2
        best_model_inds = np.argmin(distances, axis=1)

        correct_test_samples = np.equal(best_model_inds, self._y_test)
        ratio = np.sum(correct_test_samples) / len(correct_test_samples)
        print(f"different pca successful classification: {ratio}")

if __name__ == "__main__":
    def main():
        reduce_to_dim = 6
        data_set = DataSet('../mnist.pkl.gz')
        pca_per_class = PcaPerClass(data_set)
        pca_per_class.calculate_pca_for_all_classes()
        # pca_per_class.draw_first_k_components(k=6, rows=2, cols=3)
        pca_per_class.project_test_set_to_all_models(dim=reduce_to_dim)
        pca_per_class.restore_pred_to_all_models(dim=reduce_to_dim)
        ind = 11
        pca_per_class.draw_sample_restored_image(index=ind)
        pca_per_class.calculate_distances_to_original(index=ind)
        pca_per_class.calculate_all_distances_to_original()

    main()