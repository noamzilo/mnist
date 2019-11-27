import gzip
import pickle
import matplotlib.pyplot as plt


class DataSet(object):
    def __init__(self, path):
        self._path = path
        self._shape = (28, 28)
        self._load()

    def _load(self):
        with gzip.open(self._path, 'rb') as f:
            u = pickle.load(f)
        # u.encoding = 'latin1'
        train_set, validation_set, test_set = u  # separete to data and labels:
        self.x_train, self.y_train = train_set
        self.x_validation, self.y_validation = validation_set
        self.x_test, self.y_test = test_set

    @staticmethod
    def show_mnist_image(title, image):
        plt.figure()
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.title(title)
        plt.show()


    def show_train_image(self, index):
        assert 0 < index < len(self.x_train)
        im = self.x_train[index].reshape(self._shape)
        label = self.y_train[index]
        self.show_mnist_image(title=label, image=im)

    def show_validation_image(self, index):
        assert 0 < index < len(self.x_validation)
        im = self.x_validation[index].reshape(self._shape)
        label = self.y_validation[index]
        self.show_mnist_image(title=label, image=im)

    def show_test_image(self, index):
        assert 0 < index < len(self.x_test)
        im = self.x_test[index].reshape(self._shape)
        label = self.y_test[index]
        self.show_mnist_image(title=label, image=im)


if __name__ == "__main__":
    def main():
        loader = DataSet('../mnist.pkl.gz')
        loader.show_train_image(100)

    main()
