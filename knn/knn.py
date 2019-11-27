from loader.DataSet import DataSet


class KnnClassifier(object):
    def __init__(self, data_path):
        self._data_path = data_path
        self._data = DataSet(self._data_path)

    def run(self):
        pass


if __name__ == "__main__":
    def main():
        classifier = KnnClassifier(data_path='../mnist.pkl.gz')
        classifier.run()

    main()
