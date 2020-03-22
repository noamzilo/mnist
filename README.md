# mnist
Open University Computer Vision 22928- Ex1 Classical Machine Learning Algorithms with MNIST.

Full report can be found under /report/mnist.docx


# KNN
Calculate the performance of the KNN classifier, for k=1..10.

# PCA
* Calculate PCA on MNIST.
* Draw the average digit, and the 6 first principal components.
* Draw a graph of the explained variance, by n most significant components.
* How many components are required to get to 95% variance? 80%?
* Project the digits to dimension 2, and draw the obtained vectors, each digit with its own color.
* Repeat the KNN question, where each digit is represented by its projection to dimension 2, 10, 20.
* For some digit, project it to dimension k, then restore it. Draw the restoration for different k.
* Calculate PCA for every digit separately. Present the 1st 6 principal components of each.
* Calculate the projection of each test set image to each model.
* Restore by each model.
* Calculate the distance from each restoration to the original image.
* Select the model for which the distance is smallest.

# BOW [Bag of Words]
http://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip
* Choose 2 categories from the above dataset
* Calculate dense-sift for all picrutes
* Vector quantization - Calculate K-Means for all extracted features. Choose at least K=100 means.
* For each image, calculate a histogram of its features of the k'th cluster.
* Train a linear SVM, where each picture is represented by its histogram, and each picture is labeled by its class.

* For the test phase:
* Calculate SIFT for each image.
* For each feature, find its nearest neighbour.
* For each image, calculate its histogram.
* Classify using now trained SVM.
