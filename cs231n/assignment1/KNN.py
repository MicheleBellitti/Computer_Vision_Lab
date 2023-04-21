import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split


class KnnClassifier(object):
    def __init__(self, k=5):
        self.x = None
        self.y = None
        self.K = k

    def fit_(self, x, y):
        self.x = x
        self.y = y

    def predict_(self, v):
        len_test = v.shape[0]

        y_pre = np.zeros((len_test,))
        for i in range(len_test):
            dist = np.sum(np.abs(self.x - v[i, :]), axis=1)

            # take the k minimum distances

            min_index = np.argpartition(dist, self.K)[:self.K]

            y_pre[i] = np.round(np.mean(self.y[min_index]))
        return y_pre


# visualize knn prediction results

def visualize_knn(x_test, y_test, pred):
    plt.figure()
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=pred)
    plt.show()


# test knn classifier

if __name__ == '__main__':
    # load a sample dataset from sklearn

    iris = load_iris()
    x = iris.data[:, :2]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)

    # train knn classifier
    accuracy = []
    # validation  -> find the best k
    for k in np.arange(1, 10, 1):
        knn = KnnClassifier(k)
        knn.fit_(x_train, y_train)
        y_pre = knn.predict_(x_test)
        accuracy.append(np.sum(y_test == y_pre) / y_test.shape[0])
    k = np.argmax(accuracy) + 1
    print(f'K = {k}')
    model = KnnClassifier(k=k)
    model.fit_(x_train, y_train)
    y_h = model.predict_(x_test)
    print(f'accuracy: {round(np.sum(y_test == y_h) / y_test.shape[0]*100)}%')
    visualize_knn(x_test, y_test, y_h)
