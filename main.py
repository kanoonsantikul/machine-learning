import pandas
import numpy
import matplotlib.pyplot as plt

from learning import Perceptron ,AdalineSGD, AdalineGD

def main():
    data_frame = pandas.read_csv('./iris.data', header=None)

    y = data_frame.iloc[0:100, 4].values
    y = numpy.where(y == 'Iris-setosa', -1, 1)
    X = data_frame.iloc[0:100, [0,2]].values

    X_std = numpy.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    ada1 = AdalineGD(0.01, 10).fit(X, y)
    ada2 = AdalineGD(0.0001, 10).fit(X, y)
    ada_gd = AdalineGD(0.01, 15).fit(X_std, y)
    ada_sgd = AdalineSGD(0.01, 15, True, 1).fit(X_std, y)
    print(ada1.cost_)
    print()
    print(ada2.cost_)
    print()
    print(ada_gd.cost_)
    print()
    print(ada_sgd.cost_)

    ppn = Perceptron(0.1, 10)
    ppn.fit(X, y)

if __name__ == '__main__':
    main()
