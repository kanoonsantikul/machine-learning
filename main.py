import pandas
import numpy
import matplotlib.pyplot as plt

from learning import Perceptron ,AdalineGD

def main():
    data_frame = pandas.read_csv(
            './iris.data',
            header=None)

    y = data_frame.iloc[0:100, 4].values
    y = numpy.where(y == 'Iris-setosa', -1, 1)
    X = data_frame.iloc[0:100, [0,2]].values

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada1 = AdalineGD(0.01, 10).fit(X, y)
    ada2 = AdalineGD(0.0001, 10).fit(X, y)
    print(ada1.cost_)
    print(ada2.cost_)

    ppn = Perceptron(0.1, 10)
    ppn.fit(X, y)

if __name__ == '__main__':
    main()
