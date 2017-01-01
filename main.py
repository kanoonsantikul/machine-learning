import pandas
import numpy
import matplotlib.pyplot

from my_perceptron import Perceptron

def main():
    data_frame = pandas.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            header=None)
    print(data_frame.tail())

    y = data_frame.iloc[0:100, 4].values
    y = numpy.where(y == 'Iris-setosa', -1, 1)
    X = data_frame.iloc[0:100, [0,2]].values

    ppn = Perceptron(0.1, 10)
    ppn.fit(X, y)
    print(ppn.errors_)

if __name__ == '__main__':
    main()
