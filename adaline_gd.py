import numpy
from numpy.random import seed

class AdalineGD(object):
    def __init__(self, learning_rate, repeat):
        self.learning_rate = learning_rate
        self.repeat = repeat

    def fit(self, training_vector, target_value):
        self.weight_ = numpy.zeros(1 + training_vector.shape[1])
        self.cost_ = []

        for i in range(self.repeat):
            output = self.net_input(training_vector)
            errors = (target_value - output)
            self.weight_[1:] += self.learning_rate * training_vector.T.dot(errors)
            self.weight_[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, training_vector):
        return numpy.dot(training_vector, self.weight_[1:]) + self.weight_[0]

    def predict(self, training_vector):
        return numpy.where(self.net_input(training_vector) >= 0.0, 1, -1)
