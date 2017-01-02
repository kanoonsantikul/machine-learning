import numpy

class Perceptron(object):
    def __init__(self, learning_rate, repeat):
        self.learning_rate = learning_rate
        self.repeat = repeat

    def fit(self, training_vector, target_value):
        self.weight_ = numpy.zeros(1 + training_vector.shape[1])
        self.errors_ = []

        for _ in range(self.repeat):
            errors = 0

            for training_row, target in zip(training_vector, target_value):
                update = self.update_weight(training_row, target)
                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self

    def net_input(self, training_row):
        return numpy.dot(training_row, self.weight_[1:]) + self.weight_[0]

    def predict(self, training_row):
        return numpy.where(self.net_input(training_row) >= 0.0, 1, -1)

    def update_weight(self, training_row, target):
        update = self.learning_rate * (target - self.predict(training_row))
        self.weight_[1:] += update * training_row
        self.weight_[0] += update
        return update
