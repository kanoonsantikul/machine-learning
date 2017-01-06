import numpy
from numpy.random import seed

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
        return numpy.where(self.activation(training_vector) >= 0.0, 1, -1)

class AdalineSGD(object):
    def __init__(self, learning_rate, repeat, shuffle, random_state):
        self.learning_rate = learning_rate
        self.repeat = repeat
        self.weight_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, training_vector, target_value):
        self._initialize_weights(training_vector.shape[1])
        self.cost_ = []

        for i in range(self.repeat):
            if self.shuffle:
                training_vector, target_value = self._shuffle(training_vector, target_value)
            cost = []

            for training_row, target in zip(training_vector, target_value):
                cost.append(self._update_weight(training_row ,target))
            avg_cost = sum(cost)/len(target_value)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, training_vector, target_value):
        if not self.weight_initialized:
            self._initialize_weights(X.shape[1])
        if target_value.ravel().shape[0] > 1:
            for training_row, target in zip(training_vector, target_value):
                self._update_weight(training_row, target)
        else:
            self._update_weight(training_vector, target_value)
        return self

    def _shuffle(self, training_vector, target_value):
        r = numpy.random.permutation(len(target_value))
        return training_vector[r], target_value[r]

    def _initialize_weights(self, len):
        self.weight_ = numpy.zeros(1 + len)
        self.weight_initialized = True

    def _update_weight(self, training_row, target):
        output = self.net_input(training_row)
        error = (target - output)
        self.weight_[1:] += self.learning_rate * training_row.dot(error)
        self.weight_[0] += self.learning_rate * error
        cost =  0.5 * error**2
        return cost

    def net_input(self, training_vector):
        return numpy.dot(training_vector, self.weight_[1:]) + self.weight_[0]

    def predict(self, training_vector):
        return numpy.where(self.activation(training_vector) >= 0.0, 1, -1)
