"""
Created by Gotham on 24-06-2018.
"""
import numpy


class ActivationFunction:
    def __init__(self, func_type="sigmoid"):
        self.func_type = func_type

    def func(self, x):
        if self.func_type == "sigmoid":
            return 1 / (1 + numpy.exp(-x))
        elif self.func_type == "tanh":
            return numpy.tanh(x)

    def derivative(self, y):
        if self.func_type == "sigmoid":
            return y * (1 - y)
        elif self.func_type == "tanh":
            return 1 - (y * y)


class NeuralNetwork:
    def __init__(self, network=None, new_nn_dict=None):
        if network is None:
            self.input_nodes = new_nn_dict['input_nodes']
            self.hidden_nodes = new_nn_dict['hidden_nodes']
            self.output_nodes = new_nn_dict['out_nodes']
            self.weights_ih = self.get_random_filled_matrix(rows=self.hidden_nodes, cols=self.input_nodes)
            self.weights_ho = self.get_random_filled_matrix(rows=self.output_nodes, cols=self.hidden_nodes)
            self.bias_h = self.get_random_filled_matrix(self.hidden_nodes, 1)
            self.bias_o = self.get_random_filled_matrix(self.output_nodes, 1)
        else:
            self.input_nodes = network.input_nodes
            self.hidden_nodes = network.hidden_nodes
            self.output_nodes = network.output_nodes
            self.weights_ih = numpy.copy(network.weight_ih)
            self.weights_ho = numpy.copy(network.weight_ho)
            self.bias_h = numpy.copy(network.bias_h)
            self.bias_o = numpy.copy(network.bias_o)
        self.activation_function = ActivationFunction()
        self.vectorized_func = numpy.vectorize(self.activation_function.func)
        self.vectorized_derivative = numpy.vectorize(self.activation_function.derivative)
        self.learning_rate = 0.1

    def set_activation_function(self, func="sigmoid"):
        self.activation_function = ActivationFunction(func_type=func)
        self.vectorized_func = numpy.vectorize(self.activation_function.func)
        self.vectorized_derivative = numpy.vectorize(self.activation_function.derivative)

    def set_learning_rate(self, rate=0.1):
        self.learning_rate = rate

    @staticmethod
    def get_matrix(rows, cols):
        return numpy.array([[0] * cols for _ in range(rows)])

    @staticmethod
    def get_random_filled_matrix(rows, cols):
        return numpy.random.rand(rows, cols)

    def predict(self, input_list):
        inputs = numpy.array(input_list)
        hidden = numpy.dot(self.weights_ih, inputs)
        hidden = numpy.add(hidden, self.bias_h)
        hidden = self.vectorized_func(hidden)

        output = numpy.dot(self.weights_ho, hidden)
        output = numpy.add(output, self.bias_o)
        output = self.vectorized_func(output)

        return output.tolist()

    def train(self, input_list, target_list):
        # FEED FORWARD ALGO
        inputs = numpy.array(input_list)
        hidden = numpy.dot(self.weights_ih, inputs)
        hidden = numpy.add(hidden, self.bias_h)
        hidden = self.vectorized_func(hidden)
        outputs = numpy.dot(self.weights_ho, hidden)
        outputs = numpy.add(outputs, self.bias_o)
        outputs = self.vectorized_func(outputs)
        targets = numpy.array(target_list)

        # BACKPROPOGATION ALGO
        output_errors = numpy.subtract(targets, outputs)
        gradients = self.vectorized_derivative(outputs)
        gradients = numpy.multiply(gradients, output_errors)
        gradients = numpy.multiply(gradients, self.learning_rate)
        hidden_t = numpy.transpose(hidden)

        weight_ho_deltas = numpy.dot(gradients, hidden_t)
        self.weights_ho = numpy.add(self.weights_ho, weight_ho_deltas)
        self.bias_o = numpy.add(self.bias_o, gradients)
        who_t = numpy.transpose(self.weights_ho)
        hidden_errors = numpy.dot(who_t, output_errors)
        hidden_gradient = self.vectorized_derivative(hidden)
        hidden_gradient = numpy.multiply(hidden_gradient, hidden_errors)
        hidden_gradient = numpy.multiply(hidden_gradient, self.learning_rate)

        inputs_t = numpy.transpose(inputs)
        weight_ih_deltas = numpy.dot(hidden_gradient, inputs_t)
        self.weights_ih = numpy.add(self.weights_ih, weight_ih_deltas)
        self.bias_h = numpy.add(self.bias_h, hidden_gradient)

    def copy(self):
        return NeuralNetwork(self)

    def mutate(self, func):
        func1 = numpy.vectorize(func)
        self.weights_ih = func1(self.weights_ih)
        self.weights_ho = func1(self.weights_ho)
        self.bias_h = func1(self.bias_h)
        self.bias_o = func1(self.bias_o)