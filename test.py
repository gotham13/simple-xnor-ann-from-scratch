'''
Created by Gotham on 25-06-2018.
'''
from nn import NeuralNetwork
import random

training_data = \
    [
        {
            "inputs": [[0], [0]],
            "outputs": [[1]]
        },
        {
            "inputs": [[1], [1]],
            "outputs": [[1]]
        },
        {
            "inputs": [[1], [0]],
            "outputs": [[0]]
        },
        {
            "inputs": [[0], [1]],
            "outputs": [[0]]
        }
    ]

if __name__ == "__main__":
    neural_net = NeuralNetwork(new_nn_dict={"input_nodes": 2,
                                            "hidden_nodes": 8,
                                            "out_nodes": 1})

    for i in range(0, 60000):
        data = random.choice(training_data)
        neural_net.train(data['inputs'], data['outputs'])

    print(neural_net.predict([[0], [0]]))
    print(neural_net.predict([[0], [1]]))
    print(neural_net.predict([[1], [0]]))
    print(neural_net.predict([[1], [1]]))
