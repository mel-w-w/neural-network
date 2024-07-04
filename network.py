import random
from tqdm import tqdm
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """
        Args:
            sizes (List[int]): Contains the size of each layer in the network.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    """
    4.1 Feed forward the input x through the network.
    """

    def feedforward(self, x):
        """
        Args:
            x (npt.array): Input to the network.
        Returns:
            List[npt.array]: List of weighted input values to each node
            List[npt.array]: List of activation output values of each node
        """
    
        a_0 = x

        z = []
        a = [a_0]

        for i in range(self.num_layers-1):
            z += [np.matmul(self.weights[i], a[i]) + self.biases[i]]
            a += [sigmoid(z[i])]

        return z, a

    """
    4.2 Backpropagation to compute gradients.
    """

    def backprop(self, x, y, zs, activations):
        """
        Args:
            x (npt.array): Input vector.
            y (float): Target value.
            zs (List[npt.array]): List of weighted input values to each node.
            activations (List[npt.array]): List of activation output values of each node.
        Returns:
            List[npt.array]: List of gradients of bias parameters.
            List[npt.array]: List of gradients of weight parameters.
        """

        L = self.num_layers
        gradients_biases = [None] * L
        gradients_weights = [None] * L

        sig_prime = sigmoid_prime(zs[L-2]) #
        output_a = activations[L-1]
        a = activations[L-2]

        C_prime = self.loss_derivative(output_a, y)
        
        delta = np.multiply(C_prime, sig_prime)
        prev_delta = delta

        # compute delta for l = L-1
        gradients_biases[L-1] = delta
        gradients_weights[L-1] = np.outer(delta, a)

        # # compute delta for l = L-2
        for layer in range(L-2, 0, -1):
            W = self.weights[layer].transpose() # ok
            sig_prime = sigmoid_prime(zs[layer-1]) # ok
            delta = np.multiply(W @ gradients_biases[layer+1], sig_prime)

            gradients_biases[layer] = delta
            a = activations[layer-1]
            gradients_weights[layer] = np.outer(delta, a)

        return gradients_biases[1:], gradients_weights[1:]

    """
    4.3 Update the network's biases and weights after processing a single mini-batch.
    """

    def update_mini_batch(self, mini_batch, alpha):
        """
        Args:
            mini_batch (List[Tuple]): List of (input vector, output value) pairs.
            alpha: Learning rate.
        Returns:
            float: Average loss on the mini-batch.
        """
   
        cum_bias = []
        for i in self.biases:
            cum_bias.append(np.zeros_like(i))
        
        cum_weight = []
        for i in range(len(self.weights)):
            cum_weight.append(np.zeros_like(i))
        
        cum_loss = 0
        
    
        for sample in mini_batch:
            zs, activations = self.feedforward(sample[0])
            grad_biases, grad_weights = self.backprop(sample[0], sample[1], zs, activations)
            loss = self.loss_function(sample[1], activations[-1])

            cum_loss += loss

            for i in range(len(self.biases)):
                cum_bias[i] = np.add(cum_bias[i], grad_biases[i])
            
            for i in range(len(self.weights)):
                cum_weight[i] = np.add(cum_weight[i], grad_weights[i])
        
        n = len(mini_batch)

        # updates
        for i in range(len(cum_bias)):
            bias_update = (alpha/n) * cum_bias[i]
            self.biases[i] = self.biases[i] - bias_update
        
        for i in range(len(cum_weight)):
            weight_update = (alpha/n) * cum_weight[i]
            self.weights[i] = self.weights[i] - weight_update
        

        average_loss = cum_loss/n
        return average_loss


    """
    Train the neural network using mini-batch stochastic gradient descent.
    """

    def SGD(self, data, epochs, alpha, decay, batch_size=32, test=None):
        n = len(data)
        losses = []
        for j in range(epochs):
            print(f"training epoch {j+1}/{epochs}")
            random.shuffle(data)
            for k in tqdm(range(n // batch_size)):
                mini_batch = data[k * batch_size : (k + 1) * batch_size]
                loss = self.update_mini_batch(mini_batch, alpha)
                losses.append(loss)
            alpha *= decay
            if test:
                print(f"Epoch {j+1}: eval accuracy: {self.evaluate(test)}")
            else:
                print(f"Epoch {j+1} complete")
        return losses

    """
    Returns classification accuracy of network on test_data.
    """

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)[1][-1]), y) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results) / len(test_data)

    def loss_function(self, y, y_prime):
        return 0.5 * np.sum((y - y_prime) ** 2)

    """
    Returns the gradient of the squared error loss function.
    """

    def loss_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
    
