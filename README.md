# neural-network

This project was completed for the course Artificial Intelligence at Columbia University.

The goal of the project was to implement and train a fully connected neural network in Python. This neural network was then applied to a small test problem, and then was used to tackle the MNIST dataset to recognize handwritten digits.

For the sake of academic privacy, I have only included the code I specifically wrote (and not what the TAs and the Professor prepared for the project).

The functions I designed and implemented are listed below:

- **feedforward()**: implements the feedforward propagation through a neural network with multiple layers, using the sigmoid activation function.
- **backprop()**: computes the gradients of the loss function *C* with respect to the bias and weight parameters of a neural network.
- **update_mini_batch()**: updates the neural network weights using mini-batch gradient descent.
