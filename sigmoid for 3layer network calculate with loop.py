import numpy as np

# define wights and bias and 3input variable(x)
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
w2 = np.zeros((1, 3))
w2[0, :] = np.array([0.5, 0.5, 0.5])
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])
w = [w1, w2]
b = [b1, b2]

# a dummy x input vector
x = [1.5, 2.0, 3.0]


# Setup sigmoid function
def f(x):
    return 1 / (1 + np.exp(-x))


# simple loop neural network calculation
def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        # Setup the input array which the weights will be multiplied by for each layer
        # if it's the first layer, the input array will be the x input vector
        # if it's not the first layer, the input to the next layer will  be the output of the previous layer
        if l == 0:
            node_in = x
        else:
            node_in = h
        # Setup the output array for the node in layer l + 1
        h = np.zeros((w[l].shape[0],))
        # loop through the rows of the weight array
        for i in range(w[l].shape[0]):
            # Setup the sum inside the activation function
            f_sum = 0
            # loop through the columns of the weight array
            for j in range(w[l].shape[1],):
                f_sum += w[l][i][j] * node_in[j]
            # add the bias
            f_sum += b[l][i]
            # finally use the activation function to calculate  the i-th output i.e h1, h2, h3
            h[i] = f(f_sum)
    return h

q = (simple_looped_nn_calc(3, x, w, b))
print(q)
