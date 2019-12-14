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


# matrix feed forward neural network calculation that take less time in compare with loop calculate

def matrix_feed_forward_calc(n_layers, x, w, b):

    global h
    for l in range(n_layers - 1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return h


print(matrix_feed_forward_calc(3, x, w, b))
