import numpy.random as r
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score

digits = load_digits()
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
y = digits.target  # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# this function convert digit value to vector in output layer of network
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)

nn_structure = [64, 30, 10]


def f(x):
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


# this code is for initialisation weight
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l - 1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b


# this code set the mean weight and bias to zero
def init_tri_values(nn_structrue):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structrue)):
        tri_W[l] = np.zeros((nn_structrue[1], nn_structrue[l - 1]))
        tri_b[l] = np.zeros((nn_structrue[l],))
    return tri_W, tri_b


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is a first layer, then the input into the weight is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l + 1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*h^(l)+b^(l)
        h[l + 1] = f(z[l + 1])  # h^(l) = f(z^(l))
    return h, z


# this two functions calculate the output layer delta,
# and any hidden layer delta values to perform the backpropagation

def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z^(l))
    return -(y - h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (tarnspose(W^l) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


def train_nn(nn_straucture, X, y, iter_num=3000, alpha=0.25):
    W, b = setup_and_init_weights(nn_straucture)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt % 1000 == 0:
            print('iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_straucture)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be use
            # in the gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the error
            for l in range(len(nn_straucture), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i, :], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i, :] - h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l + 1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1)*transpose(h^(l))
                    tri_W[l] = np.dot(delta[l + 1][:, np.newaxis], np.transpose(h[l][:, np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l + 1]
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0 / m * tri_W[l])
            b[l] += -alpha * (1.0 / m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0 / m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()


# Assessing the accuracy of the trained model
def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y


y_pred = predict_y(W, b, X_test, 3)
Percentage_correct = accuracy_score(y_test,y_pred)*100
print(Percentage_correct)
