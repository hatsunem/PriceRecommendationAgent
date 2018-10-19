import argparse
import cvxopt
from cvxopt import solvers
from pylab import *


def gaussian_kernel():
    return lambda x_k, x_l, sigma: np.exp(-norm(x_k - x_l) ** 2 / (2 * (sigma ** 2)))


def polynomial_kernel():
    return lambda x_k, x_l, d: (1 + np.dot(x_k, x_l)) ** d


def normal_kernel():
    return lambda x_k, x_l: np.dot(x_k, x_l)


def standardize(X):
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = np.std(X, axis=0, keepdims=True)
    z_score = (X - x_mean) / x_std
    return z_score


def get_classifier(X, Y, kernel, p):
    # Quadratic programming problem
    N = len(X)
    dim = len(X[0])
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = Y[i] * Y[j] * kernel(X[i], X[j], p)
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(-np.ones(N))
    G = cvxopt.matrix(np.diag([-1.0] * N))
    h = cvxopt.matrix(np.zeros(N))
    A = cvxopt.matrix(Y, (1, N), 'd')
    b = cvxopt.matrix(0.0)
    cvxopt.solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except ValueError:
        print("[ERROR]Fail to solve quadratic programming problem. Use soft margin SVM.")
        return
    a = array(sol['x']).reshape(N)

    # pick up index of support vector
    sv_idx = []
    for i in range(N):
        if 1e-5 < a[i]:
            sv_idx.append(i)

    if len(sv_idx) == 0:
        print("[ERROR]No support vector. Change kernel parameter.")
        return

    # evaluate weight
    w = np.zeros(dim)
    for i in sv_idx:
        w += a[i] * Y[i] * X[i]

    # evaluate average of threshold
    th_sum = 0
    for i in sv_idx:
        temp = 0
        for j in range(N):
            temp += a[j] * Y[j] * kernel(X[i], X[j], p)
        th_sum += (temp - Y[i])
    th = th_sum / len(sv_idx)

    def classifier(x):
        evl_sum = 0.0
        for n in range(N):
            evl_sum += a[n] * Y[n] * kernel(X[n], x, p)
        return evl_sum - th

    return classifier


def main(args):
    # set data
    kernel_type = args.kernel

    data = []
    for line in open(args.file):
        tmp = [int(s) for s in line.replace(",", " ").split()]
        data.append(tmp)

    N = len(data)
    dim = len(data[0]) - 1

    X = np.empty((0, dim), int)
    Y = []
    for i in data:
        x_i = np.array([i[:dim]])
        X = np.append(X, x_i, axis=0)
        Y.append(i[dim])
    X = standardize(X)

    number_of_group = args.cross
    size_of_group = N // number_of_group

    if kernel_type == "p":
        kernel = polynomial_kernel()
        param = np.arange(0, 17, 1)
    elif kernel_type == "g":
        kernel = gaussian_kernel()
        param = np.logspace(-10, 39, 50, base=2)
    else:
        print("[ERROR]Set kernel.")
        return
    accuracies = []
    for p in param:
        # cross validation
        accuracy_sum = 0
        for i in range(number_of_group):
            test_X = X[i*size_of_group:(i+1)*size_of_group]
            test_Y = Y[i*size_of_group:(i+1)*size_of_group]
            training_X = np.vstack((X[:i*size_of_group], X[(i+1)*size_of_group:]))
            training_Y = Y[:i * size_of_group] + Y[(i + 1) * size_of_group:]
            classifier = get_classifier(training_X, training_Y, kernel, p)
            correct = 0
            for j in range(len(test_X)):
                result = sign(classifier(test_X[j]))
                if result == test_Y[j]:
                    correct += 1

            accuracy_sum += correct / len(test_X)
        accuracy_avg = accuracy_sum / number_of_group
        print("Param: " + str(p) + ", Accuracy: " + str(accuracy_avg))
        accuracies.append(accuracy_avg)

    if kernel_type == "p":
        plot(param, accuracies)
    elif kernel_type == "g":
        plot(log2(param), accuracies)
    show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="filename for input data", type=str)
    parser.add_argument("-k", "--kernel", help="select kernel type (g: gaussian, p: polynomial)", type=str, default="n")
    parser.add_argument("--cross", help="set regularization parameter", type=int, default=2)
    parsed_args = parser.parse_args()
    main(parsed_args)
