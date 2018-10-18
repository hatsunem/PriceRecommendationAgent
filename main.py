import argparse
import cvxopt
from cvxopt import solvers
from pylab import *


def gaussian_kernel(sigma):
    return lambda x_k, x_l: np.exp(-norm(x_k - x_l) ** 2 / (2 * (sigma ** 2)))


def polynomial_kernel(d):
    return lambda x_k, x_l: (1 + np.dot(x_k, x_l)) ** d


def normal_kernel():
    return lambda x_k, x_l: np.dot(x_k, x_l)


def standardize(X):
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = np.std(X, axis=0, keepdims=True)
    z_score = (X - x_mean) / x_std
    return z_score


def draw_graph(classifier, X, Y, S):
    for i in range(len(X)):
        if Y[i] == 1:
            plot(X[i, 0], X[i, 1], 'rx')
        else:
            plot(X[i, 0], X[i, 1], 'bx')

    for n in S:
        scatter(X[n, 0], X[n, 1], s=80, c='g', marker='o')

    min_x1 = np.min(X, axis=0)[0]
    max_x1 = np.max(X, axis=0)[0]
    min_x2 = np.min(X, axis=0)[1]
    max_x2 = np.max(X, axis=0)[1]
    margin_x1 = (max_x1 - min_x1) * 0.1
    margin_x2 = (max_x2 - min_x2) * 0.1
    x1, x2 = meshgrid(linspace(min_x1-margin_x1, max_x1+margin_x1, 50),
                      linspace(min_x2-margin_x2, max_x2+margin_x2, 50))
    w, h = x1.shape
    x1.resize(x1.size)
    x2.resize(x2.size)
    Z = array([classifier(array([x1, x2])) for (x1, x2) in zip(x1, x2)])
    x1.resize((w, h))
    x2.resize((w, h))
    Z.resize((w, h))
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            contour(x1, x2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        except UserWarning:
            print("[ERROR]Fail to draw borderline. Use kernel.")
    xlim(min_x1-margin_x1, max_x1+margin_x1)
    ylim(min_x2-margin_x2, max_x2+margin_x2)
    show()


def get_classifier(X, Y, kernel, isSoft, c):
    # Quadratic programming problem
    N = len(X)
    dim = len(X[0])
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = Y[i] * Y[j] * kernel(X[i], X[j])
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(-np.ones(N))
    if isSoft:
        temp1 = np.diag([-1.0] * N)
        temp2 = np.identity(N)
        G = cvxopt.matrix(np.vstack((temp1, temp2)))
        temp1 = np.zeros(N)
        temp2 = np.ones(N) * c
        h = cvxopt.matrix(np.hstack((temp1, temp2)))
    else:
        G = cvxopt.matrix(np.diag([-1.0] * N))
        h = cvxopt.matrix(np.zeros(N))
    A = cvxopt.matrix(Y, (1, N), 'd')
    b = cvxopt.matrix(0.0)
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except ValueError:
        print("[ERROR]Fail to solve quadratic programming problem. Use soft margin SVM.")
        return
    a = array(sol['x']).reshape(N)
    # print("\n----------- Lagrange multipliers -----------\n", a, "\n")

    # pick up index of support vector
    sv_idx = []
    soft_sv_idx = []
    for i in range(N):
        if 1e-5 < a[i]:
            sv_idx.append(i)
        if 1e-5 < a[i] < c:
            soft_sv_idx.append(i)

    if len(sv_idx) == 0:
        print("[ERROR]No support vector. Change kernel parameter.")
        return

    # evaluate weight
    w = np.zeros(dim)
    for i in sv_idx:
        w += a[i] * Y[i] * X[i]
    # print("------------------ weight ------------------\n", w, "\n")

    # evaluate average of threshold
    th_sum = 0
    for i in soft_sv_idx:
        temp = 0
        for j in range(N):
            temp += a[j] * Y[j] * kernel(X[i], X[j])
        th_sum += (temp - Y[i])
    th = th_sum / len(soft_sv_idx)
    # print("----------------- threshold -----------------\n", th, "\n")

    def classifier(x):
        evl_sum = 0.0
        for n in range(N):
            evl_sum += a[n] * Y[n] * kernel(X[n], x)
        return evl_sum - th

    if dim == 2:
        draw_graph(classifier, X, Y, sv_idx)

    return classifier


def main(args):
    # set data
    kernel_type = args.kernel
    isSoft = args.soft
    if isSoft:
        c = 0.5 if args.cost is None else args.cost
    else:
        c = np.inf

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

    param = np.arange(8, 8.2, 0.1)
    accuracies = []
    for p in param:
        if kernel_type == "p":
            kernel = polynomial_kernel(p)
        elif kernel_type == "g":
            kernel = gaussian_kernel(p)
        else:
            kernel = normal_kernel()

        # cross validation
        accuracy_sum = 0
        for i in range(number_of_group):
            test_X = X[i*size_of_group:(i+1)*size_of_group]
            test_Y = Y[i*size_of_group:(i+1)*size_of_group]
            training_X = np.vstack((X[:i*size_of_group], X[(i+1)*size_of_group:]))
            training_Y = Y[:i * size_of_group] + Y[(i + 1) * size_of_group:]
            classifier = get_classifier(training_X, training_Y, kernel, isSoft, c)
            correct = 0
            for j in range(len(test_X)):
                result = sign(classifier(test_X[j]))
                if result == test_Y[j]:
                    correct += 1

            accuracy_sum += correct / len(test_X)
        accuracies.append(accuracy_sum / number_of_group)
    print(accuracies)
    plot(param, accuracies)
    show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="filename for input data", type=str)
    parser.add_argument("-k", "--kernel", help="select kernel type (g: gaussian, p: polynomial)", type=str, default="n")
    parser.add_argument("--soft", help="use soft margin SVM", action="store_true")
    parser.add_argument("-c", "--cost", help="set regularization parameter", type=int, default=0.5)
    parser.add_argument("--cross", help="set regularization parameter", type=int, default=1)
    parsed_args = parser.parse_args()
    main(parsed_args)
