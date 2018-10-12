import argparse
import cvxopt
from cvxopt import solvers
from pylab import *


def f(x, a, Y, X, th, N, kernel):
    evl_sum = 0.0
    for n in range(N):
        evl_sum += a[n] * Y[n] * kernel(X[n], x)
    return evl_sum - th


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


def draw_graph(X, Y, a, th, N, S, kernel):
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
    Z = array([f(array([x1, x2]), a, Y, X, th, N, kernel) for (x1, x2) in zip(x1, x2)])
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


def main(args):
    # Set data
    kernel_type = args.kernel
    isSoft = args.soft
    if isSoft:
        c = 0.5 if args.cost is None else args.cost
    else:
        c = np.inf
    if kernel_type == "p":
        d = 2 if args.param is None else args.param
        kernel = polynomial_kernel(d)
    elif kernel_type == "g":
        sigma = 5.0 if args.param is None else args.param
        kernel = gaussian_kernel(sigma)
    else:
        kernel = normal_kernel()

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

    # Quadratic programming problem
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
    print("\n----------- Lagrange multipliers -----------\n", a, "\n")

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
    for n in sv_idx:
        w += a[n] * Y[n] * X[n]
    print("------------------ weight ------------------\n", w, "\n")

    # evaluate average of threshold
    th_sum = 0
    for n in soft_sv_idx:
        temp = 0
        for m in range(N):
            temp += a[m] * Y[m] * kernel(X[n], X[m])
        th_sum += (temp - Y[n])
    th = th_sum / len(soft_sv_idx)
    print("----------------- threshold -----------------\n", th, "\n")

    if dim == 2:
        draw_graph(X, Y, a, th, N, sv_idx, kernel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="filename for input data", type=str)
    parser.add_argument("-k", "--kernel", help="select kernel type (g: gaussian, p: polynomial)", type=str, default="n")
    parser.add_argument("-p", "--param", help="set parameter for kernel", type=float)
    parser.add_argument("--soft", help="use soft margin SVM", action="store_true")
    parser.add_argument("-c", "--cost", help="set regularization parameter", type=int, default=0.5)
    parsed_args = parser.parse_args()
    main(parsed_args)
