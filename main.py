import argparse
import cvxopt
from cvxopt import solvers
from pylab import *
import csvreader


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


def draw_graph(X, classifier, S):
    for i in range(len(X)):
        plot(X[i, 0], X[i, 1], 'rx')

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
            cs = contour(x1, x2, Z, linewidths=1, origin='lower')
            plt.clabel(cs, inline=1, fontsize=10)
        except UserWarning:
            print("[ERROR]Fail to draw borderline. Use kernel.")
    xlim(min_x1-margin_x1, max_x1+margin_x1)
    ylim(min_x2-margin_x2, max_x2+margin_x2)
    show()


def get_classifier(X, Y, kernel, c, eps):
    # Quadratic programming problem
    N = len(X)
    dim = len(X[0])
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j])
    P = cvxopt.matrix(np.vstack((np.hstack((K, -K)), np.hstack((-K, K)))))
    q = cvxopt.matrix(eps * np.ones(2 * N) - np.hstack((Y, -Y)))
    G = cvxopt.matrix(np.vstack((np.diag([-1.0] * (2 * N)), np.diag([1.0] * (2 * N)))))
    h = cvxopt.matrix(np.hstack((np.zeros(2 * N), np.ones(2 * N) * c)))
    A = cvxopt.matrix(np.hstack((np.ones(N), -np.ones(N))), (1, 2 * N), 'd')
    b = cvxopt.matrix(0.0)
    cvxopt.solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except ValueError:
        print("[ERROR]Fail to solve quadratic programming problem.")
        sys.exit(1)
    a = array(sol['x']).reshape(N * 2)
    print("\n----------- Lagrange multipliers -----------\n", a, "\n")

    # pick up index of support vector
    sv1_idx = []
    sv2_idx = []
    for i in range(N):
        if 1e-5 < a[i] < c - 1e-5:
            sv1_idx.append(i)
        elif 1e-5 < a[i + N] < c - 1e-5:
            sv2_idx.append(i)

    if len(sv1_idx) == 0 and len(sv2_idx) == 0:
        print("[ERROR]No support vector. Change kernel parameter.")
        sys.exit(1)

    # evaluate weight
    w = np.zeros(dim)
    for i in range(N):
        w += (a[i] - a[i + N]) * X[i]
    print("------------------ weight ------------------\n", w, "\n")

    # evaluate average of threshold
    th_sum = 0
    for i in sv1_idx:
        temp = 0
        for j in range(N):
            temp += (a[j] - a[j + N]) * kernel(X[i], X[j])
        th_sum += (temp - Y[i] + eps)
    for i in sv2_idx:
        temp = 0
        for j in range(N):
            temp += (a[j] - a[j + N]) * kernel(X[i], X[j])
        th_sum += (temp - Y[i] - eps)
    th = th_sum / len(sv1_idx + sv2_idx)
    print("----------------- threshold -----------------\n", th, "\n")

    def classifier(x):
        evl_sum = 0.0
        for n in range(N):
            evl_sum += (a[n] - a[n + N]) * kernel(X[n], x)
        return evl_sum - th

    return classifier, sv1_idx + sv2_idx


def main(args):
    # set data
    file = args.file
    kernel_type = args.kernel
    c = args.cost
    eps = args.eps

    if ".csv" in file:
        cr = csvreader
        X, Y = cr.read(file)
    else:
        data = []
        for line in open(file):
            tmp = [float(s) for s in line.replace(",", " ").split()]
            data.append(tmp)
        dim = len(data[0]) - 1
        X = np.empty((0, dim), float)
        Y = np.empty((1, 0), float)
        for i in data:
            x_i = np.array([i[:dim]])
            y_i = np.array(i[dim])
            X = np.append(X, x_i, axis=0)
            Y = np.append(Y, y_i, axis=None)
    # X = standardize(X)

    if kernel_type == "p":
        param = 2 if args.param is None else args.param
        kernel = polynomial_kernel(param)
    elif kernel_type == "g":
        param = 5 if args.param is None else args.param
        kernel = gaussian_kernel(param)
    else:
        kernel = normal_kernel()

    classifier, S = get_classifier(X, Y, kernel, c, eps)
    for x, y in zip(X, Y):
        print(classifier(x) - y)
    if len(X[0]) == 2:
        draw_graph(X, classifier, S)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="filename for input data", type=str)
    parser.add_argument("-k", "--kernel", help="select kernel type (g: gaussian, p: polynomial)", type=str, default="n")
    parser.add_argument("-p", "--param", help="set parameter for kernel", type=float)
    parser.add_argument("-c", "--cost", help="set regularization parameter", type=int, default=1000)
    parser.add_argument("-e", "--eps", help="set epsilon", type=float, default=0.1)
    parsed_args = parser.parse_args()
    main(parsed_args)
