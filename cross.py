import argparse
import cvxopt
from cvxopt import solvers
from pylab import *
import csvreader


def gaussian_kernel():
    return lambda x_k, x_l, sigma: np.exp(-norm(x_k - x_l) ** 2 / (2 * (sigma ** 2)))


def polynomial_kernel():
    return lambda x_k, x_l, d: (1 + np.dot(x_k, x_l)) ** d


def normal_kernel():
    return lambda x_k, x_l, _: np.dot(x_k, x_l)


def standardize(X):
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = np.std(X, axis=0, keepdims=True)
    z_score = (X - x_mean) / x_std
    return z_score


def get_classifier(X, Y, kernel, p, c, eps):
    # Quadratic programming problem
    N = len(X)
    dim = len(X[0])
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j], p)
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

    # pick up index of support vector
    sv1_idx = []
    sv2_idx = []
    for i in range(N):
        if 0 < a[i] < c:
            sv1_idx.append(i)
        elif 0 < a[i + N] < c:
            sv2_idx.append(i)

    if len(sv1_idx) == 0 and len(sv2_idx) == 0:
        print("[ERROR]No support vector. Change kernel parameter.")
        sys.exit(1)

    # evaluate weight
    w = np.zeros(dim)
    for i in range(N):
        w += (a[i] - a[i + N]) * X[i]

    # evaluate average of threshold
    th_sum = 0
    for i in sv1_idx:
        temp = 0
        for j in range(N):
            temp += (a[j] - a[j + N]) * kernel(X[i], X[j], p)
        th_sum += (temp - Y[i] + eps)
    for i in sv2_idx:
        temp = 0
        for j in range(N):
            temp += (a[j] - a[j + N]) * kernel(X[i], X[j], p)
        th_sum += (temp - Y[i] - eps)
    th = th_sum / len(sv1_idx + sv2_idx)

    def classifier(x):
        evl_sum = 0.0
        for n in range(N):
            evl_sum += (a[n] - a[n + N]) * kernel(X[n], x, p)
        return evl_sum - th

    return classifier, sv1_idx + sv2_idx


def main(args):
    # set data
    file = args.file
    kernel_type = args.kernel
    C = [1, 10, 100, 1000]
    epsilon = [0.1, 0.01]

    if ".csv" in file:
        cr = csvreader
        X, Y = cr.read(file)
    else:
        data = []
        for line in open(args.file):
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
        kernel = polynomial_kernel()
        param = np.arange(0, 14, 1)
    elif kernel_type == "g":
        kernel = gaussian_kernel()
        param = np.logspace(-10, 19, 30, base=2)
    else:
        kernel = normal_kernel()
        param = [1]

    # divide into groups
    number_of_group = args.cross
    size_of_group = len(X) // number_of_group
    test_X = [X[i*size_of_group:(i+1)*size_of_group] for i in range(number_of_group)]
    test_Y = [Y[i*size_of_group:(i+1)*size_of_group] for i in range(number_of_group)]
    training_X = [np.vstack((X[:i*size_of_group], X[(i+1)*size_of_group:])) for i in range(number_of_group)]
    training_Y = [np.hstack((Y[:i*size_of_group], Y[(i+1)*size_of_group:])) for i in range(number_of_group)]

    for c in C:
        for eps in epsilon:
            mses = []
            for p in param:
                # cross validation
                mse_sum = 0
                for i in range(number_of_group):
                    classifier, S = get_classifier(training_X[i], training_Y[i], kernel, p, c, eps)
                    pred_Y = [classifier(x) for x in test_X[i]]
                    mse_sum += np.sum((test_Y[i] - pred_Y) ** 2) / len(test_X[i])
                mse_avg = mse_sum / number_of_group
                mses.append(mse_avg)
                print("cost: " + str(c) + ", eps: " + str(eps) + ", param: " + str(p) + ", mse: " + str(mse_avg))
            if kernel_type == "p":
                plot(param, mses, label="c = " + str(c) + ", e = " + str(eps))
            elif kernel_type == "g":
                plot(log2(param), mses, label="c = " + str(c) + ", e = " + str(eps))
    legend()
    show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="filename for input data", type=str)
    parser.add_argument("-k", "--kernel", help="select kernel type (g: gaussian, p: polynomial)", type=str, default="n")
    parser.add_argument("--cross", help="set regularization parameter", type=int, default=2)
    parsed_args = parser.parse_args()
    main(parsed_args)
