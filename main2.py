import argparse
import cvxopt
from cvxopt import solvers
from pylab import *
import csvreader
import random


def gaussian_kernel(sigma):
    return lambda x_k, x_l: np.exp(-norm(x_k - x_l) ** 2 / (2 * (sigma ** 2)))


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

    def classifier(x):
        evl_sum = 0.0
        for n in range(N):
            evl_sum += (a[n] - a[n + N]) * kernel(X[n], x)
        return evl_sum - th

    return classifier, sv1_idx + sv2_idx


def main(args):
    # set data
    file = args.file
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

    training_X = np.array([x for x, y in zip(X[::2], Y[::2]) if 0 < y < 501])
    training_Y = np.array([y for y in Y[::2] if 0 < y < 501])
    test_X = X[1::2]
    test_Y = Y[1::2]

    kernel = gaussian_kernel(1)

    # get classifier
    classifier, S = get_classifier(training_X, training_Y, kernel, 100, 0.01)

    # evaluate performance
    ver2_selling_price = [p if p > 30 else 0 for p in [classifier(x) - random.uniform(30, 60) for x in test_X]]
    ver2_selling_items = [i for i in ver2_selling_price if i > 30]

    ver1_selling_price = [max(classifier(x) - 37, 0) for x in test_X]

    ver2_sold_items = np.array(
        [selling_price for selling_price, ver1, test_y in zip(ver2_selling_price, ver1_selling_price, test_Y)
         if selling_price < test_y and selling_price < ver1])
    ver2_total_sales = np.sum(ver2_sold_items)

    ver1_sold_items = np.array(
        [selling_price for selling_price, ver2, test_y in zip(ver1_selling_price, ver2_selling_price, test_Y)
         if selling_price < test_y and (selling_price < ver2 or ver2 == 0)])
    ver1_total_sales = np.sum(ver1_sold_items)

    simple_selling_price = np.average(training_Y)
    simple_sold_items = np.array(
        [simple_selling_price for test_y, ver1 in zip(test_Y, ver1_selling_price)
         if simple_selling_price < test_y and simple_selling_price < ver1])
    simple_total_sales = np.sum(simple_sold_items)

    print("\n----------------- My Agent Ver2-----------------")
    print("selling items : " + str(len(ver2_selling_items)))
    print("sold items : " + str(len(ver2_sold_items)))
    print("total sales : " + str(ver2_total_sales))
    print("efficiency : " + str((ver2_total_sales / len(ver2_selling_items)) / (np.sum(test_Y) / 100)))
    print("\n----------------- Simple Agent -----------------")
    print("sold items : " + str(len(simple_sold_items)))
    print("total sales : " + str(simple_total_sales))
    print("efficiency : " + str((simple_total_sales / 100) / (np.sum(test_Y) / 100)))
    print("\n----------------- Ideal -----------------")
    print("sold items : " + str(len(test_Y)))
    print("total sales : " + str(np.sum(test_Y)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="filename for input data", type=str)
    parsed_args = parser.parse_args()
    main(parsed_args)
