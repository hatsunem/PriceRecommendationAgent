import csv
import numpy as np

# AR(43): zipcode
# BB(53): accommodates
# CB(79): review_scores_rating

# BI(60): price


def read(filename):
    X = np.empty((0, 3), float)
    Y = np.empty((1, 0), float)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        for i, row in enumerate(reader):
            if i == 100:
                break
            if row[43] == '' or row[53] == '' or row[79] == '':
                continue
            x_i = np.array([[float(row[43]), float(row[53]), float(row[79])]])
            y_i = np.array(float(row[60][1:].replace(",", "")))
            X = np.append(X, x_i, axis=0)
            Y = np.append(Y, y_i, axis=None)
    return X, Y
