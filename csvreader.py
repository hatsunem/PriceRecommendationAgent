import csv
import numpy as np

# AR(43): zipcode
# BB(53): accommodates
# BC(54): bathrooms
# BD(55): bedrooms
# BE(56): beds
# BN(65): guests_included
# CB(79): review_scores_rating

# AC(28): host_is_superhost
# AJ(35): host_has_profile_pic
# AY(50): is_location_exact
# CL(89): instant_bookable
# CO(92): require_guest_profile_picture
# CP(93): require_guest_phone_verification

# BI(60): price


def read(filename):
    X = np.empty((0, 11), float)
    Y = np.empty((1, 0), float)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # skip = 0
        # while skip < 800:
        #     next(reader)
        #     skip += 1

        count = 0
        for row in reader:
            if row[54] == '' or row[55] == '' or row[56] == '':
                continue
            x_i = np.array([[
                float(row[53]),  # accommodates
                float(row[54]),  # bathrooms
                float(row[55]),  # bedrooms
                float(row[56]),  # beds
                float(row[65]),  # guests_included
                1 if row[28] == 't' else 0,  # host_is_superhost
                1 if row[35] == 't' else 0,  # host_has_profile_pic
                1 if row[50] == 't' else 0,  # is_location_exact
                1 if row[89] == 't' else 0,  # instant_bookable
                1 if row[92] == 't' else 0,  # require_guest_profile_picture
                1 if row[93] == 't' else 0   # require_guest_phone_verification
            ]])
            y_i = np.array(float(row[60][1:].replace(",", "")))
            X = np.append(X, x_i, axis=0)
            Y = np.append(Y, y_i, axis=None)
            count += 1
            if count == 200:
                break
    return X, Y
