# PA1.py
# author: Nick Cochran,

import math
import csv

# TODO
# figure out how to handle multivariate
# --> work out the equation first, then implement it
# implement pulling in the data from a file at a later time (probably csv)



# input Ms and Xs as tuples
def gradient_descent(b, Ys, *ms_and_xs):
    Ms = []
    Xs = []
    num_features = len(ms_and_xs)/2
    for mx in ms_and_xs:
        Ms.append(mx[0])
        Xs.append(mx[1])
    num_samples = len(Ys)
    alpha = 0.1 # learning rate
    max_iter = 1 # chosen stopping point

    if num_features == 1:
        X = Xs[0]
        m = Ms[0]
        for i in range(0, max_iter):
            m_gradient, b_gradient = 0, 0
            for s in range(0, num_samples):
                error = Ys[s] - linear_regression(b, Ms, Xs)
                m_gradient += -2 * X[s] * error
                b_gradient += -2 * error
            m = m - alpha * (m_gradient / len(X))
            b = b - alpha * (b_gradient / len(X))
    else: # multivariate
        X = Xs[0] # included here to avoid errors until we implement multivariate
        for i in range(0, max_iter):
            m_gradient, b_gradient = 0, 0
            for s in range(0, num_samples):
                error = Ys[s] - linear_regression(b, Ms, Xs)
                m_gradient += -2 * X[s] * error  # not sure how this works with multivariate
                b_gradient += -2 * error
            m = m - alpha * (m_gradient / len(X)) # not sure how this works with multivariate
            b = b - alpha * (b_gradient / len(X))

    print(m, b, sep=" ")
    return m, b


# def mse():
#     citric_acid = [0.5, 0, 0, 0.06, 0.65, 0.37, 0.40, 0.62, 0.38, 0.04]
#     residual_sugar = [2.0, 1.9, 1.8, 1.6, 1.2, 1.2, 1.5, 19.3, 1.5, 1.1]
#     red = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
#     rose = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
#     white = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
#     y = [4, 1, 2, 2, 8, 3, 8, 4, 9, 6] # response values
#     n = len(citric_acid)
#     b = 1.81
#     m_ca, m_s, m_re, m_ro, m_w = 7.41, -0.35, 0.04, 0.33, 4.31
#
#     sum = 0
#     for i in range(0, n):
#         sum += math.pow(y[i] - (m_ca * citric_acid[i] + m_s * residual_sugar[i]
#                           + m_re * red[i] + m_ro * rose[i] + m_w * white[i] + b), 2)
#     return sum / n


def linear_regression(b, Ms, Xs):
    sum = 0
    for i in range(0, len(Ms)):
        sum += Ms[i] * Xs[i]
    return sum + b

def main():
    # add functionality to read in data from file

    gradient_descent()


main()