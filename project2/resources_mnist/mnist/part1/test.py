import os
import sys
import time
import traceback
import numpy as np
import linear_regression
import svm
import softmax
import features
import kernel

sys.path.append("..")
import utils

verbose = False

epsilon = 1e-6

def green(s):
    return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


def check_real(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not np.isreal(res):
        log(red("FAIL"), ex_name, ": does not return a real number, type: ", type(res))
        return True
    if not -epsilon < res - exp_res < epsilon:
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def equals(x, y):
    if type(y) == np.ndarray:
        return (np.abs(x - y) < epsilon).all()
    return -epsilon < x - y < epsilon

def check_tuple(ex_name, f, exp_res, *args, **kwargs):
    try:
        res = f(*args, **kwargs)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == tuple:
        log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a tuple of size ", len(exp_res), " but got tuple of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_array(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == np.ndarray:
        log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        return True
    if not equals(res, exp_res):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)

        return True

def check_list(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == list:
        log(red("FAIL"), ex_name, ": does not return a list, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a list of size ", len(exp_res), " but got list of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_get_mnist():
    ex_name = "Get MNIST data"
    train_x, train_y, test_x, test_y = utils.get_MNIST_data()
    log(green("PASS"), ex_name, "")


def check_closed_form():
    ex_name = "Closed form"
    X = np.arange(1, 16).reshape(3, 5)
    Y = np.arange(1, 4)
    lambda_factor = 0.5
    exp_res = np.array([-0.03411225,  0.00320187,  0.04051599,  0.07783012,  0.11514424])
    if check_array(
            ex_name, linear_regression.closed_form,
            exp_res, X, Y, lambda_factor):
        return

    log(green("PASS"), ex_name, "")

def check_svm():
    ex_name = "One vs rest SVM"
    n, m, d = 5, 3, 7
    train_x = np.random.random((n, d))
    test_x = train_x[:m]
    train_y = np.zeros(n)
    train_y[-1] = 1
    exp_res = np.zeros(m)

    if check_array(
            ex_name, svm.one_vs_rest_svm,
            exp_res, train_x, train_y, test_x):
        return

    train_y = np.ones(n)
    train_y[-1] = 0
    exp_res = np.ones(m)

    if check_array(
            ex_name, svm.one_vs_rest_svm,
            exp_res, train_x, train_y, test_x):
        return

    train_x =np.array([[0.26851685, 0.04599562],
        [0.10115761, 0.01353639],
        [0.31891041, 0.71610098],
        [0.33937115, 0.05235358],
        [0.26814632, 0.08114137],
        [0.16865807, 0.9994675 ],
        [0.70928218, 0.33803127],
        [0.02836257, 0.9925004 ],
        [0.58958086, 0.19417839],
        [0.23979632, 0.1276912 ]])
    train_y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0])
    test_x= np.array([[0.66996862, 0.36352826],
        [0.51906488, 0.41920761],
        [0.74128256, 0.72149055],
        [0.07098625, 0.72491776],
        [0.36983766, 0.50860052],
        [0.33586929, 0.40427077],
        [0.79226918, 0.91143374],
        [0.55250951, 0.42342281],
        [0.46202418, 0.89625536],
        [0.85167195, 0.53881762],
        [0.45533267, 0.77720456],
        [0.38543964, 0.5597373 ],
        [0.85752041, 0.10202539],
        [0.32052446, 0.75807725],
        [0.78864729, 0.94930757],
        [0.99715649, 0.05472494],
        [0.57438457, 0.74386228]])
    Submission_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    if check_array(
            ex_name, svm.one_vs_rest_svm,
            Submission_output, train_x, train_y, test_x):
        return

    train_x = np.array([[0.39034702, 0.58699944, 0.47399077],
        [0.05061553, 0.70856241, 0.38631591],
        [0.49092421, 0.86764713, 0.30275599],
        [0.17847428, 0.37378382, 0.51201963],
        [0.27469369, 0.38977735, 0.4280551 ],
        [0.77464143, 0.24271959, 0.25361009],
        [0.65298228, 0.53598226, 0.58605626],
        [0.98808658, 0.01048324, 0.30508715],
        [0.28378293, 0.83000529, 0.83065303],
        [0.09293132, 0.37767687, 0.09071906],
        [0.84283448, 0.46722462, 0.583765  ]])
    train_y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    test_x=np.array([[0.75875071, 0.53920321, 0.80437011],
        [0.34726175, 0.37140109, 0.95471962],
        [0.94263039, 0.1264374,  0.4432612 ],
        [0.80730767, 0.0223613,  0.4657955 ],
        [0.78047251, 0.6418516,  0.73110117],
        [0.69898289, 0.7972728,  0.81758147],
        [0.54495659, 0.46034209, 0.7237491 ],
        [0.04380754, 0.58298445, 0.64068301],
        [0.17194364, 0.80119221, 0.80196542],
        [0.1210299,  0.71753145, 0.84848576],
        [0.45617336, 0.06788977, 0.02430045],
        [0.02182345, 0.74759806, 0.13961117]])
    Submission_output = np.array([0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0])


    if check_array(
            ex_name, svm.one_vs_rest_svm,
            Submission_output, train_x, train_y, test_x):
        return

    log(green("PASS"), ex_name, "")


def check_compute_probabilities():
    ex_name = "Compute probabilities"
    n, d, k = 3, 5, 7
    X = np.arange(0, n * d).reshape(n, d)
    zeros = np.zeros((k, d))
    temp = 0.2
    exp_res = np.ones((k, n)) / k
    if check_array(
            ex_name, softmax.compute_probabilities,
            exp_res, X, zeros, temp):
        return

    theta = np.arange(0, k * d).reshape(k, d)
    softmax.compute_probabilities(X, theta, temp)
    exp_res = np.zeros((k, n))
    exp_res[-1] = 1
    if check_array(
            ex_name, softmax.compute_probabilities,
            exp_res, X, theta, temp):
        return

    log(green("PASS"), ex_name, "")

def check_compute_cost_function():
    ex_name = "Compute cost function"
    n, d, k = 3, 5, 7
    X = np.arange(0, n * d).reshape(n, d)
    Y = np.arange(0, n)
    zeros = np.zeros((k, d))
    temp = 0.2
    lambda_factor = 0.5
    exp_res = 1.9459101490553135
    if check_real(
            ex_name, softmax.compute_cost_function,
            exp_res, X, Y, zeros, lambda_factor, temp):
        return
    log(green("PASS"), ex_name, "")

def check_run_gradient_descent_iteration():
    ex_name = "Run gradient descent iteration"
    n, d, k = 3, 5, 7
    X = np.arange(0, n * d).reshape(n, d)
    Y = np.arange(0, n)
    zeros = np.zeros((k, d))
    alpha = 2
    temp = 0.2
    lambda_factor = 0.5
    exp_res = np.zeros((k, d))
    exp_res = np.array([
       [ -7.14285714,  -5.23809524,  -3.33333333,  -1.42857143, 0.47619048],
       [  9.52380952,  11.42857143,  13.33333333,  15.23809524, 17.14285714],
       [ 26.19047619,  28.0952381 ,  30.        ,  31.9047619 , 33.80952381],
       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286]
    ])

    if check_array(
            ex_name, softmax.run_gradient_descent_iteration,
            exp_res, X, Y, zeros, alpha, lambda_factor, temp):
        return
    softmax.run_gradient_descent_iteration(X, Y, zeros, alpha, lambda_factor, temp)
    log(green("PASS"), ex_name, "")

def check_update_y():
    ex_name = "Update y"
    train_y = np.arange(0, 10)
    test_y = np.arange(9, -1, -1)
    exp_res = (
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            np.array([0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
            )
    if check_tuple(
            ex_name, softmax.update_y,
            exp_res, train_y, test_y):
        return
    log(green("PASS"), ex_name, "")

###Correction note:  check_project_onto_PC fucntion have been modified since release.
def check_project_onto_PC():
    ex_name = "Project onto PC"
    X = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9],
        [4, 8, 12],
    ]);
    x_centered, feature_means = features.center_data(X)
    pcs = features.principal_components(x_centered)
    exp_res = np.array([
        [5.61248608, 0],
        [1.87082869, 0],
        [-1.87082869, 0],
        [-5.61248608, 0],
    ])
    n_components = 2
    if check_array(
            ex_name, features.project_onto_PC,
            exp_res, X, pcs, n_components, feature_means):
        return
    log(green("PASS"), ex_name, "")

def check_polynomial_kernel():
    ex_name = "Polynomial kernel"
    n, m, d = 3, 5, 7
    c = 1
    p = 2
    X = np.random.random((n, d))
    Y = np.random.random((m, d))
    try:
        K = kernel.polynomial_kernel(X, Y, c, d)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    for i in range(n):
        for j in range(m):
            exp = (X[i] @ Y[j] + c) ** d
            got = K[i][j]
            if (not equals(exp, got)):
                log(
                    red("FAIL"), ex_name,
                    ": values at ({}, {}) do not match. Expected {}, got {}"
                    .format(i, j, exp, got)
                )
    log(green("PASS"), ex_name, "")

def check_rbf_kernel():
    ex_name = "RBF kernel"
    n, m, d = 3, 5, 7
    gamma = 0.5
    X = np.random.random((n, d))
    Y = np.random.random((m, d))
    try:
        K = kernel.rbf_kernel(X, Y, gamma)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    for i in range(n):
        for j in range(m):
            exp = np.exp(-gamma * (np.linalg.norm(X[i] - Y[j]) ** 2))
            got = K[i][j]
            if (not equals(exp, got)):
                log(
                    red("FAIL"), ex_name,
                    ": values at ({}, {}) do not match. Expected {}, got {}"
                    .format(i, j, exp, got)
                )
    log(green("PASS"), ex_name, "")


def main():
    log(green("PASS"), "Import mnist project")
    try:
        check_get_mnist()
        check_closed_form()
        check_svm()
        check_compute_probabilities()
        check_compute_cost_function()
        check_run_gradient_descent_iteration()
        check_update_y()
        check_project_onto_PC()
        check_polynomial_kernel()
        check_rbf_kernel()
    except Exception:
        log_exit(traceback.format_exc())

if __name__ == "__main__":
    main()
