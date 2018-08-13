import networkx as nx
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import csv
import timeit
import os
import scipy
import sys




def update_A(I, rho, W):
    return np.asarray(I - rho * W)


def update_beta(Y, X, A):
    # qr.solve(t(X)%*%X,t(X)%*%W1a%*%y
    out = np.matmul(np.matmul(np.matmul(scipy.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), A), Y)
    return np.asarray(out)


def update_mu(Y, X, A, beta):
    out = np.matmul(A, Y).transpose() - np.matmul(X, beta)
    return out


def update_sigmasq(mu):
    return np.matmul(mu.transpose(), mu) / len(mu)

def update_deviance(A, sigma):
    n = A.shape[0]
    return (n * (1 + np.log(2 * np.pi * sigma)) - 2 * np.log(np.abs(scipy.linalg.det(A))))


def loglik(x, X, Y, W):
    n = W.shape[0]
    I = np.identity(n)
    (sign, logdet) = np.linalg.slogdet(I - x[1] * W)
    det = sign * np.exp(logdet)

    return (
        n * np.log(2 * np.pi * x[0] * x[0]) + 1 / (x[0] * x[0]) *
        np.matmul((np.matmul((I - x[1] * W), Y) - np.matmul(X, np.asarray(x[2:]))).transpose(),
                  (np.matmul((I - x[1] * W), Y) - np.matmul(X, np.asarray(x[2:])))) -
        2 * np.log(np.abs(det))
    )


def jac_fun(x, X, Y, W):
    # x: sigma, rho, beta
    n = W.shape[0]
    I = np.identity(n)
    A = update_A(I, x[1], W)

    # sigma
    deriv_sigma = ((n/ (x[0] * x[0])) - 1 / (x[0] * x[0] * x[0] * x[0]) *
    np.matmul((np.matmul((I - x[1] * W), Y) - np.matmul(X, np.asarray(x[2:]))).transpose(),
    (np.matmul((I - x[1] * W), Y) - np.matmul(X, np.asarray(x[2:]))))
     ) * 2 * x[0]
    deriv_beta = 2 / (x[0] * x[0]) * (-1 * np.matmul(np.matmul(X.transpose(), A), Y) - np.matmul(np.matmul(X.transpose(), X), x[2:]))

    ei_value = scipy.linalg.eigvals(W)
    ei_value = np.real(ei_value)

    X_t_X = np.matmul(X.transpose(), X)
    X_t_X_inv = scipy.linalg.inv(X_t_X)

    # M = I - X(X'X)-1X'
    M = np.subtract(I, np.matmul(np.matmul(X, X_t_X_inv), X.transpose()))
    Y_t_M_Y = np.matmul(np.matmul(Y.transpose(), M), Y)
    Y_t_M_W_Y = np.matmul(np.matmul(np.matmul(Y.transpose(), M), W), Y)
    WY_t_M_W_Y = np.matmul(np.matmul(np.matmul(np.transpose(np.matmul(W, Y)), M), W), Y)

    deriv_rho = 0
    for ei in ei_value:
        deriv_rho +=  -1 * (ei) / (1 - x[1] * ei)
    deriv_rho = deriv_rho *  (-2 / n)
    deriv_rho += (-2 * Y_t_M_W_Y + 2 * x[1] * WY_t_M_W_Y) / (Y_t_M_Y - 2 * x[1] * Y_t_M_W_Y + x[1] * x[1] * WY_t_M_W_Y)
    deriv_rho = -2 * deriv_rho
    out = [deriv_sigma, deriv_rho]
    out.extend(deriv_beta)
    return np.asarray(out)

def init(new, n_node = 100, norm = False, symm = False):
    if(new == 1):
        if(symm == 1):
            G = nx.gnp_random_graph(n_node, 0.01, directed = False)
            # G = nx.connected_components(G)
            # print(G)
            W = np.asarray(nx.to_numpy_matrix(G))
            np.fill_diagonal(W, 0)
        else:
            G = nx.erdos_renyi_graph(n_node, 0.01, directed = True)
            W = np.asarray(nx.to_numpy_matrix(G))
            np.fill_diagonal(W, 0)
        if(norm == 1):
            row_sums = W.sum(axis=1)
            for i in range(len(row_sums)):
                if(row_sums[i] == 0):
                    row_sums[i] = 1
            W = W / row_sums[:, np.newaxis]

        n = W.shape[0]
        No_cate_var = 2
        No_cont_var = 2
        X1 = np.random.randint(2, size = (n, No_cate_var))
        X2 = 5 * np.random.random((n, No_cont_var))

        X = np.concatenate((X1, X2), axis = 1)

        Y = np.random.randint(2, size = (n, ))

        X = np.asarray(X, dtype = 'float32')
        Y = np.asarray(Y, dtype = 'float32')
        W = np.asarray(W, dtype = 'float32')
    else:
        Y_X = []
        W = []
        with open('feature.csv', "r") as infile:
            inreader = csv.reader(infile, dialect = "excel")
            for record in inreader:
                Y_X.append(record)
            infile.close()
        with open('adj.csv', "r") as infile:
            inreader = csv.reader(infile, dialect = "excel")
            for record in inreader:
                W.append(record)
            infile.close()
        Y_X = np.asarray(Y_X, dtype = 'float32')
        W = np.asarray(W, dtype = 'float32')
        Y = Y_X[:, 0]
        X = Y_X[:, 1:]

    n = W.shape[0]
    I = np.identity(n)
    rho_init = 0
    A_init = update_A(I, rho_init, W)
    beta_init = update_beta(Y, X, A_init)
    mu_init = update_mu(Y, X, A_init, beta_init)
    sigmasq_init = update_sigmasq(mu_init)
    # x0: sigma, rho, beta
    x0 = [np.sqrt(np.asscalar(sigmasq_init)), rho_init]
    x0.extend(beta_init)
    return x0, X, Y, W


def indirect_method(x0, X, Y, W, tol = 1e-10, disp = True, method = "BFGS", J = None, H = None):
    sigma_init = x0[0]
    n = W.shape[0]
    I = np.identity(n)
    A_init = update_A(I, x0[1], W)
    new_deviance = update_deviance(A_init, sigma_init)
    old_deviance = float('inf')
    x = x0
    count = 0
    while(np.abs(new_deviance - old_deviance) > tol):
        count += 1
        if(disp == True):
            print("Round: ", count)
            print("Loglikelihood: ", loglik(x, X, Y, W))
            print("Estimation: ", x)

        old_deviance = new_deviance
        res = minimize(loglik, x, args = (X, Y, W), method = method, options={'disp': disp}, jac = J)
        x = res.x

        new_A = update_A(I, x[1], W)
        new_beta = update_beta(Y, X, new_A)
        new_mu = update_mu(Y, X, new_A, new_beta)
        new_sigmasq = update_sigmasq(new_mu)
        x = [np.sqrt(np.asscalar(new_sigmasq)), x[1]]
        x.extend(new_beta)
        new_deviance = update_deviance(new_A, new_sigmasq)

        if(count > 1000):
            break

    return x

def direct_method(X, Y, W):
    N = W.shape[0]
    I = np.identity(N)
    X_t_X = np.matmul(X.transpose(), X)
    X_t_X_inv = scipy.linalg.inv(X_t_X)

    # M = I - X(X'X)-1X'
    M = np.subtract(I, np.matmul(np.matmul(X, X_t_X_inv), X.transpose()))
    # ei_value, ei_vector = np.linalg.eig(W)
    # ei_value = scipy.linalg.eigvals(W)
    # ei_value = np.real(ei_value)
    #
    # for ei in np.nditer(ei_value):
    #     print(np.real(ei))
    #     print(ei)
    # exit()

    Y_t_M_Y = np.matmul(np.matmul(Y.transpose(), M), Y)
    Y_t_M_W_Y = np.matmul(np.matmul(np.matmul(Y.transpose(), M), W), Y)
    WY_t_M_W_Y = np.matmul(np.matmul(np.matmul(np.transpose(np.matmul(W, Y)), M), W), Y)

    def f(x):
        (sign, logdet) = np.linalg.slogdet(I - x*W)
        det = sign * np.exp(logdet)
        sum = (-2 / N) * np.log(np.abs(det)) + np.log(Y_t_M_Y - 2 * x * Y_t_M_W_Y + x ** 2 * WY_t_M_W_Y)

        # sum = (-2 / N) * np.sum(np.log(1 - x * ei_value)) + np.log(Y_t_M_Y - 2 * x * Y_t_M_W_Y + x ** 2 * WY_t_M_W_Y)
        # sum = 0
        # for ei in np.nditer(ei_value):
        #     sum += np.log(1 - x * ei)
        # sum = sum * (-2 / N)
        # sum += np.log(Y_t_M_Y - 2 * x * Y_t_M_W_Y + x ** 2 * WY_t_M_W_Y)
        return sum

    abstol = 1e-6
    cons = ({'type': 'ineq', 'fun': lambda x: np.linalg.det(I - x * W)-abstol})
    res = minimize(f, 0)
    # res = minimize(f, 0, constraints = cons)
    rho = res.x[0]

    A = np.subtract(I, rho * W)
    A_Y = np.matmul(A, Y)
    beta = np.matmul(np.matmul(X_t_X_inv, X.transpose()), A_Y)

    omega = (1 / N) * (Y_t_M_Y - 2 * rho * Y_t_M_W_Y + rho ** 2 * WY_t_M_W_Y)
    # B = np.matmul(W, scipy.linalg.inv(A))
    # B_t_B = np.matmul(B.transpose(), B)
    # alpha = 0
    #
    # for ei in np.nditer(ei_value):
    #     alpha += (-1) * ei ** 2 / (1 - rho * ei) ** 2
    #
    # omega_X_t_X_B_X_beta = omega * np.matmul(np.matmul(np.matmul(X.transpose(), B), X), beta)
    # omega_X_t_X = omega * X_t_X
    #
    # k = beta.shape[0]
    # V_inv = np.zeros(shape=(k + 2, k + 2), dtype=np.complex_)
    #
    # for i in range(0, k + 2):
    #     for j in range(0, k + 2):
    #         if (i == 0 and j == 0):
    #             V_inv[i, j] = N / 2
    #         elif (i == 0 and j == 1):
    #             V_inv[i, j] = omega * np.trace(B)
    #         elif (i == 0 and j > 1):
    #             continue
    #         elif (i == 1 and j == 0):
    #             V_inv[i, j] = omega * np.trace(B)
    #         elif (i == 1 and j == 1):
    #             V_inv[i, j] = omega ** 2 * (np.trace(B_t_B) - alpha) + omega * (
    #             np.matmul(np.matmul(np.matmul(np.matmul(beta.transpose(), X.transpose()), B_t_B), X), beta))
    #         elif (i == 1 and j > 1):
    #             V_inv[i, j] = omega_X_t_X_B_X_beta[j - 2]
    #         elif (j == 0 and i > 1):
    #             continue
    #         elif (j == 1 and i > 1):
    #             V_inv[i, j] = omega_X_t_X_B_X_beta[i - 2]
    #         elif (i > 1 and j > 1):
    #             V_inv[i, j] = omega_X_t_X[i - 2, j - 2]

    # V = omega ** 2 * np.linalg.inv(V_inv)
    # for i in range(k + 2):
    #     print(np.real(np.sqrt(V[i, i])))

    # print("rho")
    # print(np.real(rho))
    #
    # print("omega")
    # print(np.real(omega))
    #
    # print("sigma")
    # print(np.sqrt(np.real(omega)))
    #
    # print("beta")
    # print(np.real(beta))

    out = [np.sqrt(np.real(omega)), float(np.real(rho))]
    out.extend(list(np.real(beta)))
    return out

def write_to_file(X, Y, W):
    Y = Y[:, None]
    feature = np.concatenate((Y, X), axis = 1)
    for x in feature:
        with open("feature.csv", "a") as outfile:
            outwriter = csv.writer(outfile, dialect = "excel", lineterminator = "\n")
            outwriter.writerow(x)

    for w in W:
        with open("adj.csv", "a") as outfile:
            outwriter = csv.writer(outfile, dialect = "excel", lineterminator = "\n")
            outwriter.writerow(w)

def direct_init_method(X, Y, W, method = "BFGS", J = None):
    x0 = direct_method(X, Y, W)
    # J = jac_fun(x0, X, Y, W)

    # disturb = 1e-6
    # for i in range(len(x0)):
    #     sign = np.random.randint(-1, 2)
    #     x0[i] += disturb * x0[i] * sign

    # print(x0)
    # x1 = indirect_method(x0, X, Y, W, disp = False, method = "Newton-CG", J = jac_fun)
    x1 = indirect_method(x0, X, Y, W, disp = False, method = method, J = J)

    # x1 = indirect_method(x0, X, Y, W, disp = False, method = "TNC", J = jac_fun)
    # x1 = indirect_method(x0, X, Y, W, disp = False, method = "BFGS", J = jac_fun)

    return x1


def write_f(file, *argv):
    with open(file, "a") as outfile:
        for arg in argv:
            outfile.write(str(arg))
        outfile.write("\n")
        outfile.close()


if __name__ == "__main__":
    # args: file_name, lower, upper, symm, norm
    # lwr = int(sys.argv[1])
    # upr = int(sys.argv[2])
    # symm = int(sys.argv[3])
    # norm = int(sys.argv[4])
    lwr = 100
    upr = 1000
    symm = 0
    norm = 1


    logfile = "/home/yongcheng/lnam/result2_{0}_{1}_{2}_{3}.txt".format(lwr, upr, symm, norm)
    # logfile = "C:\\Users\\Yongcheng\\Desktop\\result_{0}_{1}_{2}_{3}.txt".format(lwr, upr, symm, norm)
    # os.chdir(r"C:\Users\Yongcheng\Desktop\test matrix")
    for i in range(lwr, upr):
        write_f(logfile, "# Node: ", i)
        x0, X, Y, W = init(new = 1, n_node = i, symm = symm, norm = norm)
        # print(x0)
        # write_to_file(X, Y, W)
        t1 = timeit.default_timer()
        x = indirect_method(x0, X, Y, W, disp = False, method = "BFGS")
        t2 = timeit.default_timer()
        x2 = direct_method(X, Y, W)
        t3 = timeit.default_timer()
        # x3 = direct_init_method(X, Y, W, method = "BFGS")
        # t4 = timeit.default_timer()
        write_f(logfile, "x")
        write_f(logfile, x)
        write_f(logfile, x2)
        # write_f(logfile, x3)
        write_f(logfile, "loglik")
        write_f(logfile, loglik(x, X, Y, W))
        write_f(logfile, loglik(x2, X, Y, W))
        # write_f(logfile, loglik(x3, X, Y, W))
        write_f(logfile, "time")
        write_f(logfile, t2 - t1)
        write_f(logfile, t3 - t2)
        # write_f(logfile, t4 - t3)
        # x_r = [0.4925, 0.069435, 0.061245, 0.138165, 0.062968, 0.007544]
        # print(loglik(x_r, X, Y, W))