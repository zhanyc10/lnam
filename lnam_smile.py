import networkx as nx
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import csv
import timeit
import os
import scipy
import sys
import numdifftools as nd

class LnamModel:
    """Create a lnam computing instance"""
    def __init__(self, X, Y, W):
        """X is the feature matrix, Y is the dependent variable, W is the adjacency matrix/a list of adjacency matrix"""
        X = np.array(X)
        Y = np.array(Y)
        W = np.array(W)
        self.X = X
        self.Y = Y
        if(W.ndim == 2):
            self.W = np.array([W])
        else:
            self.W = W
        self._n = W[0].shape[0]
        self._I = np.identity(self._n)

    def update_A(self, rho):
        rho_W = np.empty_like(self.W[0])
        for i in range(len(rho)):
            rho_W += rho[i] * self.W[i]
        A = np.asarray(self._I - rho_W)
        return A


    def update_beta(self, A):
        # qr.solve(t(X)%*%X,t(X)%*%W1a%*%y
        out = np.matmul(np.matmul(np.matmul(scipy.linalg.inv(np.matmul(self.X.transpose(), self.X)), self.X.transpose()), A), self.Y)
        return np.asarray(out)


    def update_mu(self, beta, A):
        out = np.matmul(A, self.Y).transpose() - np.matmul(self.X, beta)
        return out


    def update_sigmasq(self, mu):
        return np.matmul(mu.transpose(), mu) / len(mu)

    def update_deviance(self, sigma, A):
        return (self._n * (1 + np.log(2 * np.pi * sigma)) - 2 * np.log(np.abs(np.linalg.det(A))))

    def loglik(self, x):
        num_rho = len(self.W)
        A = np.empty_like(self.W[0])
        for i in range(num_rho):
            A += x[i + 1] * self.W[i]
        A = self._I - A
        (sign, logdet) = np.linalg.slogdet(A)
        det = sign * np.exp(logdet)

        return (
            self._n * np.log(2 * np.pi * x[0] * x[0]) + 1 / (x[0] * x[0]) *
            np.matmul((np.matmul((A), self.Y) - np.matmul(self.X, np.asarray(x[num_rho + 1:]))).transpose(),
                      (np.matmul((A), self.Y) - np.matmul(self.X, np.asarray(x[num_rho + 1:])))) -
            2 * np.log(np.abs(det))
        )


    def indirect_method(self, x0, tol=1e-10, disp=True, method="BFGS", J=None, H=None):
        sigma_init = x0[0]
        num_rho = len(self.W)
        A_init = self.update_A(x0[1:num_rho + 1])
        new_deviance = self.update_deviance(sigma_init, A_init)
        old_deviance = float('inf')
        x = x0
        count = 0
        while (np.abs(new_deviance - old_deviance) > tol):
            count += 1
            if (disp == True):
                print("Round: ", count)
                print("Loglikelihood: ", self.loglik(x, self))
                print("Estimation: ", x)

            old_deviance = new_deviance
            res = minimize(self.loglik, x, args=(), method=method, options={'disp': disp}, jac=J)
            x = res.x

            new_A = self.update_A(x[1:num_rho + 1])
            new_deviance = self.update_deviance(x[0], new_A)
            if (count > 1000):
                break
        return x

    def direct_method(self):
        X_t_X = np.matmul(self.X.transpose(), self.X)
        X_t_X_inv = scipy.linalg.inv(X_t_X)

        # M = I - X(X'X)-1X'
        M = np.subtract(self._I, np.matmul(np.matmul(self.X, X_t_X_inv), self.X.transpose()))

        def f(x):
            num_rho = len(self.W)
            A = np.empty_like(self.W[0])
            for i in range(num_rho):
                A += x[i] * self.W[i]
            A = self._I - A
            z = np.matmul(A, self.Y)
            (sign, logdet) = np.linalg.slogdet(A)
            det = sign * np.exp(logdet)
            sum = (-2 / self._n) * np.log(np.abs(det)) + np.log(np.matmul(np.matmul(z.transpose(), M), z))
            return sum

        rho_init = [0] * len(self.W)
        res = minimize(f, rho_init, method="BFGS")
        rho = res.x


        A = self.update_A(rho)
        A_Y = np.matmul(A, self.Y)
        beta = np.matmul(np.matmul(X_t_X_inv, self.X.transpose()), A_Y)

        omega = (1 / self._n) * (np.matmul(np.matmul(A_Y.transpose(), M), A_Y))

        out = [np.sqrt(np.real(omega))]
        out.extend(np.real(rho))
        out.extend(list(np.real(beta)))
        return out

    def direct_init_method(self, method = "BFGS", J = None):
        x0 = self.direct_method(self.X, self.Y, self.W)
        x1 = self.indirect_method(self, x0, disp = False, method = method, J = J)
        return x1
