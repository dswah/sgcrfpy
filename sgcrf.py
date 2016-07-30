# -*- coding: utf-8 -*-

from __future__ import division

from collections import namedtuple

import numpy as np
from numpy import random as rng
# from scipy import *
from sklearn.base import BaseEstimator
from progressbar import ProgressBar

from copy import deepcopy

def soft_thresh(r, w):
    return np.sign(w) * np.max(np.abs(w)-r, 0)

def check_pd(mat):
    # check for positive definiteness via cholesky decomposition
    try:
        return True, np.linalg.cholesky(mat)
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return False, None
    assert 1 == 0

def log(x):
    return np.log(x) if x > 0 else -np.inf

class SparseGaussianCRF(BaseEstimator):
    """
    GCRF models conditional probability density of x in R^p given y in R^q as
    p(y|x, Λ, Θ) = exp(-y' * Λ * y - 2 * x' * Θ * y) / Z(x)

    where Z(x) = c * |Λ|^-1 * exp(x' * Θ * Λ^-1 * Θ' * x)

    Inference in  GCRF gives:
    p(y|x) = N(-Θ * Λ^-1 * x, Λ^-1)

    Parameters
    ----------
    learning_rate : float, default 1.0
        step size

    lamL : float, default 0.01
        l1 regularization for the Λ matrix

    lamT : float, default 0.01
        l1 regularization for the Θ matrix

    References
    ----------

    Wytock and Kolter 2013
    Probabilistic Forecasting using Sparse Gaussian CRFs
    http://www.zicokolter.com/wp-content/uploads/2015/10/wytock-cdc13.pdf

    Wytock and Kolter 2013
    Sparse Gaussian CRFs Algorithms Theory and Application
    https://www.cs.cmu.edu/~mwytock/papers/gcrf_full.pdf

    McCarter and Kim 2015
    Large-Scale Optimization Algorithms for Sparse CGGMs
    http://arxiv.org/pdf/1509.04681.pdf

    McCarter and Kim 2016
    On Sparse Gaussian Chain Graph Models
    (info on Multi-Layer Sparse Gaussian CRFs)
    http://papers.nips.cc/paper/5320-on-sparse-gaussian-chain-graph-models.pdf

    Klinger and Tomanek 2007
    Classical Probabilistic Models and Conditional Random Fields
    http://www.scai.fraunhofer.de/fileadmin/images/bio/data_mining/paper/crf_klinger_tomanek.pdf

    Tong Tong Wu and Kenneth Lange 2008
    Coordinate Descent Algorithms for Lasso Penalized Regression
    http://arxiv.org/pdf/0803.3876.pdf

    """
    def __init__(self, learning_rate=1.0, lamL=1, lamT=1, n_iter=1000):
        self.lamL = lamL
        self.lamT = lamT
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.Lam = None
        self.Theta = None

        # stuff for line search
        self.beta = 0.5
        self.slack = 0.05

    @property
    def covariance(self):
        # TODO inverse is expensive
        # consider using Cholesky decomposition
        # especially if we are already computing it during line searching!
        return np.linalg.inv(self.Lam)

    def fit(self, X, Y):
        """TODO: Docstring for fit.

        Parameters
        ----------
        X : np.array, shape (n_samples, input_dimension)
        Y : np.array, shape (n_samples, output_dimension)

        Returns
        -------
        TODO

        """
        assert X.shape[0] == Y.shape[0], 'Inputs and Outputs must have the same number of observations'

        self.alt_newton_coord_descent(X=X, Y=Y)
        return self


    def loss(self, X, Y, Lam=None, Theta=None):
        if Lam is None:
            Lam = self.Lam
        if Theta is None:
            Theta = self.Theta

        n, p, q = self._problem_size(X, Y)
        FixedParams = namedtuple('FixedParams', ['Sxx', 'Sxy', 'Syy'])
        VariableParams = namedtuple('VariableParams', ['Sigma', 'Psi', 'Gamma'])

        fixed = FixedParams(Sxx=np.dot(X.T, X) / n,
                            Syy=np.dot(Y.T, Y) / n,
                            Sxy=np.dot(X.T, Y) / n)
        Sigma = self.covariance
        R = np.dot(np.dot(X, self.Theta), Sigma) / np.sqrt(n)
        vary = VariableParams(Sigma=Sigma,
                              Psi=np.dot(R.T, R),
                              Gamma=np.dot(np.dot(fixed.Sxx, self.Theta), Sigma))
        return self.neg_log_likelihood(Lam, Theta, fixed, vary)


    def check_gradient(self, fixed, vary):
        grad_lam = np.zeros_like(self.Lam)
        grad_theta = np.zeros_like(self.Theta)
        for i in range(self.Lam.shape[0]):
            for j in range(self.Lam.shape[1]):
                L = self.Lam.copy()
                run = 1e-10
                L[i,j] += run
                rise = self.neg_log_likelihood(L, self.Theta, fixed, vary) - self.neg_log_likelihood(self.Lam, self.Theta, fixed, vary)
                grad_lam[i,j] = rise/run

        for i in range(self.Theta.shape[0]):
            for j in range(self.Theta.shape[1]):
                T = self.Theta.copy()
                run = 1e-10
                T[i,j] += run
                rise = self.neg_log_likelihood(self.Lam, T, fixed, vary) - self.neg_log_likelihood(self.Lam, self.Theta, fixed, vary)
                grad_theta[i,j] = rise/run
        return grad_lam, grad_theta


    def neg_log_likelihood(self, Lam, Theta, fixed, vary):
        # compute the negative log-likelihood of the GCRF
        # NOTE it might be cool to keep track of this so that we can do early stopping!!
        return -log(np.linalg.det(Lam)) + \
                np.trace(np.dot(fixed.Syy, Lam) + \
                2*np.dot(fixed.Sxy.T, Theta) + \
                np.dot(vary.Psi, Lam))

    @staticmethod
    def l1_norm_off_diag(A):
        # let's speed this up later
        return np.linalg.norm(A, ord=1) - np.linalg.norm(A.diagonal(), ord=1)

    def l1_neg_log_likelihood(self, Lam, Theta, fixed, vary):
        return self.neg_log_likelihood( Lam, Theta, fixed, vary) + \
               self.lamL * self.l1_norm_off_diag(Lam) + \
               self.lamT * np.linalg.norm(Theta, ord=1)

    def neg_log_likelihood_wrt_Lam(self, Lam, fixed, vary):
        # compute the negative log-likelihood of the GCRF when Theta is fixed
        return -log(np.linalg.det(Lam)) + \
                np.trace(np.dot(fixed.Syy, Lam) + \
                np.dot(vary.Psi, Lam))

    def l1_neg_log_likelihood_wrt_Lam(self, Lam, fixed, vary):
        # regularized neg log loss
        return self.neg_log_likelihood_wrt_Lam(Lam, fixed, vary) + \
               self.lamL * np.linalg.norm(Lam - Lam.diagonal(), ord=1)

    # def neg_log_likelihood_wrt_Theta(self, Theta, fixed, vary):
    #     # compute the negative log-likelihood of the GCRF when Lamba is fixed
    #     return 2*np.dot(fixed.Sxy.T, Theta) + np.dot(vary.Sigma, vary.Psi)

    def grad_wrt_Lam(self, fixed, vary):
        return fixed.Syy - vary.Sigma - vary.Psi

    def grad_wrt_Theta(self, fixed, vary):
        return 2 * (fixed.Sxy + vary.Gamma)
        # 2 * Sxy + 2* Sxx Theta Sigma

    def active_set(self, fixed, vary):
        return (self.active_set_Lam(fixed, vary),
                self.active_set_Theta(fixed, vary))

    def active_set_Lam(self, fixed, vary):
        grad = self.grad_wrt_Lam(fixed, vary)
        assert np.allclose(grad, grad.T, 1e-3)
        return np.where((np.abs(np.triu(grad)) > self.lamL) | (self.Lam != 0))
        # return np.where((np.abs(grad) > self.lamL) | (~np.isclose(self.Lam, 0)))

    def active_set_Theta(self, fixed, vary):
        grad = self.grad_wrt_Theta(fixed, vary)
        return np.where((np.abs(grad) > self.lamT) | (self.Theta != 0))
        # return np.where((np.abs(grad) > self.lamT) | (~np.isclose(self.Theta, 0)))

    def _problem_size(self, X, Y):
        (n, p), q = X.shape, Y.shape[1]
        return n, p, q

    def check_descent(self, newton_lambda, alpha, fixed, vary):
        # check if we have made suffcient descent
        DLam = np.trace(np.dot(self.grad_wrt_Lam(fixed, vary), newton_lambda)) + \
               self.lamL * np.linalg.norm(self.Lam + newton_lambda, ord=1) - \
               self.lamL * np.linalg.norm(self.Lam, ord=1)

        nll_a = self.l1_neg_log_likelihood_wrt_Lam(self.Lam + alpha * newton_lambda, fixed, vary)
        nll_b = self.l1_neg_log_likelihood_wrt_Lam(self.Lam, fixed, vary) + alpha * self.slack * DLam

        return nll_a <= nll_b

    def line_search(self, newton_lambda, fixed, vary):
        # returns cholesky decomposition of Lambda and the learning rate
        alpha = self.learning_rate
        while True:
            pd, L = check_pd(self.Lam + alpha * newton_lambda)
            if pd and self.check_descent(newton_lambda, alpha, fixed, vary):
                # step is positive definite and we have sufficient descent
                break
                #TODO maybe want to return newt+alpha, to reuse computation
            alpha = alpha * self.beta
            # if alpha < 0.1:
            #   return L, alpha
        return L, alpha

    def lambda_newton_direction(self, active, fixed, vary, max_iter=1):
        delta = self.Lambda # warm start
        U = np.dot(delta, vary.Sigma)

        for _ in range(max_iter):
            for i, j in rng.permutation(np.array(active).T):
                if i > j:
                    continue
                sds = np.dot(np.dot(vary.Sigma, delta), vary.Sigma)
                pds = np.dot(np.dot(vary.Psi, delta), vary.Sigma)

                if i==j:
                    a = vary.Sigma[i,i] ** 2 + 2 * vary.Sigma[i,i] * vary.Psi[i,i]
                else:
                    a = (vary.Sigma[i, j] ** 2 + vary.Sigma[i, i] * vary.Sigma[j, j] +
                         vary.Sigma[i, i] * vary.Psi[j, j] + 2 * vary.Sigma[i, j] * vary.Psi[i, j] +
                         vary.Sigma[j, j] * vary.Psi[i, i])

                b = (fixed.Syy[i, j] - vary.Sigma[i, j] - vary.Psi[i, j] +
                     np.dot(vary.Sigma[i,:], U[:,j]) +
                     np.dot(vary.Psi[i,:], U[:,j]) +
                     np.dot(vary.Psi[j,:], U[:,i]))

                if i==j:
                    u = -b/a
                else:
                    c = self.Lam[i, j] + delta[i, j]
                    u = soft_thresh(self.lamL / a, c - b/a) - c
                    delta[j, i] += u
                delta[i, j] += u
                U[j, :] +=  u * self.vary.Sigma[i, :]
                U[i, :] +=  u * self.vary.Sigma[j, :]

        return delta

    def theta_coordinate_descent(self, active, fixed, vary, max_iter=1):
        V = np.dot(self.Theta, self.vary.Sigma)
        for _ in range(max_iter):
            for i, j in np.array(active).T:
                sts = np.dot(np.dot(fixed.Sxx, self.Theta), vary.Sigma)
                a = 2 * vary.Sigma[j, j] * fixed.Sxx[i, i]

                b = 2 * fixed.Sxy[i, j] + 2 * np.dot(fixed.Sxx[i,:], V[:,j])

                c = self.Theta[i, j]

                self.Theta[i, j] += soft_thresh(self.lamT / a, c - b/a) - c
                V[i,:] += u * self.vary.Sigma[j,:]

        return self.Theta


    def alt_newton_coord_descent(self, X, Y):
        """
        JK Trying to follow Calvin's algorithm, then will merge back into orig
        """
        n, p, q = self._problem_size(X, Y)

        FixedParams = namedtuple('FixedParams', ['Sxx', 'Sxy', 'Syy'])
        VariableParams = namedtuple('VariableParams', ['Sigma', 'Psi', 'Gamma'])

        fixed = FixedParams(Sxx=np.dot(X.T, X) / n,
                            Syy=np.dot(Y.T, Y) / n,
                            Sxy=np.dot(X.T, Y) / n)

        # debugging -------------
        # Sigma = self.covariance
        # R = np.dot(np.dot(X, self.Theta), Sigma) / np.sqrt(n)
        # vary = VariableParams(Sigma=Sigma,
        #                       Psi=np.dot(R.T, R),
        #                       Gamma=np.dot(np.dot(fixed.Sxx, self.Theta), Sigma))
        # print self.neg_log_likelihood(self.Lam, self.Theta, fixed, vary)
        # ------------------

        self.Theta = np.zeros((p, q))
        self.Lam = np.eye(q)

        from progressbar import ProgressBar
        pbar = ProgressBar()

        self.nll = []
        self.lnll = []
        self.nllwl = []
        self.lnllwl = []
        self.lams = []
        self.thetas = []
        self.lrs = []

        for it in pbar(range(self.n_iter)):
            self.lams.append(self.Lam.copy())
            self.thetas.append(self.Theta.copy())

            # update variable params
            Sigma = self.covariance
            R = np.dot(np.dot(X, self.Theta), Sigma) / np.sqrt(n)
            vary = VariableParams(Sigma=Sigma,
                                  Psi=np.dot(R.T, R),
                                  Gamma=np.dot(np.dot(fixed.Sxx, self.Theta), Sigma))

            self.nll.append(self.neg_log_likelihood(self.Lam, self.Theta, fixed, vary))

            self.lnll.append(self.l1_neg_log_likelihood(self.Lam, self.Theta, fixed, vary))
            self.nllwl.append(self.neg_log_likelihood_wrt_Lam(self.Lam, fixed, vary))
            self.lnllwl.append(self.l1_neg_log_likelihood_wrt_Lam(self.Lam, fixed, vary))

            # determine active sets
            active_Lam, active_Theta = self.active_set(fixed, vary)

            # solve D_lambda via coordinate descent
            newton_lambda = self.lambda_newton_direction(active_Lam, fixed, vary, max_iter=1)
            assert np.allclose(newton_lambda, newton_lambda.T, 1e-3)

            # line search for best step size
            learning_rate = self.learning_rate
            L, learning_rate = self.line_search(newton_lambda, fixed, vary)
            self.lrs.append(learning_rate)
            self.Lam = self.Lam.copy() + learning_rate * newton_lambda

            # update variable params
            Sigma = self.covariance
            R = np.dot(np.dot(X, self.Theta), Sigma) / np.sqrt(n)
            vary = VariableParams(Sigma=Sigma,
                                  Psi=np.dot(R.T, R),
                                  Gamma=np.dot(np.dot(fixed.Sxx, self.Theta), Sigma))

            # determine active sets
            active_Lam, active_Theta = self.active_set(fixed, vary)

            # solve theta
            self.Theta = self.theta_coordinate_descent(active_Theta, fixed, vary, max_iter=1)

    def sample(self, X, n=1, verbose=True):
        # Inference in  GCRF given by:
        # p(y|x) = N(-Θ * Λ^-1 * x, Λ^-1)
        Sigma = np.linalg.inv(self.Lam)
        mean = -np.dot(np.dot(Sigma, self.Theta.T), X.T)

        if mean.ndim == 1:
            mean = mean[:,None]
        means = mean

        pbar = ProgressBar()
        samples = []
        for mean in pbar(means.T):
            samples.append(np.random.multivariate_normal(mean, Sigma, n))
        return np.array(samples).squeeze()

    def predict(self, X, Y):
        pass

    def get_params(self, deep=True):
        # TODO should we include n_iter of alt newton as a parameter?
        return {'lamL': self.lamL,
                'lamT': self.lamT,
                'learning_rate': self.learning_rate}

    def set_parameters(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
