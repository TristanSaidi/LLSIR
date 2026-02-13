import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.kernel_regression import KernelReg


# gaussian pdf
def gaussian_pdf(x, mu, sigma = np.eye(2) * 0.01):
    d = x.shape[0]
    coeff = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(sigma) ** 0.5)
    exponent = -0.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu)
    return coeff * np.exp(exponent)

class LLSIR:
    def __init__(self, X, Y, epsilon):
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.kde = gaussian_kde(X.T)

    def density_ratio(self, x, x_0, sigma = 0.01):
        Sigma = np.eye(x.shape[0]) * sigma
        p_x = self.kde(x)
        p_gx = gaussian_pdf(x, mu=x_0, sigma=Sigma)
        return p_gx / p_x
    
    def fit(self, x_0):
        # get \epsilon neighborhood
        neighborhood_idx = np.linalg.norm(self.X - x_0, axis=1) < self.epsilon
        X_neighborhood = self.X[neighborhood_idx]
        Y_neighborhood = self.Y[neighborhood_idx]
        # center neighborhood points
        X_centered = (X_neighborhood - x_0)/self.epsilon

        # get density ratio at all points with gaussian centered at x_0
        ratios = np.array([self.density_ratio(Xi, x_0) for Xi in X_neighborhood])
        # do density weighted linear regression to estimate tangent
        # fit linear regression to neighborhood points weighted by density ratio
        reg = LinearRegression()
        # ratios are [n, 1], need to flatten to [n,]
        reg.fit(X_centered, Y_neighborhood, sample_weight=ratios.flatten())
        # get beta vector
        beta = reg.coef_
        # scale beta to unit length
        bnorm = np.linalg.norm(beta)
        if bnorm < 1e-5 or not np.isfinite(bnorm):
            print("Warning: beta norm is zero at x_0 =", x_0)
            return beta, None, None
        else:
            beta = beta / bnorm
        # project centered points onto beta
        projections = X_centered @ beta
        self.projections = projections
        self.Y_neighborhood = Y_neighborhood
        if projections.size  == 0:
            print("Warning: too few points for KernelReg:", projections.size)
            return beta, None, None
        # nonparametric regression of projections to Y_neighborhood
        if np.std(projections) < 1e-5:
            print("Warning: projections nearly constant; skipping KernelReg.")
            return beta, float(np.mean(Y_neighborhood))
        else:
            nonparam_reg = KernelReg(endog=Y_neighborhood, exog=projections, var_type='c', bw=[5e-1])
        # get fitted values
        yhat0 = nonparam_reg.fit(np.array([0.0]))[0][0]
        return beta, yhat0
    