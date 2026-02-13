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

class DWLLS:
    def __init__(self, X, Y, sigma = 0.00):
        self.X = X
        self.Y = Y
        self.kde = gaussian_kde(X.T)
        self.sigma = sigma

    def density_ratio(self, x, x_0, sigma = np.eye(2) * 0.4):
        p_x = self.kde(x)
        p_gx = gaussian_pdf(x, mu=x_0, sigma=sigma)
        return p_gx / p_x
    
    def fit_velocity(self):
        # get \epsilon neighborhood
        X = self.X
        # center neighborhood points
        X_centered = X - X.mean(axis=0)
        self.X_centered = X_centered
        x_0 = X.mean(axis=0)

        # get density ratio at all points with gaussian centered at x_0
        ratios = np.array([self.density_ratio(Xi, x_0) for Xi in X])
        # do density weighted linear regression to estimate tangent
        # fit linear regression to neighborhood points weighted by density ratio
        reg = LinearRegression()
        # ratios are [n, 1], need to flatten to [n,]
        reg.fit(X_centered, self.Y, sample_weight=ratios.flatten())
        # get beta vector
        beta = reg.coef_
        # scale beta to unit length
        bnorm = np.linalg.norm(beta)
        if bnorm < 1e-5 or not np.isfinite(bnorm):
            print("Warning: beta norm is zero at x_0 =", x_0)
            return beta, None, None
        else:
            beta = beta / bnorm
        self.beta = beta
        return beta
    
    def fit_regression(self, loc):
        # project entire dataset onto self.beta
        X = self.X
        loc_centered = loc - X.mean(axis=0)
        if not hasattr(self, "beta"):
            _ = self.fit_velocity()

        projections = self.X_centered @ self.beta
        # nonparametric regression
        nonparam_reg = KernelReg(endog=self.Y, exog=projections, var_type='c', bw=[5e-1])
        # get fitted values
        loc_proj = np.array([loc_centered @ self.beta])
        yhat0 = nonparam_reg.fit(np.array(loc_proj))[0][0]
        return self.beta, yhat0