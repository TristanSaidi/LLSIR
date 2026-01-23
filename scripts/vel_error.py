import numpy as np
import matplotlib.pyplot as plt
from src.curves import *
from src.llsir import LLSIR
from src.link import *


curve = gamma(identity, dict=None)

Ns = np.arange(1000, 20001, 2000)

trials = 10
mean_sq_errors = []
var_sq_errors = []
x_0 = np.array([[0.8, 0.3]]) # test point

for N in Ns:
    sq_errors = []
    print(f"Running LLSIR with N = {N}")
    for t in range(trials):
        # sample uniformly from [0, 1] x [0, 1]
        x = np.random.rand(N)
        y = np.random.rand(N)
        X = np.vstack((x, y)).T


        llsir = LLSIR(X, curve, link_id, epsilon=0.1)
        beta, y_hat, y_true = llsir.fit(x_0[0])
        true_beta = curve.unit_gradient(curve.project(x_0[0])[1])

        sq_error = min(np.linalg.norm(beta - true_beta)**2, np.linalg.norm(beta + true_beta)**2)
        sq_errors.append(sq_error)
    mean_sq_error = np.mean(sq_errors)
    print(f"Mean squared error over {trials} trials: {mean_sq_error}")
    var_sq_error = np.var(sq_errors)
    mean_sq_errors.append(mean_sq_error)
    var_sq_errors.append(var_sq_error)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Explicitly set standard LaTeX font
})

plt.figure()
# plot log-log of sq_error vs N
plt.loglog(Ns, mean_sq_errors, marker='o', markersize=3)
plt.xlim(Ns[0] * 0.9, Ns[-1] * 1.1)
# plot variances as error bars
# 1. Calculate the standard deviation
standard_error = np.sqrt(var_sq_errors) / np.sqrt(len(x_0))
# 2. Calculate asymmetric log-error bars to keep them looking correct on a log scale
# This prevents the lower bar from trying to hit zero.
lower_error = mean_sq_errors - np.maximum(1e-10, mean_sq_errors - standard_error) # clamp to avoid <= 0
upper_error = standard_error

plt.errorbar(Ns, mean_sq_errors, 
             yerr=[np.zeros_like(upper_error), upper_error], # Only upper bars
             fmt='none', ecolor='black', alpha=0.5, capsize=3)
plt.xlabel(r'Number of samples $n$')
plt.ylabel(r"$\sum_{i = 1}^n\|\hat{v}'(x_i) - \gamma'(\Pi(x_i))\|_2^2$")
# plot reference line of slope -1/(d + 4)
d = x_0.shape[1]
theoretic_slope = -1 / (d + 4)
ref_Ns = np.array([Ns[0], Ns[-1]])
ref_errors = mean_sq_errors[0] * (ref_Ns / Ns[0])**theoretic_slope
plt.loglog(ref_Ns, ref_errors, linestyle='--', color='red', label=r'Theoretical slope, $-1/(d+4)$')
# fit line to data
log_Ns = np.log(Ns)
log_errors = np.log(mean_sq_errors)
coeffs = np.polyfit(log_Ns, log_errors, 1)
fitted_errors = np.exp(np.polyval(coeffs, log_Ns))
plt.loglog(Ns, fitted_errors, linestyle=':', color='green', label=f'Fitted slope {coeffs[0]:.2f}')
plt.legend()

plt.savefig('vel_error_identity.png', dpi=1200)