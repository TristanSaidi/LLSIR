import numpy as np
import matplotlib.pyplot as plt
from src.curves import *
from src.llsir import LLSIR
from src.link import *
import argparse
import os


curve_map = {
    'identity': identity,
    'quadratic': quadratic,
    'circle': circle,
    'polygon': polygon
}

def main(args):
    
    curve_fn = curve_map.get(args.curve, identity)

    curve = gamma(curve_fn)

    mean_sq_errors = []
    epsilon_0 = 0.2

    Ns = np.arange(500, 10001, 500)
    trials = 300

    # Fix randomness across N
    Nmax = Ns[-1]
    X_full = np.random.rand(Nmax, 2)

    # Fix x0 across N (optionally keep away from boundary)
    x_0s = np.random.rand(trials, 2) * 0.8 + 0.1  # keep away from boundary

    for N in Ns:
        print(f"Running LLSIR with N = {N}")
        X = X_full[:N]
        sq_errors = []
        for x_0 in x_0s:
            epsilon = epsilon_0 * (N / Ns[0])**(-1 / (x_0.shape[0] + 4))  # d=2, but see note below
            llsir = LLSIR(X, curve, link_id, epsilon=epsilon, sigma=0.01)
            beta, y_hat, y_true = llsir.fit(x_0)
            true_beta = curve.unit_gradient(curve.project(x_0)[1])
            sq_error = min(np.linalg.norm(beta - true_beta)**2, np.linalg.norm(beta + true_beta)**2)
            sq_errors.append(sq_error)
        mean_sq_errors.append(np.mean(sq_errors))
        print(f"Mean squared error over {trials} trials: {np.mean(sq_errors)}")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"], # Explicitly set standard LaTeX font
    })

    plt.figure()
    # plot log-log of sq_error vs N
    plt.loglog(Ns, mean_sq_errors, marker='o', markersize=3)
    plt.xlim(Ns[0] * 0.9, Ns[-1] * 1.1)
    plt.xlabel(r'Number of samples $n$')
    plt.ylabel(r"$\|\hat{v}'(x_0) - \gamma'(\Pi(x_0))\|_2^2$")
    # plot reference line of slope -1/(d + 4)
    d = x_0.shape[0]
    theoretic_slope = -4 / (d + 4)
    ref_Ns = np.array([Ns[0], Ns[-1]])
    ref_errors = mean_sq_errors[0] * (ref_Ns / Ns[0])**theoretic_slope
    plt.loglog(ref_Ns, ref_errors, linestyle='--', color='red', 
            label=rf'Theoretical slope, $-4/{{(d+4)}} = {theoretic_slope:.2f}$')    # fit line to data
    log_Ns = np.log(Ns)
    log_errors = np.log(mean_sq_errors)
    coeffs = np.polyfit(log_Ns, log_errors, 1)
    fitted_errors = np.exp(np.polyval(coeffs, log_Ns))
    plt.loglog(Ns, fitted_errors, linestyle=':', color='green', label=rf'Fitted slope, ${coeffs[0]:.2f}$')
    plt.legend()
    dir = 'outputs'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'outputs/vel_error_{args.curve}.png', dpi=1200)


if __name__ == "__main__":
    np.random.seed(42)
    argument_parser = argparse.ArgumentParser(description="Run LLSIR velocity error experiment.")
    argument_parser.add_argument('--curve', type=str, default='identity', help='Curve to use: identity, quadratic, circle')
    args = argument_parser.parse_args()
    main(args)
    