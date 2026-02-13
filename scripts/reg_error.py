import numpy as np
import matplotlib.pyplot as plt
from src.curves import *
from src.llsir import LLSIR
from src.dwlls import DWLLS
from src.link import *
import argparse
import os


curve_map = {
    'identity': identity,
    'sin_curve': sin_curve,
    'circle': circle,
    'polygon': polygon
}

id_to_name = {
    'identity': 'Identity',
    'sin_curve': 'Sinusoid',
    'circle': 'Circle',
    'polygon': 'Polygon'
}

epsilon_map = {
    'identity': 0.2,
    'sin_curve': 0.3,
    'circle': 0.3,
    'polygon': 0.3
}

loc_map = {
    'identity': np.array([0.7, 0.5]),
    'sin_curve': np.array([0.6, 0.5]),
    'circle': np.array([0.3, 0.5]),
    'polygon': np.array([0.7, 0.5])
}

def main(args):
    
    curve_fn = curve_map.get(args.curve, identity)

    curve = gamma(curve_fn)

    mean_sq_errors = []
    mean_sq_errors_dwlls = []
    epsilon_0 = epsilon_map.get(args.curve)

    trials = 50
    # bake randomness into Ns
    Ns = np.arange(500, 10001, 500)
    

    # Fix randomness across N
    Nmax = Ns[-1]
    X_full = [np.random.rand(Nmax, 2) for _ in range(trials)]
    Y_full = [np.array([link_id(curve.project(Xi)[1]) for Xi in X_full_i]) for X_full_i in X_full]
    Y_full_noisy = [Y_full_i + 0.01 * np.random.standard_normal(size=Y_full_i.shape) for Y_full_i in Y_full]
    # Fix x0 across N (optionally keep away from boundary)
    x_0 = loc_map.get(args.curve)
    y_true = link_id(curve.project(x_0)[1])
    for N in Ns:
        print(f"Running LLSIR with N = {N}")
        sq_errors = []
        sq_error_dwlls = []
        for t, X_full_i in enumerate(X_full):
            Y_full_i = Y_full_noisy[t]
            Y = Y_full_i[:N]
            X = X_full_i[:N, :]
            epsilon = epsilon_0 * (N / Ns[0])**(-1 / (x_0.shape[0] + 2))  # d=2, but see note below
            # epsilon = epsilon_0
            llsir = LLSIR(X, Y, epsilon=epsilon)
            beta, y_hat = llsir.fit(x_0)
            if curve_fn is identity:
                # for identity, compare to single index regression
                dwlls = DWLLS(X, Y)
                beta_dwlls, y_hat_dwlls = dwlls.fit_regression(x_0)
                sq_error_dwlls = (y_hat_dwlls - y_true)**2
            sq_error = (y_hat - y_true) ** 2
            sq_errors.append(sq_error)
        mean_sq_errors.append(np.mean(sq_errors))
        print(f"LLSIR mean squared error over {trials} trials: {np.mean(sq_errors)}")
        if curve_fn is identity:
            mean_sq_errors_dwlls.append(np.mean(sq_error_dwlls))
            print(f"DWLLS mean squared error over {trials} trials: {np.mean(sq_error_dwlls)}")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"], # Explicitly set standard LaTeX font
    })

    plt.figure()
    # plot log-log of sq_error vs N -- make line of best fit the same color
    plt.loglog(Ns, mean_sq_errors, marker='o', markersize=3, color='green')
    plt.xlim(Ns[0] * 0.9, Ns[-1] * 1.1)
    plt.xlabel(r'Number of samples $n$')
    plt.ylabel(r"$|\hat{f}(x_0) - f(\Pi(x_0))|^2$")
    # plot reference line of slope -2/(d + 2)
    d = x_0.shape[0]
    theoretic_slope = -2 / (d + 2)
    ref_Ns = np.array([Ns[0], Ns[-1]])
    ref_errors = mean_sq_errors[0] * (ref_Ns / Ns[0])**theoretic_slope
    plt.loglog(ref_Ns, ref_errors, linestyle='--', color='red', 
            label=rf'Theoretical slope, $-2/{{(d+2)}} = {theoretic_slope:.2f}$')    # fit line to data
    log_Ns = np.log(Ns)
    log_errors = np.log(mean_sq_errors)
    coeffs = np.polyfit(log_Ns, log_errors, 1)
    fitted_errors = np.exp(np.polyval(coeffs, log_Ns))
    plt.loglog(Ns, fitted_errors, linestyle=':', color='green', label=rf'Fitted slope, ${coeffs[0]:.2f}$')
    if curve_fn is identity:
        # fit line to dwlls data
        log_errors_dwlls = np.log(mean_sq_errors_dwlls)
        coeffs_dwlls = np.polyfit(log_Ns, log_errors_dwlls, 1)
        fitted_errors_dwlls = np.exp(np.polyval(coeffs_dwlls, log_Ns))
        # plot dwlls data and fitted line
        plt.loglog(Ns, mean_sq_errors_dwlls, marker='s', markersize=3, color='orange')
        plt.loglog(Ns, fitted_errors_dwlls, linestyle=':', color='orange', label=rf'DWLLS Fitted slope, ${coeffs_dwlls[0]:.2f}$')
    plt.title(rf"{id_to_name.get(args.curve)}", fontsize=16)
        
    plt.legend()
    dir = 'outputs'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'outputs/reg_error_{args.curve}.png', dpi=1200)


if __name__ == "__main__":
    np.random.seed(42)
    argument_parser = argparse.ArgumentParser(description="Run LLSIR velocity error experiment.")
    argument_parser.add_argument('--curve', type=str, default='identity', help='Curve to use: identity, quadratic, circle')
    args = argument_parser.parse_args()
    main(args)
    