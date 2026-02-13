import numpy as np
import argparse
import os

from src.curves import identity, sin_curve, circle, polygon, gamma
from src.llsir import LLSIR
from src.dwlls import DWLLS
from src.link import link_id


# ----------------------------
# Curve registry (same keys)
# ----------------------------
curve_map = {
    "identity": identity,
    "sin_curve": sin_curve,
    "circle": circle,
    "polygon": polygon,
}

# Same style of multiplicative curve-specific factor (you had epsilon_map in reg script)
# We keep it as a multiplier on the kNN distance, per your velocity script pattern.
epsilon_map = {
    "identity": 2,
    "sin_curve": 1,
    "circle": 1,
    "polygon": 1,
}

# Base 2D x0's; padded to R^d exactly like the velocity script
loc_map_2d = {
    "identity": np.array([0.7, 0.5]),
    "sin_curve": np.array([0.6, 0.5]),
    "circle": np.array([0.3, 0.5]),
    "polygon": np.array([0.7, 0.5]),
}


# ----------------------------
# Helpers (mirrors your vary-d velocity script)
# ----------------------------
def kth_nn_distance(X, x0, k=10):
    """
    Euclidean distance from x0 to its k-th nearest neighbor in X.
    X: (N,d), x0: (d,)
    """
    dists = np.linalg.norm(X - x0[None, :], axis=1)
    return float(np.partition(dists, k - 1)[k - 1])


def embed_x0(x0_2d, d, fill=0.5):
    """Pad a 2D x0 to R^d (only for x0)."""
    x0 = np.full(d, fill, dtype=float)
    x0[:2] = x0_2d
    return x0


def fit_loglog_slope(Ns, ys, eps=1e-12):
    """Slope from polyfit(log N, log y). Raw slope (signed)."""
    ys = np.asarray(ys, dtype=float)
    ys = np.maximum(ys, eps)
    coeffs = np.polyfit(np.log(Ns), np.log(ys), 1)
    return float(coeffs[0])


def format_pm(mean, std, digits=2):
    """
    Phantom minus for nonnegative means so \pm stacks.
    """
    sign_pad = r"\phantom{-}" if mean >= 0 else ""
    return rf"${sign_pad}{mean:.{digits}f} \pm {std:.{digits}f}$"


def make_latex_table(results, ds, caption):
    """
    results[d][curve_key] = (mean_slope, std_slope)
    Column order: Identity, Circle, Sinusoid, Polygon
    """
    header = r"""\begin{table*}[h]
\small
\centering
 \begin{tabular}{l | >{\centering\arraybackslash}p{0.15\textwidth}  >{\centering\arraybackslash}p{0.15\textwidth}  >{\centering\arraybackslash}p{0.15\textwidth} >{\centering\arraybackslash}p{0.15\textwidth} } 
  \toprule  & Identity &  Circle & Sinusoid & Polygon \\ 
\midrule 
 \hline
"""
    rows = []
    for d in ds:
        cells = []
        for key in ["identity", "circle", "sin_curve", "polygon"]:
            m, s = results[d][key]
            cells.append(format_pm(m, s, digits=2))
        rows.append(rf" $d = {d}$ & " + " & ".join(cells) + r" \\")
    footer = rf"""\bottomrule
 \end{{tabular}}
 \caption{{{caption}}}
\end{{table*}}
"""
    return header + "\n".join(rows) + "\n" + footer


# ----------------------------
# Main experiment
# ----------------------------
def run_all(args):
    ds = list(range(2, 11, 2))  # 2,4,6,8,10
    Ns = np.arange(args.N_start, args.N_end + 1, args.N_step)
    Nmax = Ns[-1]
    N0 = Ns[0]

    results = {d: {} for d in ds}

    for d in ds:
        print(f"\n========== d = {d} ==========")

        for curve_key, curve_factory in curve_map.items():
            print(f"\n--- curve = {curve_key} ---")

            curve = gamma(curve_factory)
            eps_mult = epsilon_map[curve_key]
            x_0 = embed_x0(loc_map_2d[curve_key], d, fill=args.x0_fill)

            # Fix randomness across N (per trial we keep Nmax samples), deterministic per (seed, d, curve)
            local_seed = (hash((args.seed, d, curve_key)) % (2**32 - 1))
            local_rng = np.random.default_rng(local_seed)
            X_full_trials = [local_rng.random((Nmax, d)) for _ in range(args.trials)]
            Y = [np.array([link_id(curve.project(Xi)[1]) for Xi in X_full_trial]) for X_full_trial in X_full_trials]
            Y_noisy = [Y_full + args.sigma * local_rng.standard_normal(size=Y_full.shape) for Y_full in Y]
            y_true_0 = link_id(curve.project(x_0)[1])

            trial_slopes = []
            for t, X_full in enumerate(X_full_trials, start=1):
                # Set "first epsilon" (at N0) to distance to kNN, times a curve-specific multiplier
                eps0_trial = kth_nn_distance(X_full[:N0, :], x_0, k=15) * eps_mult
                Y_full_noisy = Y_noisy[t - 1]
                errs = []
                for N in Ns:
                    X = X_full[:N, :]
                    Y = Y_full_noisy[:N]
                    # Regression script scaling: epsilon ~ n^{-1/(d+2)}
                    epsilon = eps0_trial * (N / N0) ** (-1.0 / (d + 2))

                    if args.print_eps:
                        print(f"N={N}, epsilon={epsilon:.4f}")

                    llsir = LLSIR(X, Y, epsilon=epsilon)
                    _beta, y_hat = llsir.fit(x_0)
                    sq_err = float((y_hat - y_true_0) ** 2)
                    errs.append(sq_err)

                slope = fit_loglog_slope(Ns, errs)  # RAW (signed)
                trial_slopes.append(slope)

                if args.verbose and (t % max(1, args.trials // 5) == 0):
                    print(f"  trial {t:>3}/{args.trials}: slope={slope:.3f}")

            trial_slopes = np.asarray(trial_slopes, dtype=float)
            mean_slope = float(trial_slopes.mean())
            std_slope = float(trial_slopes.std(ddof=1)) if len(trial_slopes) > 1 else 0.0
            results[d][curve_key] = (mean_slope, std_slope)

            print(f"slope: mean={mean_slope:.3f}, std={std_slope:.3f}")

            # Optional: DWLLS baseline only for identity (mirrors your original reg script logic)
            if args.run_dwlls and curve_key == "identity":
                # You didn't ask for DWLLS in the table, but this keeps parity with your old script
                dwlls_slopes = []
                for X_full in X_full_trials:
                    eps0_trial = kth_nn_distance(X_full[:N0, :], x_0, k=args.knn_k) * eps_mult
                    errs_dw = []
                    for N in Ns:
                        X = X_full[:N, :]
                        epsilon = eps0_trial * (N / N0) ** (-1.0 / (d + 2))
                        dwlls = DWLLS(X, curve, link_id, sigma=args.sigma)
                        _beta_dw, y_hat_dw, y_true_dw = dwlls.fit_regression(x_0)
                        errs_dw.append(float((y_hat_dw - y_true_dw) ** 2))
                    dwlls_slopes.append(fit_loglog_slope(Ns, errs_dw))
                dwlls_slopes = np.asarray(dwlls_slopes, dtype=float)
                print(
                    f"DWLLS(identity) slope: mean={dwlls_slopes.mean():.3f}, std={dwlls_slopes.std(ddof=1):.3f}"
                )

    os.makedirs(args.out_dir, exist_ok=True)
    table_tex = make_latex_table(results, ds, caption=args.caption)
    out_path = os.path.join(args.out_dir, args.out_name)
    with open(out_path, "w") as f:
        f.write(table_tex)

    print(f"\nWrote LaTeX table to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate raw log-log slopes for regression error across ambient dimension and curves."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=0.01)

    parser.add_argument("--N_start", type=int, default=500)
    parser.add_argument("--N_end", type=int, default=10000)
    parser.add_argument("--N_step", type=int, default=500)

    parser.add_argument("--x0_fill", type=float, default=0.5,
                        help="Padding value for x0 in coordinates 3..d (x0 itself only).")

    parser.add_argument("--print_eps", action="store_true",
                        help="Print epsilon for each N (very verbose).")

    parser.add_argument("--run_dwlls", action="store_true",
                        help="Also compute DWLLS slopes for identity (printed only, not in table).")

    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--out_name", type=str, default="slope_table_reg.tex")
    parser.add_argument("--caption", type=str, default=r"\tls{data in the table is just filler for now}")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    run_all(args)
