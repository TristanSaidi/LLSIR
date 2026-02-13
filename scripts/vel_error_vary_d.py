import numpy as np
import argparse
import os

from src.curves import identity, sin_curve, circle, polygon, gamma
from src.llsir import LLSIR
from src.link import link_id


# Keep exactly your curve keys
curve_map = {
    "identity": identity,
    "sin_curve": sin_curve,
    "circle": circle,
    "polygon": polygon,
}

# Your previous defaults (used as base across d)
epsilon_map = {
    "identity": 2,
    "sin_curve": 1,
    "circle": 1,
    "polygon": 1,
}

def kth_nn_distance(X, x0, k=10):
    """
    Euclidean distance from x0 to its k-th nearest neighbor in X (excluding itself implicitly).
    X: (N,d), x0: (d,)
    """
    dists = np.linalg.norm(X - x0[None, :], axis=1)
    # k-th nearest => index k-1 in 0-based sorted order
    return float(np.partition(dists, k-1)[k-1])

# Base locations; we will lift to d by padding (ONLY for x0; no curve embedding)
# If you truly want no padding at all, provide a d-dim x0 elsewhere.
loc_map_2d = {
    "identity": np.array([0.7, 0.5]),
    "sin_curve": np.array([0.6, 0.5]),
    "circle": np.array([0.8, 0.5]),
    "polygon": np.array([0.7, 0.5]),
}


def embed_x0(x0_2d, d, fill=0.5):
    """Pad a 2D x0 to R^d (only for x0)."""
    x0 = np.full(d, fill, dtype=float)
    x0[:2] = x0_2d
    return x0


def fit_loglog_slope(Ns, ys, eps=1e-12):
    """Slope from polyfit(log N, log y). Raw slope (signed)."""
    ys = np.asarray(ys, dtype=float)
    ys = np.maximum(ys, eps)  # protect log(0)
    coeffs = np.polyfit(np.log(Ns), np.log(ys), 1)
    return float(coeffs[0])

def lift_beta(beta_2d, d):
    beta = np.zeros(d, dtype=float)
    beta[:2] = beta_2d
    return beta

def format_pm(mean, std, digits=2):
    """
    Add a phantom minus sign for nonnegative means so that \pm stacks.
    """
    sign_pad = r"\phantom{-}" if mean >= 0 else ""
    return rf"${sign_pad}{mean:.{digits}f} \pm {std:.{digits}f}$"


def make_latex_table(results, ds, caption):
    """
    results[d][curve_key] = (mean_slope, std_slope)
    Column order must match your table: Identity, Circle, Sinusoid, Polygon
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


def run_all(args):
    rng = np.random.default_rng(args.seed)

    ds = list(range(2, 11, 2))  # 2,4,6,8,10
    Ns = np.arange(args.N_start, args.N_end + 1, args.N_step)
    Nmax = Ns[-1]

    results = {d: {} for d in ds}

    for d in ds:
        print(f"\n========== d = {d} ==========")
        # NOTE: this assumes your codebase supports working in ambient dimension d.
        # We only change X and x0 dimensions here.
        for curve_key, curve_factory in curve_map.items():
            print(f"\n--- curve = {curve_key} ---")

            # IMPORTANT: gamma expects a factory f such that f() returns a t->R^d callable
            curve = gamma(curve_factory)

            epsilon_0 = epsilon_map[curve_key]
            x_0 = embed_x0(loc_map_2d[curve_key], d, fill=args.x0_fill)

            # Fix randomness across N for fairness: per trial, we store a full Nmax sample
            # Use independent but deterministic seed per (d, curve_key)
            local_seed = (hash((args.seed, d, curve_key)) % (2**32 - 1))
            local_rng = np.random.default_rng(local_seed)
            X_full_trials = [local_rng.random((Nmax, d)) for _ in range(args.trials)]

            trial_slopes = []
            for t, X_full in enumerate(X_full_trials, start=1):
                errs = []
                eps0_trial = kth_nn_distance(X_full[:Ns[0], :], x_0, k=15) * epsilon_0
                
                for N in Ns:
                    X = X_full[:N, :]
                    
                    # epsilon scaling with ambient dimension d
                    epsilon = eps0_trial * (N / Ns[0]) ** (-1.0 / (d + 4))
                    print(f"N={N}, epsilon={epsilon:.4f}")
                    llsir = LLSIR(X, curve, link_id, epsilon=epsilon, sigma=args.sigma)
                    beta_hat, _y_hat, _y_true = llsir.fit(x_0)

                    # True tangent at projection point (uses your curve object)
                    _, tproj = curve.project(x_0)
                    true_beta_2d = curve.unit_gradient(tproj)
                    true_beta = lift_beta(true_beta_2d, d)

                    sq_err = min(
                        np.linalg.norm(beta_hat - true_beta) ** 2,
                        np.linalg.norm(beta_hat + true_beta) ** 2,
                    )
                    errs.append(sq_err)

                slope = fit_loglog_slope(Ns, errs)  # RAW slope (signed)
                trial_slopes.append(slope)

                if args.verbose and (t % max(1, args.trials // 5) == 0):
                    print(f"  trial {t:>3}/{args.trials}: slope={slope:.3f}")

            trial_slopes = np.asarray(trial_slopes, dtype=float)
            mean_slope = float(trial_slopes.mean())
            std_slope = float(trial_slopes.std(ddof=1)) if len(trial_slopes) > 1 else 0.0
            results[d][curve_key] = (mean_slope, std_slope)

            print(f"slope: mean={mean_slope:.3f}, std={std_slope:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    table_tex = make_latex_table(results, ds, caption=args.caption)
    out_path = os.path.join(args.out_dir, args.out_name)
    with open(out_path, "w") as f:
        f.write(table_tex)

    print(f"\nWrote LaTeX table to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate raw log-log slopes across ambient dimension and curve families.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=0.01)

    parser.add_argument("--N_start", type=int, default=500)
    parser.add_argument("--N_end", type=int, default=10000)
    parser.add_argument("--N_step", type=int, default=500)

    parser.add_argument("--x0_fill", type=float, default=0.5,
                        help="Padding value for x0 in coordinates 3..d (x0 itself only).")

    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--out_name", type=str, default="slope_table_vel.tex")
    parser.add_argument("--caption", type=str, default=r"\tls{data in the table is just filler for now}")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    run_all(args)
