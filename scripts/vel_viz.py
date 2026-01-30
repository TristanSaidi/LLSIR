import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
from src.curves import *
from src.link import *
from src.llsir import LLSIR

kwargs = {
    "smooth": False,
    "smooth_iters": 5,
    "smooth_alpha": 0.05,
}

curve_map = {
    "identity" : identity,
    "sin_curve" : sin_curve,
    "circle" : circle,
    "polygon" : polygon
}

id_to_title = {
    "identity" : "Identity",
    "sin_curve" : "Sinusoid",
    "circle" : "Circle",
    "polygon" : "Polygon"
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Explicitly set standard LaTeX font
})

def main():

    # sample uniformly from [0, 1] x [0, 1]
    N = 2000
    x = np.random.rand(N)
    y = np.random.rand(N)
    X = np.vstack((x, y)).T

    # sample 25 evenly spaced points in [0.1, 0.9]^2 to be test points
    x_0s = np.array([[i, j] for i in np.linspace(0.1, 0.9, 6) for j in np.linspace(0.1, 0.9, 6)])


    for curve_name in curve_map.keys():
        print(f"Processing curve: {curve_name}")
        curve = gamma(curve_map[curve_name], dict=kwargs)
        llsir = LLSIR(X, curve, link_id, epsilon=0.1, sigma=0.01)
        betas = []
        y_hats = []
        y_trues = []
        for x_0 in x_0s:
            print(f"Fitting LLSIR at x_0 = {x_0}")
            beta, y_hat, y_true = llsir.fit(x_0)
            betas.append(beta)
            y_hats.append(y_hat)
            y_trues.append(y_true)
        betas = np.array(betas)
        y_hats = np.array(y_hats)

        # test LLSIR
        # plot data with color by Y value
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], cmap='viridis', s=0.2)
        # make aspect equal
        plt.gca().set_aspect('equal')
        curve.plot(color='blue')
        plt.axis('off');
        for i, x_0i in enumerate(x_0s):
            beta = betas[i]
            true_beta = curve.unit_gradient(curve.project(x_0[0])[1])
            print("mse between beta and true beta:", np.mean((beta - true_beta)**2))
            # draw beta vector at x_0i
            # plot as arrow
            plt.arrow(x_0i[0], x_0i[1], 0.075 * beta[0], 0.075 * beta[1], head_width=0.01, head_length=0.02, fc='red', ec='red')
        plt.scatter(
            x_0s[:, 0], x_0s[:, 1],
            color='black',
            s=10, edgecolors='black', linewidths=0.5
        )
        plt.axis('off');
        plt.title(rf"{id_to_title[curve_name]}", fontsize=16)
        plt.savefig(f"outputs/{curve_name}_velocities.png", bbox_inches='tight', dpi=1200)
        plt.close()
if __name__ == "__main__":
    main()