import numpy as np
import matplotlib.pyplot as plt

class gamma:
    # class for parametric curve
    def __init__(self, f, dict={}):
        self.f = f() # f: [0, 1] -> R^d
        self.dict = dict

    def plot(self, N=1000, **kwargs):
        t = np.linspace(0, 1, N)
        pts = np.array([self.f(ti, self.dict) for ti in t])
        # plot dotted curve
        plt.plot(pts[:, 0], pts[:, 1], linestyle='dotted', **kwargs)

    def project(self, p, N=1000):
        # project point p onto curve, return closest point on curve and parameter value
        t = np.linspace(0, 1, N)
        pts = np.array([self.f(ti, self.dict) for ti in t])
        dists = np.linalg.norm(pts - p, axis=1)
        idx = np.argmin(dists)
        return pts[idx], t[idx]
    
    def unit_gradient(self, t):
        return self.f.unit_gradient(t, self.dict)

 
class quadratic:
    def __call__(self, t, dict=None):
        # example curve: quadratic
        return np.array([t, 1 - t**2])
    def unit_gradient(self, t, dict=None):
        # derivative of quadratic curve
        dx_dt = 1
        dy_dt = -2 * t
        grad = np.array([dx_dt, dy_dt])
        norm = np.linalg.norm(grad)
        if norm == 0:
            print("ERROR: zero gradient at t =", t)
            return grad
        return grad / norm

class identity: 
    def __call__(self, t, dict=None):
        # example curve: identity
        return np.array([t, t])
    
    def unit_gradient(self, t, dict=None):
        # derivative of identity curve
        dx_dt = 1
        dy_dt = 1
        grad = np.array([dx_dt, dy_dt])
        norm = np.linalg.norm(grad)
        if norm == 0:
            print("ERROR: zero gradient at t =", t)
            return grad
        return grad / norm
    
class circle:
    def __call__(self, t, dict=None):
        # example curve: circle
        return np.array([0.5 + 0.4 * np.cos(2 * np.pi * t), 0.5 + 0.4 * np.sin(2 * np.pi * t)])
    def unit_gradient(self, t, dict=None):
        # derivative of circle curve
        dx_dt = -0.4 * 2 * np.pi * np.sin(2 * np.pi * t)
        dy_dt = 0.4 * 2 * np.pi * np.cos(2 * np.pi * t)
        grad = np.array([dx_dt, dy_dt])
        norm = np.linalg.norm(grad)
        if norm == 0:
            print("ERROR: zero gradient at t =", t)
            return grad
        return grad / norm
    
class sin_curve:
    def __call__(self, t, dict=None):
        # example curve: sine curve
        return np.array([t, 0.5 + 0.4 * np.sin(2 * np.pi * t)])
    def unit_gradient(self, t, dict=None):
        # derivative of sine curve
        dx_dt = 1
        dy_dt = 0.4 * 2 * np.pi * np.cos(2 * np.pi * t)
        grad = np.array([dx_dt, dy_dt])
        norm = np.linalg.norm(grad)
        if norm == 0:
            print("ERROR: zero gradient at t =", t)
            return grad
        return grad / norm
    
class polygon:
    def __call__(self, t, dict):
        return polygon_fn(t, dict)
    
    def unit_gradient(self, t, dict):
        # approximate derivative via finite differences
        h = 1e-3
        p1 = polygon_fn(t + h, dict)
        p0 = polygon_fn(t - h, dict)
        grad = (p1 - p0) / (2 * h)
        norm = np.linalg.norm(grad)
        if norm == 0:
            print("ERROR: zero gradient at t =", t)
            return grad
        return grad / norm

def polygon_fn(t, dict):
    """
    Open (non-closed) piecewise-linear polygonal chain parameterized by arc-length.
    Optionally smooth corners via Chaikin corner-cutting (produces a smooth polyline).

    Parameters
    ----------
    t : float
        Parameter in [0,1] (clamped).
    vertices : (m,2) array-like or None
        Vertices in order. If None, uses a default polyline.
    smooth : bool
        If True, apply Chaikin smoothing to round corners.
    smooth_iters : int
        Number of Chaikin iterations (>=0). More => smoother.
    smooth_alpha : float
        Corner-cutting parameter in (0, 0.5). Typical: 0.25.

    Returns
    -------
    np.ndarray, shape (2,)
        Point on the (possibly smoothed) polyline at parameter t.
    """
    vertices = dict.get("vertices", None)
    smooth = dict.get("smooth", False)
    smooth_iters = dict.get("smooth_iters", 3)
    smooth_alpha = dict.get("smooth_alpha", 0.05)
    if vertices is None:
        vertices = np.array([
            [0.15, 0.20],
            [0.85, 0.25],
            [0.75, 0.80],
            [0.35, 0.90],
            [0.10, 0.55],
        ], dtype=float)
    else:
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must be an array-like of shape (m, 2)")

    # Clamp t
    t = float(np.clip(t, 0.0, 1.0))

    # --- Optional smoothing: Chaikin corner-cutting (OPEN polyline, no last->first edge) ---
    pts = vertices
    if smooth:
        if not (0.0 < smooth_alpha < 0.5):
            raise ValueError("smooth_alpha must be in (0, 0.5)")
        iters = int(max(0, smooth_iters))
        for _ in range(iters):
            if len(pts) < 2:
                break
            new_pts = [pts[0]]  # keep endpoint for open curve
            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                Q = (1 - smooth_alpha) * p + smooth_alpha * q
                R = smooth_alpha * p + (1 - smooth_alpha) * q
                new_pts.extend([Q, R])
            new_pts.append(pts[-1])  # keep endpoint
            pts = np.asarray(new_pts, dtype=float)

    # --- Arc-length parameterization along the OPEN polyline ---
    if len(pts) == 0:
        raise ValueError("vertices must contain at least one point")
    if len(pts) == 1:
        return pts[0].copy()

    segs = pts[1:] - pts[:-1]
    lens = np.linalg.norm(segs, axis=1)
    total = lens.sum()
    if total == 0:
        return pts[0].copy()

    s = t * total
    cum = np.cumsum(lens)
    k = int(np.searchsorted(cum, s, side="right"))
    if k >= len(lens):
        return pts[-1].copy()

    s0 = 0.0 if k == 0 else cum[k - 1]
    u = 0.0 if lens[k] == 0 else (s - s0) / lens[k]
    return (1 - u) * pts[k] + u * pts[k + 1]
