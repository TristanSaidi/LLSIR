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

    def unit_gradient(self, t, dict, kink_mode="right"):
        """
        Exact unit tangent on the (possibly smoothed) open polyline.
        For interior points of a segment: constant unit direction.
        At vertices (kinks): choose left/right/average via kink_mode.

        kink_mode: {"right", "left", "avg"}
          - "right": use outgoing segment direction
          - "left" : use incoming segment direction
          - "avg"  : normalize sum of incoming+outgoing directions (if nonzero)
        """
        pts = _polygon_points(dict)  # vertices after optional smoothing

        if len(pts) == 0:
            raise ValueError("vertices must contain at least one point")
        if len(pts) == 1:
            return np.zeros(2, dtype=float)

        t = float(np.clip(t, 0.0, 1.0))

        segs = pts[1:] - pts[:-1]
        lens = np.linalg.norm(segs, axis=1)
        total = lens.sum()
        if total == 0:
            return np.zeros(2, dtype=float)

        s = t * total
        cum = np.cumsum(lens)

        # Find segment index k such that s lies in segment k
        # (k in {0,...,m-2})
        k = int(np.searchsorted(cum, s, side="right"))
        k = min(max(k, 0), len(lens) - 1)

        # Check if s is exactly at a vertex (within numerical tolerance)
        # Vertex j corresponds to arc-length cum[j-1] (for j>=1)
        # Here, "at kink" means s == cum[k-1] or s == cum[k].
        tol = dict.get("kink_tol", 1e-12) * total

        def unit_dir(idx):
            """Unit direction of segment idx, skipping zero-length segments."""
            if idx < 0 or idx >= len(lens):
                return None
            if lens[idx] <= 0:
                return None
            return segs[idx] / lens[idx]

        # Endpoint conventions
        if s <= 0 + tol:
            # at start
            d = unit_dir(_next_nonzero_segment(lens, start=0, step=+1))
            return d if d is not None else np.zeros(2, dtype=float)
        if s >= total - tol:
            # at end
            d = unit_dir(_next_nonzero_segment(lens, start=len(lens)-1, step=-1))
            return d if d is not None else np.zeros(2, dtype=float)

        # Interior: if not near a kink, return the current segment direction
        s_left = 0.0 if k == 0 else cum[k - 1]
        s_right = cum[k]

        at_left_kink = (abs(s - s_left) <= tol) and (k > 0)
        at_right_kink = (abs(s - s_right) <= tol) and (k < len(lens) - 1)

        if not (at_left_kink or at_right_kink):
            d = unit_dir(k)
            if d is not None:
                return d
            # if this segment is zero-length (rare), walk to nearest nonzero
            d = unit_dir(_next_nonzero_segment(lens, start=k, step=+1))
            if d is None:
                d = unit_dir(_next_nonzero_segment(lens, start=k, step=-1))
            return d if d is not None else np.zeros(2, dtype=float)

        # Kink handling (vertex)
        # Incoming segment is k-1 (if at_left_kink), outgoing is k (or k+1 if at_right_kink)
        if at_left_kink:
            in_idx = _next_nonzero_segment(lens, start=k-1, step=-1)
            out_idx = _next_nonzero_segment(lens, start=k, step=+1)
        else:  # at_right_kink
            in_idx = _next_nonzero_segment(lens, start=k, step=-1)
            out_idx = _next_nonzero_segment(lens, start=k+1, step=+1)

        din = unit_dir(in_idx) if in_idx is not None else None
        dout = unit_dir(out_idx) if out_idx is not None else None

        if kink_mode == "right":
            return dout if dout is not None else (din if din is not None else np.zeros(2, dtype=float))
        if kink_mode == "left":
            return din if din is not None else (dout if dout is not None else np.zeros(2, dtype=float))
        if kink_mode == "avg":
            if din is None and dout is None:
                return np.zeros(2, dtype=float)
            if din is None:
                return dout
            if dout is None:
                return din
            g = din + dout
            ng = np.linalg.norm(g)
            return g / ng if ng > 0 else dout  # fallback if opposite directions
        raise ValueError("kink_mode must be one of {'right','left','avg'}")


def _next_nonzero_segment(lens, start, step):
    """Return first index i starting at 'start' stepping by 'step' with lens[i] > 0, else None."""
    i = start
    while 0 <= i < len(lens):
        if lens[i] > 0:
            return i
        i += step
    return None


def _polygon_points(dict):
    """Return polyline points after optional smoothing (same logic as polygon_fn)."""
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
    return pts


def polygon_fn(t, dict):
    """
    (Unchanged) returns point on (possibly smoothed) open polyline at arc-length parameter t.
    """
    pts = _polygon_points(dict)

    t = float(np.clip(t, 0.0, 1.0))
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
