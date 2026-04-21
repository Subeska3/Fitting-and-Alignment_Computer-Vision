import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
 
# ============================================================
# QUESTION 1: Line Fitting with TLS and RANSAC
# ============================================================
 
D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()
 
def total_least_squares_line(x, y):
    xm, ym = x.mean(), y.mean()
    A = np.column_stack([x - xm, y - ym])
    _, _, Vt = np.linalg.svd(A)
    a, b = Vt[-1]          # normal direction (last row of Vt)
    c = -(a * xm + b * ym)
    norm = np.sqrt(a**2 + b**2)
    return a/norm, b/norm, c/norm
 
def line_residuals(x, y, a, b, c):
    """Perpendicular distance from each point to line ax+by+c=0."""
    return np.abs(a*x + b*y + c)   # a²+b²=1 so no division needed
 
# ---- 1(a) TLS on first line only ----
x1 = X_cols[:, 0]
y1 = Y_cols[:, 0]
a1, b1, c1 = total_least_squares_line(x1, y1)
print("=== Q1(a): TLS on line 1 ===")
print(f"  a={a1:.6f}, b={b1:.6f}, c={c1:.6f}")
print(f"  Line equation: {a1:.4f}x + {b1:.4f}y + {c1:.4f} = 0")
if abs(b1) > 1e-9:
    slope = -a1/b1
    intercept = -c1/b1
    print(f"  Slope-intercept: y = {slope:.4f}x + {intercept:.4f}")
 
# ---- 1(b) RANSAC to find three lines ----
def ransac_line(x, y, n_iter=1000, thresh=0.5, min_inliers=10):
    best_inliers = None
    best_params = None
    best_count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(len(x), 2, replace=False)
        a, b, c = total_least_squares_line(x[idx], y[idx])
        dists = line_residuals(x, y, a, b, c)
        inliers = dists < thresh
        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers
            best_params = (a, b, c)
    # Refit on all inliers
    a, b, c = total_least_squares_line(x[best_inliers], y[best_inliers])
    return a, b, c, best_inliers
 
remaining_mask = np.ones(len(X_all), dtype=bool)
lines_found = []
print("\n=== Q1(b): RANSAC for three lines ===")
for i in range(3):
    x_rem = X_all[remaining_mask]
    y_rem = Y_all[remaining_mask]
    a, b, c, local_inliers = ransac_line(x_rem, y_rem)
    # Map local inliers back to global indices
    global_indices = np.where(remaining_mask)[0]
    global_inliers = global_indices[local_inliers]
    remaining_mask[global_inliers] = False
    lines_found.append((a, b, c, global_inliers))
    print(f"  Line {i+1}: a={a:.6f}, b={b:.6f}, c={c:.6f}, inliers={len(global_inliers)}")
    if abs(b) > 1e-9:
        slope = -a/b
        intercept = -c/b
        print(f"           y = {slope:.4f}x + {intercept:.4f}")
 
# ---- Plot Q1 ----
fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("Q1: Line Fitting", fontsize=14, fontweight='bold')
 
# (a) TLS on first line
ax = axes[0]
ax.scatter(x1, y1, s=20, color='steelblue', alpha=0.6, label='Data (line 1)')
xs = np.linspace(x1.min(), x1.max(), 100)
ys = (-a1*xs - c1)/b1
ax.plot(xs, ys, 'r-', linewidth=2, label=f'TLS: y={-a1/b1:.3f}x+{-c1/b1:.3f}')
ax.set_title("(a) TLS on Line 1")
ax.legend(); ax.grid(True, alpha=0.3)
 
# (b) RANSAC three lines
ax = axes[1]
colors = ['#e74c3c', '#2ecc71', '#3498db']
labels = ['Line 1', 'Line 2', 'Line 3']
for i, (a, b, c, inliers) in enumerate(lines_found):
    ax.scatter(X_all[inliers], Y_all[inliers], s=15, color=colors[i], alpha=0.6, label=f'Inliers {labels[i]}')
    xs = np.linspace(X_all[inliers].min(), X_all[inliers].max(), 100)
    ys = (-a*xs - c)/b
    ax.plot(xs, ys, color=colors[i], linewidth=2)
ax.set_title("(b) RANSAC: Three Lines")
ax.legend(); ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig("q1_lines.png", dpi=150, bbox_inches='tight')
plt.close()
print("Q1 plot saved.")