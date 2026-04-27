import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

im1 = cv2.imread("c1.jpg")
im2 = cv2.imread("c2.jpg")
scale = 0.25
im1s = cv2.resize(im1, None, fx=scale, fy=scale)
im2s = cv2.resize(im2, None, fx=scale, fy=scale)
print(f"  Working image sizes: {im1s.shape[:2]}, {im2s.shape[:2]}")

# ---- 3(a)+(b): Manual Homography using DLT (method=0, no RANSAC) ----
pts1 = np.float32([
    [165, 43],   # top-left of board
    [236, 28],   # near USB connector top
    [241, 88],   # USB connector bottom
    [38,  55],   # power jack area
    [41, 218],   # bottom-left corner
    [222, 247],  # bottom-right area
])
pts2 = np.float32([
    [155, 65],
    [222, 44],
    [228, 108],
    [28,  75],
    [32,  238],
    [210, 268],
])

H_manual, _ = cv2.findHomography(pts1, pts2, method=0)
print(f"\n  Manual homography H (DLT):\n{np.round(H_manual, 4)}")
h2, w2 = im2s.shape[:2]
im1_warped_manual = cv2.warpPerspective(im1s, H_manual, (w2, h2))
diff_manual = cv2.absdiff(im1_warped_manual, im2s)
diff_manual_gray = cv2.cvtColor(diff_manual, cv2.COLOR_BGR2GRAY)
_, diff_thresh = cv2.threshold(diff_manual_gray, 30, 255, cv2.THRESH_BINARY)

# ADDED: Quantitative mean diff reporting
mean_diff_manual = diff_manual_gray.mean()
print(f"  Manual warp  |c2 - warp(c1)|  mean={mean_diff_manual:.2f}  max={diff_manual_gray.max()}")

# ---- 3(c)+(d): SIFT with median+Gaussian preprocessing ----
# CHANGED: Apply median filter BEFORE Gaussian for better noise removal
def preprocess_for_sift(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)        # ADDED: median filter first
    gray = cv2.GaussianBlur(gray, (5,5), 2)  # then Gaussian
    return gray

gray1 = preprocess_for_sift(im1s)
gray2 = preprocess_for_sift(im2s)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print(f"\n  SIFT keypoints: {len(kp1)} in c1, {len(kp2)} in c2")
bf = cv2.BFMatcher(cv2.NORM_L2)
raw_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
print(f"  Good SIFT matches after ratio test: {len(good_matches)}")
top = sorted(good_matches, key=lambda m: m.distance)[:60]
img_matches = cv2.drawMatches(im1s, kp1, im2s, kp2, top, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

pts1_sift = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
pts2_sift = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

H_sift, mask_sift = cv2.findHomography(pts1_sift, pts2_sift, cv2.RANSAC, 5.0)
n_inliers_sift = mask_sift.sum() if mask_sift is not None else 0
print(f"  SIFT homography inliers: {n_inliers_sift}/{len(good_matches)}")

im1_warped_sift = cv2.warpPerspective(im1s, H_sift, (w2, h2))
diff_sift = cv2.absdiff(im1_warped_sift, im2s)
diff_sift_gray = cv2.cvtColor(diff_sift, cv2.COLOR_BGR2GRAY)
_, diff_sift_thresh = cv2.threshold(diff_sift_gray, 30, 255, cv2.THRESH_BINARY)

# ADDED: Quantitative mean diff reporting for SIFT
mean_diff_sift = diff_sift_gray.mean()
print(f"  SIFT warp    |c2 - warp(c1)|  mean={mean_diff_sift:.2f}  max={diff_sift_gray.max()}")
print(f"\n  Improvement: {((mean_diff_manual - mean_diff_sift)/mean_diff_manual)*100:.1f}% reduction in mean diff")

# ---- Save all Q3 figures ----

# Figure 3a+b: Manual warped + diff
fig3a, axes = plt.subplots(1, 3, figsize=(15, 5))
fig3a.suptitle("Q3(a)(b): Manual Correspondences", fontsize=13, fontweight='bold')
axes[0].imshow(cv2.cvtColor(im2s, cv2.COLOR_BGR2RGB))
axes[0].set_title("c2 (reference)"); axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(im1_warped_manual, cv2.COLOR_BGR2RGB))
axes[1].set_title("c1 warped → c2 frame (manual H)"); axes[1].axis('off')
axes[2].imshow(diff_manual_gray, cmap='hot')
axes[2].set_title(f"|c2 - warp(c1)|\nmean={mean_diff_manual:.2f} max={diff_manual_gray.max()}")
axes[2].axis('off')
plt.tight_layout()
plt.savefig("q3_manual.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved q3_manual.png")

# Figure 3c: SIFT matches
fig3c, ax3c = plt.subplots(figsize=(16, 5))
ax3c.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
ax3c.set_title(
    f"Q3(c): SIFT + Lowe ratio matches (top 60 of {len(good_matches)} good matches)", 
    fontsize=12
)
ax3c.axis('off')
plt.tight_layout()
plt.savefig("q3_sift_matches.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved q3_sift_matches.png")

# Figure 3d: SIFT warped + diff
fig3d, axes = plt.subplots(1, 3, figsize=(15, 5))
fig3d.suptitle("Q3(d): SIFT-based Homography", fontsize=13, fontweight='bold')
axes[0].imshow(cv2.cvtColor(im2s, cv2.COLOR_BGR2RGB))
axes[0].set_title("c2 (reference)"); axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(im1_warped_sift, cv2.COLOR_BGR2RGB))
axes[1].set_title("c1 warped (SIFT+RANSAC)"); axes[1].axis('off')
axes[2].imshow(diff_sift_gray, cmap='hot')
axes[2].set_title(f"|c2 - warp(c1)| — SIFT\nmean={mean_diff_sift:.2f} max={diff_sift_gray.max()}")
axes[2].axis('off')
plt.tight_layout()
plt.savefig("q3_sift.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved q3_sift.png")

print("\nAll Q3 computations complete.")