import cv2
import numpy as np
import matplotlib.pyplot as plt

print("\n=== Q2: Earring Size ===")
f_mm = 8          # focal length in mm
pixel_size_um = 2.2   # µm
d_mm = 720        # distance from lens to imaging plane (mm)
pixel_size_mm = pixel_size_um * 1e-3  # 0.0022 mm
f_px = f_mm / pixel_size_mm
print(f"  Focal length in pixels: {f_px:.2f} px")
img_ear = cv2.imread("earrings.jpg")
img_ear_gray = cv2.cvtColor(img_ear, cv2.COLOR_BGR2GRAY)
h_ear, w_ear = img_ear_gray.shape
print(f"  Earring image size: {w_ear}x{h_ear} px")
blurred = cv2.GaussianBlur(img_ear_gray, (7,7), 2)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                            param1=100, param2=30, minRadius=50, maxRadius=500)
if circles is not None:
    circles = np.round(circles[0, :]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)[:2]
    diameters_px = [2*c[2] for c in circles]
    print(f"  Detected circle radii (px): {[c[2] for c in circles]}")
    avg_diameter_px = np.mean(diameters_px)
else:
    avg_diameter_px = 350
    print(f"  Hough failed, using estimated diameter: {avg_diameter_px} px")
image_diameter_mm = avg_diameter_px * pixel_size_mm
real_diameter_mm = image_diameter_mm * (d_mm / f_mm)
real_diameter_cm = real_diameter_mm / 10
 
print(f"  Avg earring diameter in image: {avg_diameter_px:.1f} px = {image_diameter_mm:.3f} mm on sensor")
print(f"  Real earring diameter: {real_diameter_mm:.2f} mm  ({real_diameter_cm:.2f} cm)")
print(f"  (Using: real = img_size_mm × d/f = {image_diameter_mm:.3f} × {d_mm}/{f_mm})")
 
# Visualise
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
ax2.imshow(cv2.cvtColor(img_ear, cv2.COLOR_BGR2RGB))
if circles is not None:
    for c in circles:
        circle_patch = plt.Circle((c[0], c[1]), c[2], color='lime', fill=False, linewidth=2)
        ax2.add_patch(circle_patch)
ax2.set_title(f"Q2: Earring diameter ≈ {real_diameter_mm:.1f} mm\n"
              f"(f={f_mm}mm, d={d_mm}mm, pixel={pixel_size_um}µm)", fontsize=11)
ax2.axis('off')
plt.tight_layout()
plt.savefig("q2_earring.png", dpi=150, bbox_inches='tight')
plt.close()
print("Q2 plot saved.")