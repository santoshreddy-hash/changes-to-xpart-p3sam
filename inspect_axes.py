import trimesh
import numpy as np

m = trimesh.load("outputs/complete_pipeline_showcase/Step4_SecondPass_TargetedPrompting.glb", force="mesh")
pts = m.vertices

print("Min coords:", pts.min(axis=0))
print("Max coords:", pts.max(axis=0))
print("Mean coords:", pts.mean(axis=0))

# The legs are the bottom. Let's find which extreme has the legs.
# In the original coordinates (without R_up):
# X range: [-0.95, 0.92] (span 1.87)
# Y range: [-0.51, 0.48] (span 0.99)
# Z range: [-0.30, 0.27] (span 0.57)

# Let's check which end has the smallest cross-sectional area (the legs)
# For X: min X vs max X
x_min_pts = pts[pts[:, 0] < -0.8]
x_max_pts = pts[pts[:, 0] > 0.8]
print("pts near X min:", len(x_min_pts), "pts near X max:", len(x_max_pts))

# For Y: min Y vs max Y
y_min_pts = pts[pts[:, 1] < -0.4]
y_max_pts = pts[pts[:, 1] > 0.4]
print("pts near Y min:", len(y_min_pts), "pts near Y max:", len(y_max_pts))

# For Z: min Z vs max Z
z_min_pts = pts[pts[:, 2] < -0.25]
z_max_pts = pts[pts[:, 2] > 0.22]
print("pts near Z min:", len(z_min_pts), "pts near Z max:", len(z_max_pts))
