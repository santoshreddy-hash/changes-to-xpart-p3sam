import os
import numpy as np

face_ids_paths = {
    'Step 1 (Baseline 1-Mask)': 'outputs/benchmark_multimask/Original_1-Mask_NMS_0.7/face_ids.npy',
    'Step 2 (Multi-Mask + Stability)': 'outputs/benchmark_multimask/Improved_plus_Stability_NMS_0.7/face_ids.npy',
    'Step 3 (Graph Cut Refinement)': 'outputs/benchmark_graphcut/graphcut_mesh/face_ids.npy',
    'Step 4 (Second-Pass Prompting)': 'outputs/benchmark_secondpass/SecondPass_Plus100/face_ids.npy',
    'Step 5a (Hierarchical Fine L3)': 'outputs/benchmark_hierarchical/level3_fine_face_ids.npy',
    'Step 5b (Hierarchical Medium L2)': 'outputs/benchmark_hierarchical/constrained/level2_med_face_ids.npy',
    'Step 5c (Hierarchical Coarse L1)': 'outputs/benchmark_hierarchical/constrained/level1_coarse_face_ids.npy',
}

for name, p in face_ids_paths.items():
    if not os.path.exists(p):
        print(f"Missing: {p}")
        continue
    fids = np.load(p)
    valid = fids[fids >= 0]
    unq = np.unique(valid)
    print(f"{name:32s} | Path: {p:65s} | Valid Faces: {len(valid)} / {len(fids)} | Unique Parts: {len(unq)}")
