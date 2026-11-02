import os
import json
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = "outputs/multimesh_benchmark_suite"
summary_path = os.path.join(OUT_DIR, "multimesh_benchmark_summary.json")

with open(summary_path, "r") as f:
    records = json.load(f)

# Create an 8-row x 3-column master visual benchmark poster
fig, axes = plt.subplots(8, 3, figsize=(18, 32), facecolor='#0d0f12')

for row_idx, rec in enumerate(records):
    m_id = rec["id"]
    m_name = rec["name"]
    m_cat = rec["category"]
    comp_img_path = os.path.join(OUT_DIR, f"{m_id}_comparison.png")
    
    if os.path.exists(comp_img_path):
        full_img = Image.open(comp_img_path)
        w, h = full_img.size
        # The comparison image has 3 equal columns: Baseline, Enhanced Fine, Assembly L2
        col_w = w // 3
        img_base = full_img.crop((0, 0, col_w, h))
        img_enh = full_img.crop((col_w, 0, 2 * col_w, h))
        img_l2 = full_img.crop((2 * col_w, 0, w, h))
        
        # Col 0: Baseline
        axes[row_idx, 0].imshow(img_base)
        axes[row_idx, 0].set_title(f"{rec['name']}\n1. Baseline ({rec['baseline_parts']} Parts)", color='#94a3b8', fontsize=10.5, fontweight='bold', pad=4)
        axes[row_idx, 0].axis('off')
        
        # Col 1: Enhanced Fine L3
        axes[row_idx, 1].imshow(img_enh)
        axes[row_idx, 1].set_title(f"2. Enhanced Fine L3 (+{rec['parts_gain']} Parts)\n({rec['enhanced_parts']} Structural Parts)", color='#38bdf8', fontsize=10.5, fontweight='bold', pad=4)
        axes[row_idx, 1].axis('off')
        
        # Col 2: Assembly L2
        axes[row_idx, 2].imshow(img_l2)
        axes[row_idx, 2].set_title(f"3. Functional Assembly L2\n({rec['l2_assemblies']} Contiguous Assemblies)", color='#c084fc', fontsize=10.5, fontweight='bold', pad=4)
        axes[row_idx, 2].axis('off')

plt.suptitle("P3-SAM Multi-Mesh Benchmark: Baseline vs Enhanced Fine L3 vs Functional Assembly L2", fontsize=15, color='white', fontweight='bold', y=0.995)
plt.tight_layout()
master_path = os.path.join(OUT_DIR, "master_multimesh_progression_grid.png")
plt.savefig(master_path, dpi=180, bbox_inches='tight', facecolor='#0d0f12')
plt.close()
print(f"Master multi-mesh progression grid saved to: {master_path}")
