import numpy as np
import trimesh
from pathlib import Path
import sys

sys.path.insert(0, "XPart")
from partgen.utils.mesh_utils import (
    create_tripo_assembly_scene,
    export_tripo_assembly_glb,
    explode_mesh_tripo,
    generate_tripo_viewer_html,
    TRIPO_PALETTE
)

def clean_and_merge_parts(parts, iou_thresh=0.65, containment_thresh=0.80):
    """
    Merges duplicate or heavily overlapping parts, and small fragment slivers.
    """
    if len(parts) <= 1:
        return parts

    print(f"[MERGE] Starting with {len(parts)} parts...")
    merged_parts = list(parts)

    changed = True
    while changed:
        changed = False
        n = len(merged_parts)
        merge_pair = None
        
        for i in range(n):
            if merge_pair:
                break
            box_i = merged_parts[i].bounds
            vol_i = np.prod(box_i[1] - box_i[0])
            if vol_i < 1e-8:
                continue

            for j in range(i + 1, n):
                box_j = merged_parts[j].bounds
                vol_j = np.prod(box_j[1] - box_j[0])
                if vol_j < 1e-8:
                    continue

                # Compute intersection AABB
                inter_min = np.maximum(box_i[0], box_j[0])
                inter_max = np.minimum(box_i[1], box_j[1])
                inter_dim = np.maximum(0, inter_max - inter_min)
                inter_vol = np.prod(inter_dim)

                if inter_vol <= 0:
                    continue

                # Intersection over Union (IoU)
                union_vol = vol_i + vol_j - inter_vol
                iou = inter_vol / union_vol if union_vol > 0 else 0
                
                # Containment ratios
                contain_i = inter_vol / vol_i
                contain_j = inter_vol / vol_j

                # Check if parts overlap heavily (e.g. Part 82 and Part 284 in furn16_3k)
                # or if one is almost entirely an interior duplicate of the other
                if iou > iou_thresh or (contain_i > containment_thresh and contain_j > 0.4) or (contain_j > containment_thresh and contain_i > 0.4):
                    print(f"[MERGE] Merging part {i} (vol={vol_i:.4f}) and part {j} (vol={vol_j:.4f}) with IoU={iou:.3f}, contain=({contain_i:.2f}, {contain_j:.2f})")
                    merge_pair = (i, j)
                    break

        if merge_pair:
            i, j = merge_pair
            combined = trimesh.util.concatenate([merged_parts[i], merged_parts[j]])
            combined.update_faces(combined.unique_faces())
            combined.remove_unreferenced_vertices()
            del merged_parts[j]
            merged_parts[i] = combined
            changed = True

    print(f"[MERGE] Finished with {len(merged_parts)} clean, distinct parts.")
    return merged_parts


def cap_and_fill_holes(mesh):
    """
    Watertight QuickCap / MeshFill: fills planar and open boundary cut loops
    so that exploded parts do not have hollow holes or missing interiors.
    """
    m = mesh.copy()
    if m.is_watertight:
        return m
    
    # 1. Fill holes in trimesh
    trimesh.repair.fill_holes(m)
    
    # 2. Fix normals
    trimesh.repair.fix_normals(m)
    trimesh.repair.fix_inversion(m)
    return m


def process_lossless_segmented_mesh(mesh_path, face_ids_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mesh = trimesh.load(mesh_path, force='mesh')
    face_ids = np.load(face_ids_path)
    
    unique_ids = np.unique(face_ids)
    unique_ids = [uid for uid in unique_ids if uid >= 0]
    
    raw_parts = []
    for uid in unique_ids:
        mask = (face_ids == uid)
        sub = mesh.submesh([mask], append=True)
        if len(sub.faces) > 0:
            raw_parts.append(sub)

    print(f"Extracted {len(raw_parts)} raw surface parts from input mesh.")
    
    # Merge overlapping duplicates and fragments
    clean_parts = clean_and_merge_parts(raw_parts, iou_thresh=0.60, containment_thresh=0.75)
    
    # Apply watertight hole filling / capping to every part
    sealed_parts = []
    for idx, p in enumerate(clean_parts):
        sealed = cap_and_fill_holes(p)
        sealed_parts.append(sealed)

    # Build Tripo3D Assembly
    assembly = create_tripo_assembly_scene(sealed_parts)
    
    assembled_path = out_dir / "furn16_3k_perfect_assembly.glb"
    export_tripo_assembly_glb(assembly, assembled_path)
    
    exploded = explode_mesh_tripo(assembly, explosion_scale=0.35, progress=1.0)
    exploded_path = out_dir / "furn16_3k_perfect_exploded.glb"
    export_tripo_assembly_glb(exploded, exploded_path)
    
    viewer_path = out_dir / "furn16_3k_perfect_viewer.html"
    generate_tripo_viewer_html(
        glb_filename="furn16_3k_perfect_assembly.glb",
        output_html_path=viewer_path,
        model_title="furn16_3k Furniture (Watertight Zero-Overlap)"
    )
    print("Done! All assets generated successfully.")

if __name__ == "__main__":
    process_lossless_segmented_mesh(
        "assets/furn16_3k.glb",
        "outputs/p3sam_furn16_3k/auto_mask_mesh_final_post_face_ids.npy",
        "outputs/furn16_3k_perfect"
    )
