import os
import sys
import argparse
from pathlib import Path
import trimesh
import numpy as np

# Ensure XPart is on path
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR / "XPart"))
from partgen.utils.mesh_utils import (
    create_tripo_assembly_scene,
    export_tripo_assembly_glb,
    explode_mesh_tripo,
    generate_tripo_viewer_html,
    TRIPO_PALETTE
)

def convert_mesh_to_tripo(input_path, output_dir=None, explosion_scale=0.35):
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_tripo"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONVERT] Loading mesh from {input_path}...")
    loaded = trimesh.load(str(input_path))

    part_meshes = []
    if isinstance(loaded, trimesh.Scene):
        print(f"[CONVERT] Input is a Scene with {len(loaded.geometry)} geometries")
        for name, geom in loaded.geometry.items():
            if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                transform = loaded.graph[name][0]
                world_geom = geom.copy()
                world_geom.apply_transform(transform)
                part_meshes.append(world_geom)
    elif isinstance(loaded, trimesh.Trimesh):
        print(f"[CONVERT] Input is a single Trimesh. Checking connected components...")
        components = loaded.split(only_watertight=False)
        if len(components) > 1:
            print(f"[CONVERT] Split into {len(components)} connected components")
            part_meshes = components
        else:
            print("[CONVERT] Single continuous component, keeping as 1 part")
            part_meshes = [loaded]
    else:
        raise ValueError(f"Unsupported geometry type: {type(loaded)}")

    print(f"[CONVERT] Building Tripo3D multi-node assembly for {len(part_meshes)} parts...")
    assembly_scene = create_tripo_assembly_scene(part_meshes)

    # 1. Export Assembled GLB
    assembled_glb_name = f"{input_path.stem}_assembly.glb"
    assembled_glb_path = output_dir / assembled_glb_name
    export_tripo_assembly_glb(assembly_scene, assembled_glb_path)

    # 2. Export Exploded GLB (Tripo3D exact formulas)
    exploded_scene = explode_mesh_tripo(assembly_scene, explosion_scale=explosion_scale, progress=1.0)
    exploded_glb_name = f"{input_path.stem}_exploded.glb"
    exploded_glb_path = output_dir / exploded_glb_name
    export_tripo_assembly_glb(exploded_scene, exploded_glb_path)

    # 3. Generate Interactive Three.js HTML Viewer
    viewer_html_name = f"{input_path.stem}_viewer.html"
    viewer_html_path = output_dir / viewer_html_name
    generate_tripo_viewer_html(
        glb_filename=assembled_glb_name,
        output_html_path=viewer_html_path,
        model_title=f"{input_path.stem} (Tripo3D Explosion Engine)"
    )

    print("\n================ CONVERSION COMPLETE ================")
    print(f"  Assembled Multi-Node GLB : {assembled_glb_path}")
    print(f"  Exploded GLB (Exact Wave): {exploded_glb_path}")
    print(f"  Interactive 3D Viewer    : {viewer_html_path}")
    print("=====================================================\n")
    return assembled_glb_path, exploded_glb_path, viewer_html_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert any segmented 3D mesh into Tripo3D-compatible multi-node assembly and interactive exploded viewer.")
    parser.add_argument("input_path", type=str, help="Path to input GLB / OBJ mesh")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--explosion_scale", type=float, default=0.35, help="Explosion displacement scale")
    args = parser.parse_args()

    convert_mesh_to_tripo(args.input_path, args.out_dir, args.explosion_scale)
