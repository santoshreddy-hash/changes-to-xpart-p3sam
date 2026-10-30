import trimesh
import PIL.Image
import numpy as np

print("=== INSPECTING tripo_mesh.glb ===")
scene = trimesh.load("tripo_analysis/tripo_mesh.glb")
print("Type:", type(scene))
if isinstance(scene, trimesh.Scene):
    print("Geometry keys:", list(scene.geometry.keys()))
    print("Number of geometries:", len(scene.geometry))
    for k, geom in list(scene.geometry.items())[:15]:
        print(f"Geom {k}: vertices={geom.vertices.shape}, faces={geom.faces.shape}")
        if hasattr(geom, "visual"):
            if hasattr(geom.visual, "material"):
                mat = geom.visual.material
                mat_name = getattr(mat, "name", type(mat).__name__)
                print(f"   Material: {mat_name}")
            if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
                print(f"   UV shape: {geom.visual.uv.shape}")
            if hasattr(geom.visual, "vertex_colors") and geom.visual.vertex_colors is not None:
                print(f"   Vertex colors shape: {geom.visual.vertex_colors.shape}")
    print("Scene graph nodes:", len(scene.graph.nodes))
    for node in list(scene.graph.nodes)[:15]:
        xform, geom_name = scene.graph.get(node)
        print(f"Node: {node}, geom={geom_name}")

print("\n=== INSPECTING segmentation_map.png ===")
img = PIL.Image.open("tripo_analysis/segmentation_map.png")
print("Image mode:", img.mode, "Size:", img.size)
arr = np.array(img)
print("Array shape:", arr.shape, "dtype:", arr.dtype)
if arr.ndim == 3:
    flat = arr.reshape(-1, arr.shape[-1])
    unique_colors = np.unique(flat, axis=0)
    print("Number of unique colors:", len(unique_colors))
    print("First 15 unique colors:\n", unique_colors[:15])
else:
    unique_vals = np.unique(arr)
    print("Number of unique values:", len(unique_vals))
    print("Unique values:", unique_vals[:20])

print("\n=== INSPECTING point_cloud.glb ===")
pc_scene = trimesh.load("tripo_analysis/point_cloud.glb")
print("PC Type:", type(pc_scene))
if isinstance(pc_scene, trimesh.Scene):
    print("PC Geometry keys:", list(pc_scene.geometry.keys()))
    for k, geom in pc_scene.geometry.items():
        verts = getattr(geom, "vertices", None)
        print(f"PC Geom {k}: vertices shape = {verts.shape if verts is not None else None}")
        if hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
            print(f"   Vertex colors: {geom.visual.vertex_colors.shape}")
