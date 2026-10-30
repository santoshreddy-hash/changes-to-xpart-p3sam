import trimesh
import numpy as np

# Create two sample box parts
box1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
box2 = trimesh.creation.box(extents=[0.5, 0.5, 0.5])

# In world space, box1 is at [0, 0, 0], box2 is at [2, 0, 0]
box1_world = box1.copy()
box2_world = box2.copy()
box2_world.apply_translation([2.0, 0.0, 0.0])

# Assembly logic:
parts = [box1_world, box2_world]
scene = trimesh.Scene()

for i, p in enumerate(parts):
    c = p.bounds.mean(axis=0)
    p_local = p.copy()
    p_local.vertices -= c
    
    transform = np.eye(4)
    transform[:3, 3] = c
    node_name = f"tripo_part_{i}"
    scene.add_geometry(p_local, geom_name=node_name, node_name=node_name, transform=transform)

scene.export("tripo_analysis/test_assembly.glb")
print("Exported test_assembly.glb successfully")

# Reload and check structure
import json, struct
with open("tripo_analysis/test_assembly.glb", "rb") as f:
    f.read(12)
    chunk0_len, _ = struct.unpack('<II', f.read(8))
    gltf_json = json.loads(f.read(chunk0_len).decode('utf-8'))

print("Nodes in exported GLB:")
for idx, node in enumerate(gltf_json.get('nodes', [])):
    print(f"Node {idx}: name={node.get('name')}, translation={node.get('translation')}, mesh={node.get('mesh')}")
