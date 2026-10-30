import struct
import json

with open('tripo_analysis/tripo_mesh.glb', 'rb') as f:
    magic, version, length = struct.unpack('<4sII', f.read(12))
    print(f"Magic: {magic}, Version: {version}, Total Length: {length}")
    chunk0_len, chunk0_type = struct.unpack('<II', f.read(8))
    print(f"Chunk 0 len: {chunk0_len}, type: {chunk0_type}")
    chunk0_data = f.read(chunk0_len)
    gltf_json = json.loads(chunk0_data.decode('utf-8'))

with open('tripo_analysis/gltf_structure.json', 'w') as out_f:
    json.dump(gltf_json, out_f, indent=2)

print("Saved GLTF JSON structure to tripo_analysis/gltf_structure.json")
print("GLTF JSON top-level keys:", list(gltf_json.keys()))
if 'extensionsUsed' in gltf_json:
    print("Extensions used:", gltf_json['extensionsUsed'])
if 'meshes' in gltf_json:
    print(f"Number of meshes: {len(gltf_json['meshes'])}")
    for idx, m in enumerate(gltf_json['meshes'][:15]):
        print(f"Mesh {idx}: name={m.get('name')}, primitives={len(m.get('primitives', []))}")
        for p_idx, p in enumerate(m.get('primitives', [])):
            print(f"   Prim {p_idx}: attributes={list(p.get('attributes', {}).keys())}, material={p.get('material')}")
if 'nodes' in gltf_json:
    print(f"Number of nodes: {len(gltf_json['nodes'])}")
    for idx, n in enumerate(gltf_json['nodes'][:20]):
        print(f"Node {idx}: name={n.get('name')}, mesh={n.get('mesh')}, children={n.get('children')}, translation={n.get('translation')}, extras={n.get('extras')}")
if 'materials' in gltf_json:
    print(f"Number of materials: {len(gltf_json['materials'])}")
    for idx, mat in enumerate(gltf_json['materials'][:10]):
        print(f"Material {idx}: name={mat.get('name')}")
