import json, struct
with open("tripo_analysis/test_assembly.glb", "rb") as f:
    f.read(12)
    chunk0_len, _ = struct.unpack('<II', f.read(8))
    gltf_json = json.loads(f.read(chunk0_len).decode('utf-8'))

for idx, node in enumerate(gltf_json.get('nodes', [])):
    print(f"Node {idx}: name={node.get('name')}, translation={node.get('translation')}, matrix={node.get('matrix')}")
