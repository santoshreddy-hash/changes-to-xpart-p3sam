import json, struct

def convert_matrix_to_trs(glb_bytes):
    magic, version, length = struct.unpack('<4sII', glb_bytes[:12])
    chunk0_len, chunk0_type = struct.unpack('<II', glb_bytes[12:20])
    chunk0_data = glb_bytes[20:20+chunk0_len]
    chunk1_data = glb_bytes[20+chunk0_len:]
    
    gltf = json.loads(chunk0_data.decode('utf-8'))
    for node in gltf.get('nodes', []):
        if 'matrix' in node and node['matrix'] is not None:
            m = node['matrix']
            # If identity rotation and scale
            if (m[0], m[5], m[10], m[15]) == (1.0, 1.0, 1.0, 1.0) and \
               (m[1], m[2], m[3], m[4], m[6], m[7], m[8], m[9], m[11]) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
                node['translation'] = [round(m[12], 8), round(m[13], 8), round(m[14], 8)]
                del node['matrix']
    
    # Re-encode JSON
    new_json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    # Pad to 4-byte boundary with spaces (0x20)
    pad = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * pad
    
    new_total_len = 12 + 8 + len(new_json_bytes) + len(chunk1_data)
    header = struct.pack('<4sII', magic, version, new_total_len)
    chunk0_header = struct.pack('<II', len(new_json_bytes), chunk0_type)
    return header + chunk0_header + new_json_bytes + chunk1_data

with open("tripo_analysis/test_assembly.glb", "rb") as f:
    orig = f.read()

converted = convert_matrix_to_trs(orig)
with open("tripo_analysis/test_assembly_trs.glb", "wb") as f:
    f.write(converted)

with open("tripo_analysis/test_assembly_trs.glb", "rb") as f:
    f.read(12)
    chunk0_len, _ = struct.unpack('<II', f.read(8))
    gltf = json.loads(f.read(chunk0_len).decode('utf-8'))

for i, n in enumerate(gltf['nodes']):
    print(f"Node {i}: name={n.get('name')}, translation={n.get('translation')}, matrix={n.get('matrix')}")
