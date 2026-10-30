import json

d = json.load(open('tripo_analysis/gltf_structure.json'))
for i in range(10):
    node = d['nodes'][i]
    mesh = d['meshes'][node['mesh']]
    pos_accessor_idx = mesh['primitives'][0]['attributes']['POSITION']
    pos_acc = d['accessors'][pos_accessor_idx]
    trans = node.get('translation')
    p_min = pos_acc.get('min')
    p_max = pos_acc.get('max')
    print(f"Part {i} ({node.get('name')}):")
    print(f"   Node translation: {trans}")
    print(f"   Position min: {p_min}")
    print(f"   Position max: {p_max}")
    if p_min and p_max:
        c = [(a + b) / 2 for a, b in zip(p_min, p_max)]
        print(f"   Local BBox center: {c}")
        if trans:
            world_c = [c[k] + trans[k] for k in range(3)]
            print(f"   World BBox center: {world_c}")
