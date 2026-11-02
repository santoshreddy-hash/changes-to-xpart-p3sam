import os
import trimesh

mesh_candidates = [
    ("Wooden Sideboard", "uploads/furn16_woodenrack_raw_854k_d95a8fe0.glb"),
    ("Vintage Table", "uploads/bedside_wood_table_vintage_furniture_7aa2f793.glb"),
    ("Jade Sword", "uploads/jade_sword_650f414d.glb"),
    ("Deer 100k", "uploads/deer_100k_1ae4cfca.glb"),
    ("Stylized Character", "uploads/fall_guys_trice_skin_07d92877.glb"),
    ("Vase Prop", "uploads/vase_8437fc2d.glb"),
    ("Creature Model", "All_Input_GLBs (1)/All_Input_GLBs/man_creature_input.glb"),
    ("Table & Chairs Assembly", "uploads/vihals-vihals-table-and-4-chairs-white-white-tibbleby-beige-grey__1388772_pe964945_s5_10k_2k_59ec9ff8.glb")
]

for name, path in mesh_candidates:
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue
    m = trimesh.load(path, force='mesh')
    print(f"{name:25s} | Faces: {len(m.faces):7d} | Verts: {len(m.vertices):7d} | Bounds: {np.round(m.extents, 2) if 'np' in locals() else m.extents}")
