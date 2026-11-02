import os
import sys
import shutil
import trimesh
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import render_utils

def add_header(img, title, subtitle="", height=70):
    """Adds a clean dark header with title and subtitle to an image."""
    w, h = img.size
    new_img = Image.new("RGB", (w, h + height), color=(20, 22, 28))
    new_img.paste(img, (0, height))
    draw = ImageDraw.Draw(new_img)
    
    # Try to load a nice font, fallback to default
    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_sub = ImageFont.truetype("arial.ttf", 15)
    except:
        font_title = ImageFont.load_default()
        font_sub = font_title
        
    draw.text((w // 2, 22), title, fill=(240, 242, 248), font=font_title, anchor="mm")
    if subtitle:
        draw.text((w // 2, 48), subtitle, fill=(160, 168, 185), font=font_sub, anchor="mm")
    return new_img

def main():
    bench_dir = "outputs/benchmark_multimask"
    artifact_dir = r"C:\Users\7950x_RTX4090\.gemini\antigravity-ide\brain\3a2de535-94b8-414e-8553-89ab2351766d"
    os.makedirs(artifact_dir, exist_ok=True)
    
    print("Generating visual comparison figures...")
    
    # Key runs to compare:
    # 1. Original @ 0.9 (Baseline)
    # 2. Improved (3-Masks) @ 0.9
    # 3. Improved + Stability @ 0.9
    path_orig = os.path.join(bench_dir, "Original_1-Mask_NMS_0.9", "segmented_mesh.glb")
    path_3mask = os.path.join(bench_dir, "Improved_3-Masks_NMS_0.9", "segmented_mesh.glb")
    path_stab = os.path.join(bench_dir, "Improved_plus_Stability_NMS_0.9", "segmented_mesh.glb")
    
    print("Loading meshes...")
    m_orig = trimesh.load(path_orig, force="mesh")
    m_3mask = trimesh.load(path_3mask, force="mesh")
    m_stab = trimesh.load(path_stab, force="mesh")
    
    # Render 2 angles for each: perspective (elev=22, azim=40)
    print("Rendering perspective views...")
    img_orig = render_utils.render_mesh_to_image(m_orig, width=640, height=640, elev=22, azim=40)
    img_3mask = render_utils.render_mesh_to_image(m_3mask, width=640, height=640, elev=22, azim=40)
    img_stab = render_utils.render_mesh_to_image(m_stab, width=640, height=640, elev=22, azim=40)
    
    img_orig = add_header(img_orig, "Original Baseline (1-Mask)", "400 Candidates | 23 Parts | mIoU: 0.6594")
    img_3mask = add_header(img_3mask, "Improved (All 3-Masks)", "1200 Candidates | 23 Parts | mIoU: 0.6402")
    img_stab = add_header(img_stab, "Improved + Stability Filter", "877 Candidates | 23 Parts | mIoU: 0.6117")
    
    # Combine side-by-side into a 3-column banner
    w, h = img_orig.size
    triptych = Image.new("RGB", (w * 3 + 20, h + 80), color=(14, 16, 20))
    draw_trip = ImageDraw.Draw(triptych)
    
    try:
        banner_font = ImageFont.truetype("arialbd.ttf", 26)
        banner_sub = ImageFont.truetype("arial.ttf", 16)
    except:
        banner_font = ImageFont.load_default()
        banner_sub = banner_font
        
    draw_trip.text(((w * 3 + 20) // 2, 28), "P3-SAM Automatic Segmentation: Pipeline Mode Comparison", fill=(255, 255, 255), font=banner_font, anchor="mm")
    draw_trip.text(((w * 3 + 20) // 2, 56), "Wooden Rack Mesh (854,828 Faces) | Fixed NMS Threshold = 0.9", fill=(170, 180, 200), font=banner_sub, anchor="mm")
    
    triptych.paste(img_orig, (0, 80))
    triptych.paste(img_3mask, (w + 10, 80))
    triptych.paste(img_stab, (w * 2 + 20, 80))
    
    triptych_save_path = os.path.join(bench_dir, "p3sam_multimask_before_after.png")
    triptych.save(triptych_save_path, quality=95)
    print(f"Saved {triptych_save_path}")
    
    # Copy to artifact directory
    artifact_triptych = os.path.join(artifact_dir, "p3sam_multimask_before_after.png")
    shutil.copy2(triptych_save_path, artifact_triptych)
    print(f"Copied to artifact directory: {artifact_triptych}")
    
    # Also render NMS Threshold Sweep (0.3 vs 0.5 vs 0.7 vs 0.9) for Improved (3-Masks)
    print("Rendering NMS threshold sweep comparison...")
    nms_paths = [
        ("NMS = 0.3", os.path.join(bench_dir, "Improved_3-Masks_NMS_0.3", "segmented_mesh.glb"), "32 Parts | Over-segmented slats | mIoU: 0.4281"),
        ("NMS = 0.5", os.path.join(bench_dir, "Improved_3-Masks_NMS_0.5", "segmented_mesh.glb"), "29 Parts | Fragmented rails | mIoU: 0.4738"),
        ("NMS = 0.7", os.path.join(bench_dir, "Improved_3-Masks_NMS_0.7", "segmented_mesh.glb"), "26 Parts | Moderate grouping | mIoU: 0.5364"),
        ("NMS = 0.9", os.path.join(bench_dir, "Improved_3-Masks_NMS_0.9", "segmented_mesh.glb"), "23 Parts | Optimal cohesion | mIoU: 0.6402"),
    ]
    
    nms_imgs = []
    for title, p, sub in nms_paths:
        m = trimesh.load(p, force="mesh")
        im = render_utils.render_mesh_to_image(m, width=480, height=480, elev=22, azim=40)
        im = add_header(im, title, sub, height=65)
        nms_imgs.append(im)
        
    w4, h4 = nms_imgs[0].size
    quad_banner = Image.new("RGB", (w4 * 4 + 30, h4 + 80), color=(14, 16, 20))
    draw_quad = ImageDraw.Draw(quad_banner)
    draw_quad.text(((w4 * 4 + 30) // 2, 28), "Improved 3-Mask Pipeline: Impact of NMS Threshold on Segmentation Granularity", fill=(255, 255, 255), font=banner_font, anchor="mm")
    draw_quad.text(((w4 * 4 + 30) // 2, 56), "Lower thresholds merge coarse bounding envelopes; higher thresholds retain fine subparts", fill=(170, 180, 200), font=banner_sub, anchor="mm")
    
    for idx, im in enumerate(nms_imgs):
        quad_banner.paste(im, (idx * (w4 + 10), 80))
        
    quad_save_path = os.path.join(bench_dir, "p3sam_nms_threshold_sweep.png")
    quad_banner.save(quad_save_path, quality=95)
    print(f"Saved {quad_save_path}")
    
    artifact_quad = os.path.join(artifact_dir, "p3sam_nms_threshold_sweep.png")
    shutil.copy2(quad_save_path, artifact_quad)
    print(f"Copied to artifact directory: {artifact_quad}")
    
    print("Visual comparison generation completed successfully!")

if __name__ == "__main__":
    main()
