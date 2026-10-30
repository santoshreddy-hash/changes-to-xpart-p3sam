import PIL.Image
import numpy as np

img = PIL.Image.open('tripo_analysis/segmentation_map.png')
print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
arr = np.array(img)
print("Shape:", arr.shape)
print("Min:", arr.min(), "Max:", arr.max())

# Count unique colors
if arr.ndim == 3:
    flat = arr.reshape(-1, arr.shape[-1])
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    print(f"Total unique colors in segmentation_map.png: {len(colors)}")
    # sort by count descending
    sort_idx = np.argsort(-counts)
    print("Top 20 most frequent colors (RGB/RGBA, count):")
    for idx in sort_idx[:20]:
        print(f"  {colors[idx]} : {counts[idx]} pixels")
