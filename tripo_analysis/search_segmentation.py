import re

content = open('tripo_analysis/scripts/C32b7X3t.js', 'r', encoding='utf-8', errors='ignore').read()

terms = ['ai_segmentation', 'segmentation_granularity', 'segmentation_map', 'point_cloud_model_url', 'worker', 'picker', 'brush', 'quick_cap']
radius = 400

for term in terms:
    print(f"\n================ SEARCHING: {term} ================")
    matches = [m.start() for m in re.finditer(re.escape(term), content, re.IGNORECASE)]
    print(f"Found {len(matches)} matches")
    for i, idx in enumerate(matches[:5]):
        start = max(0, idx - radius)
        end = min(len(content), idx + radius)
        print(f"--- Match {i+1} at index {idx}:")
        print(content[start:end])
