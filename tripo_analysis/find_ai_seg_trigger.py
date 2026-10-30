import os
import re

scripts = os.listdir('tripo_analysis/scripts')
results = []

for s in scripts:
    path = os.path.join('tripo_analysis/scripts', s)
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    if 'ai_segmentation' in txt:
        matches = [m.start() for m in re.finditer('ai_segmentation', txt)]
        results.append((s, len(matches)))
        print(f"\nFound {len(matches)} matches in {s}:")
        for idx in matches[:3]:
            start = max(0, idx - 250)
            end = min(len(txt), idx + 250)
            print("--- Context:")
            print(txt[start:end])
            print("="*50)

print("Summary of files with ai_segmentation:", results)
