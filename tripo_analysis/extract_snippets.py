import os
import re

def extract_contexts(fname, terms, radius=400):
    path = os.path.join('tripo_analysis/scripts', fname)
    if not os.path.exists(path):
        print(f"File {fname} not found")
        return
    content = open(path, 'r', encoding='utf-8', errors='ignore').read()
    print(f"=== {fname} (Length: {len(content)}) ===")
    for term in terms:
        print(f"\n--- Searching for: '{term}' in {fname} ---")
        matches = [m.start() for m in re.finditer(re.escape(term), content, re.IGNORECASE)]
        for i, idx in enumerate(matches[:10]):
            start = max(0, idx - radius)
            end = min(len(content), idx + radius)
            print(f"Match {i+1} at index {idx}:")
            print(content[start:end])
            print("="*60)

extract_contexts('C32b7X3t.js', ['explode', 'explosion', 'tripo_part'])
extract_contexts('CRvlez9U.js', ['explode'])
