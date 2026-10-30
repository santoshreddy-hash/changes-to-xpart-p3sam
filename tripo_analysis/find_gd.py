import re

txt = open('tripo_analysis/scripts/C32b7X3t.js', 'r', encoding='utf-8', errors='ignore').read()
matches = [m.start() for m in re.finditer(r'\bGd\b', txt)]
print(f"Found {len(matches)} matches for Gd:")
for idx in matches:
    start = max(0, idx - 200)
    end = min(len(txt), idx + 200)
    print("--- Match at", idx)
    print(txt[start:end])
