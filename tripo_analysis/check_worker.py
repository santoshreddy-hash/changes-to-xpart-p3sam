import re

txt = open('tripo_analysis/scripts/model-picking-worker-DwbDbumI.js', 'r', encoding='utf-8', errors='ignore').read()
matches = [m.start() for m in re.finditer(r'onmessage', txt)]
print("Matches for onmessage:", len(matches))
for idx in matches:
    print(txt[max(0, idx - 200):min(len(txt), idx + 400)])
    print("="*50)

print("\nLast 2000 chars:")
print(txt[-2000:])
