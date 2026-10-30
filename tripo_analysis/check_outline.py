txt = open('tripo_analysis/scripts/outline-worker-CeAmN33n.js', 'r', encoding='utf-8', errors='ignore').read()
print("Outline worker length:", len(txt))
print("First 500 chars:")
print(txt[:500].encode('ascii', errors='replace').decode('ascii'))
