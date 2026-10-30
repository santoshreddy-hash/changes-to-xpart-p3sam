content = open('tripo_analysis/scripts/C32b7X3t.js', 'r', encoding='utf-8', errors='ignore').read()

start = 101000
end = 104500
with open('tripo_analysis/seg_logic_snippet.js', 'w', encoding='utf-8') as f:
    f.write(content[start:end])
print("Saved snippet")
