content = open('tripo_analysis/scripts/CRvlez9U.js', 'r', encoding='utf-8').read()

# Simple token based beautifier
out = []
indent = 0
for ch in content:
    if ch == '{':
        indent += 2
        out.append('{\n' + ' ' * indent)
    elif ch == '}':
        indent = max(0, indent - 2)
        out.append('\n' + ' ' * indent + '}\n' + ' ' * indent)
    elif ch == ';':
        out.append(';\n' + ' ' * indent)
    else:
        out.append(ch)

beautified = "".join(out)
with open('tripo_analysis/CRvlez9U_beautified.js', 'w', encoding='utf-8') as f:
    f.write(beautified)

print("Saved beautified script. Length:", len(beautified))
