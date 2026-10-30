import urllib.request
import os
import re

os.makedirs('tripo_analysis/scripts', exist_ok=True)
html = open('tripo_analysis/page.html', 'r', encoding='utf-8', errors='ignore').read()

scripts = re.findall(r'src=[\'"]([^\'"]+\.js)[\'"]', html)
scripts += re.findall(r'href=[\'"]([^\'"]+\.js)[\'"]', html)
scripts = sorted(list(set(scripts)))

keywords = ['explode', 'explosion', 'segmentation', 'tripo_part', 'segment']

matching_files = {}

for s in scripts:
    fname = s.split('/')[-1]
    local_path = os.path.join('tripo_analysis/scripts', fname)
    if not os.path.exists(local_path):
        try:
            req = urllib.request.Request(s, headers={'User-Agent': 'Mozilla/5.0'})
            content = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            continue
    else:
        with open(local_path, 'r', encoding='utf-8') as f:
            content = f.read()

    # Search for keywords
    found_kw = []
    for kw in keywords:
        matches = [m.start() for m in re.finditer(kw, content, re.IGNORECASE)]
        if matches:
            found_kw.append((kw, len(matches)))
    if found_kw:
        matching_files[fname] = found_kw

print(f"Searched {len(scripts)} scripts. Files with matches:")
for fname, kw_list in matching_files.items():
    print(f"  {fname}: {kw_list}")
