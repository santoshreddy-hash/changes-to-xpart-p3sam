import urllib.request
import re
import json

url = 'https://studio.tripo3d.ai/workspace/segmentation/d7372434-9771-444b-ba56-fa193fdc0e95'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')

with open('tripo_analysis/page.html', 'w', encoding='utf-8') as f:
    f.write(html)

# Extract script src and modulepreload href
scripts = re.findall(r'src=[\'"]([^\'"]+\.js)[\'"]', html)
scripts += re.findall(r'href=[\'"]([^\'"]+\.js)[\'"]', html)
scripts = sorted(list(set(scripts)))

print(f"Found {len(scripts)} scripts:")
for s in scripts:
    print(" ", s)
