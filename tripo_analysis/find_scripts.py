import re

html = open('content.md', 'r', encoding='utf-8', errors='ignore').read()
urls = re.findall(r'(https://staticassets\.tripo3d\.ai/[^\s"\'<>]+)', html)
urls += re.findall(r'(/_nuxt/[^\s"\'<>]+)', html)
urls = sorted(list(set(urls)))
print(f"Total asset URLs found: {len(urls)}")
for u in urls:
    if any(ext in u for ext in ['.js', '.mjs', 'worker', 'segmentation', 'workspace']):
        print(u)
