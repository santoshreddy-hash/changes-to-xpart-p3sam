import os
import re

scripts = os.listdir('tripo_analysis/scripts')
for s in scripts:
    path = os.path.join('tripo_analysis/scripts', s)
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    if 'point_cloud_model_url' in txt:
        print(f"File {s} has point_cloud_model_url")
        for m in re.finditer('point_cloud_model_url', txt):
            idx = m.start()
            print(txt[max(0, idx-200):min(len(txt), idx+200)])
            print("="*50)
