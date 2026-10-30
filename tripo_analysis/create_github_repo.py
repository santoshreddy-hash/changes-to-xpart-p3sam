import subprocess
import urllib.request
import json

# Retrieve token from git credential
proc = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
out, _ = proc.communicate("protocol=https\nhost=github.com\nusername=santoshreddy-hash\n\n")

token = None
for line in out.splitlines():
    if line.startswith("password="):
        token = line.split("=", 1)[1].strip()

if not token:
    print("Could not retrieve token")
    exit(1)

print(f"Retrieved token for santoshreddy-hash (starts with {token[:4]}...)")

repo_name = "changes-to-xpart-p3sam"
payload = {
    "name": repo_name,
    "description": "Changes to XPart and P3-SAM with optimized marching cubes, batch decoding, and Tripo3D-compatible multi-node exploded GLB pipelines.",
    "private": False
}

req = urllib.request.Request(
    "https://api.github.com/user/repos",
    data=json.dumps(payload).encode('utf-8'),
    headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "antigravity-agent",
        "Content-Type": "application/json"
    },
    method="POST"
)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode())
        print("Successfully created repository!")
        print("Clone URL:", res.get("clone_url"))
        print("HTML URL:", res.get("html_url"))
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f"HTTPError {e.code}: {err}")
