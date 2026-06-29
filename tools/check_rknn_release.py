"""Check RKNN-Toolkit2 release assets"""
import json
import urllib.request
import sys

repos = [
    "rockchip-linux/rknn-toolkit2",
    "airockchip/rknn-toolkit2",
]

for repo in repos:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "RK-MRT2")
        with urllib.request.urlopen(req) as f:
            data = json.load(f)
        tag = data.get("tag_name", "unknown")
        print(f"\n{repo}: {tag}")
        assets = data.get("assets", [])
        if not assets:
            print("  WARNING: No assets found (may require authentication)")
            print(f"  Release URL: https://github.com/{repo}/releases/latest")
        for a in assets:
            size_mb = a["size"] / (1024 * 1024)
            print(f"  {a['name']} ({size_mb:.0f} MB)")
    except Exception as e:
        print(f"\n{repo}: ERROR - {e}")
