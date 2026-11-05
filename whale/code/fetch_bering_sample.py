"""Download an evenly spaced sample of files from every Bering deployment (unlabeled pretraining corpus)."""
import json, urllib.request, os, sys, collections
API = "https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o"
B = "https://storage.googleapis.com/noaa-passive-bioacoustic/"
per = int(sys.argv[1]) if len(sys.argv) > 1 else 3
out = '../bering/raw'; tok = None; by = collections.defaultdict(list)
while True:
    d = json.load(urllib.request.urlopen(f"{API}?prefix=mbarc_bering/audio/&fields=items(name,size),nextPageToken&maxResults=1000" + (f"&pageToken={tok}" if tok else "")))
    for i in d.get('items', []):
        if i['name'].endswith('.flac'): by[i['name'].split('/')[2]].append(i)
    tok = d.get('nextPageToken')
    if not tok: break
json.dump({k: [i['name'] for i in v] for k, v in by.items()}, open('../bering/inventory.json', 'w'), indent=0)
for dep, items in sorted(by.items()):
    items.sort(key=lambda i: i['name'])
    pick = [items[int((j + .5) * len(items) / per)] for j in range(per)] if len(items) >= per else items
    print(dep, len(items), 'files', sum(int(i['size']) for i in items) / 1e9, 'GB', flush=True)
    for i in pick:
        dst = os.path.join(out, os.path.basename(i['name']))
        if not (os.path.exists(dst) and os.path.getsize(dst) == int(i['size'])):
            urllib.request.urlretrieve(B + i['name'], dst)
print('done')
