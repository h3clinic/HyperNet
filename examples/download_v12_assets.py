"""Fetch this project's exact synthetic-only CNN and procedural fresh RGB cohort."""
from pathlib import Path
import argparse,hashlib,json,shutil,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verified(path,record):return path.is_file() and path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
def extract_verified(zip_path,data):
    with zipfile.ZipFile(zip_path) as z:
        manifest=json.loads(z.read('manifest.json'))
        expected={r['rgb_path']:r['rgb_sha256'] for e in manifest['episodes'] for r in [e['support']]+e['queries']}
        if set(z.namelist())!=set(expected)|{'manifest.json'}:raise ValueError('Unexpected archive members')
        resolved=data.resolve()
        for name in z.namelist():
            target=(data/name).resolve()
            if not target.is_relative_to(resolved) or target==resolved:raise ValueError('Unsafe archive path')
            content=z.read(name)
            if name in expected and hashlib.sha256(content).hexdigest()!=expected[name]:raise ValueError('RGB archive hash differs')
            if target.exists() and target.read_bytes()!=content:raise FileExistsError('Refusing to overwrite different local data')
        for name in z.namelist():
            target=data/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(z.read(name))
    return len(expected)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'data/shapes2d_fresh_v12')
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/detector_micro_aware.pt')
    ap.add_argument('--from-directory',type=Path,help='Verify local release assets instead of downloading')
    ap.add_argument('--cache',type=Path,default=ROOT/'.asset_cache');args=ap.parse_args()
    records=json.loads((ROOT/'benchmarks/v0.12/release_assets.json').read_text());assets={}
    for name,record in records.items():
        path=(args.from_directory or args.cache)/name
        if args.from_directory:
            if not verified(path,record):raise ValueError('Local release asset hash mismatch')
        elif not verified(path,record):
            if path.exists():raise ValueError('Existing cached asset has wrong hash')
            path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(record['url']) as source,part.open('wb') as output:shutil.copyfileobj(source,output)
            if not verified(part,record):raise ValueError('Downloaded asset hash mismatch')
            part.replace(path)
        assets[name]=path
    checkpoint=args.checkpoint
    if checkpoint.exists() and sha(checkpoint)!=records['detector_micro_aware.pt']['sha256']:raise FileExistsError('Different checkpoint exists')
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if not checkpoint.exists():shutil.copyfile(assets['detector_micro_aware.pt'],checkpoint)
    count=extract_verified(assets['fresh_v12_rgb.zip'],args.data)
    print(json.dumps({'verified':True,'rgb_images':count,'checkpoint_sha256':sha(checkpoint),'data':str(args.data),'checkpoint':str(checkpoint)},indent=2))
if __name__=='__main__':main()
