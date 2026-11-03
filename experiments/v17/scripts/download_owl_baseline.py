"""Optional official OWL-ViT checkpoint fetch for the published replay bundle."""
from pathlib import Path
import argparse, hashlib, json, urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='google/owlvit-base-patch32'
REVISION='cbc355fb364588351c5d51c7f74465e8e7ec6f72'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main(args):
    manifest=json.loads((ROOT/'published_benchmark.json').read_text())
    entry=manifest['methods']['owlvit_base_patch32']
    record_path=ROOT/'data/owlvit/acquisition.json'
    if manifest.get('completed') is not True or sha(record_path)!=entry['owl_checkpoint_files']['acquisition.json']:
        raise ValueError('Completed public manifest and exact acquisition record required')
    record=json.loads(record_path.read_text())
    if record['repo_id']!=REPO or record['revision']!=REVISION:
        raise ValueError('Unexpected model/revision')
    if set(record['files'])!={'config.json','preprocessor_config.json','model.safetensors'}:
        raise ValueError('Unexpected checkpoint files')
    destination=Path(args.destination).resolve(); destination.mkdir(parents=True,exist_ok=True)
    for name,info in record['files'].items():
        url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        if info['url']!=url or info['sha256']!=entry['owl_checkpoint_files'][name]:
            raise ValueError('Pinned URL/hash differs')
        target=destination/name
        if target.exists():
            if sha(target)!=info['sha256']: raise ValueError('Existing checkpoint file differs: '+str(target))
            continue
        partial=destination/(name+'.partial')
        with urllib.request.urlopen(url,timeout=120) as response, partial.open('xb') as f:
            for block in iter(lambda:response.read(1024*1024),b''): f.write(block)
        if partial.stat().st_size!=info['bytes'] or sha(partial)!=info['sha256']:
            raise ValueError('Download failed verification; partial retained for inspection')
        if target.exists(): raise FileExistsError('Target appeared while downloading')
        partial.rename(target)
        print(json.dumps({'verified':name,'bytes':info['bytes']}),flush=True)
    target=destination/'acquisition.json'
    if target.exists():
        if sha(target)!=sha(record_path): raise ValueError('Existing acquisition record differs')
    else:
        with target.open('xb') as f: f.write(record_path.read_bytes())
    print(json.dumps({'checkpoint_ready':str(destination),'repo':REPO,'revision':REVISION}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True,type=Path)
    main(parser.parse_args())
