"""Portable pair manifests and deterministic same-source image stress tests.

The generated data measure source-image identity under controlled transforms.
They do NOT supply independently photographed physical object identities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw

CONDITIONS = ("clean", "rotation", "scale", "perspective", "lighting", "occlusion", "combined")
SCHEMA_VERSION = "hypernet-pairs-v1"


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha256(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def _resolve(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Image paths must be nonempty relative strings")
    p = Path(relative)
    # Also reject Windows drive syntax when a manifest is validated on Unix.
    if p.is_absolute() or ":" in relative or relative.startswith(("/", "\\")):
        raise ValueError(f"Absolute image path is not portable: {relative}")
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Image path escapes manifest folder: {relative}")
    if not resolved.is_file():
        raise ValueError(f"Missing image: {relative}")
    return resolved


def load_manifest(path, validate=True):
    """Return parsed metadata with an absolute runtime-only `root` string."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["root"] = str(path.parent)
    if validate:
        manifest["validation"] = validate_manifest(manifest, path.parent)
    return manifest


def validate_manifest(manifest, root=None):
    """Validate paths, labels, reference uniqueness, hashes and split separation.

    `source_id`/`query_source_id` must identify capture lineage when supplied.
    The validator cannot infer that two different photos show the same physical
    object. Dataset authors must assign those object IDs correctly.
    """
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Manifest needs a nonempty pairs list")
    root = Path(root or manifest.get("root", ".")).resolve()
    identities = defaultdict(set); lineage = defaultdict(set); image_splits = defaultdict(set)
    reference_by_object = {}; seen_ids=set(); hashes={}; pixel_hashes={}
    conditions=Counter(); split_counts=Counter(); labels=Counter()
    synthetic = manifest.get("protocol") == "synthetic_same_source"
    for row in pairs:
        required = {"id", "object_id", "reference", "query", "present", "condition", "split"}
        if not isinstance(row, dict) or required - row.keys():
            raise ValueError(f"Pair missing required fields: {required - row.keys() if isinstance(row,dict) else required}")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen_ids:
            raise ValueError("Pair IDs must be unique nonempty strings")
        seen_ids.add(row["id"])
        if type(row["present"]) is not bool:
            raise ValueError("present must be a JSON boolean, not a number or string")
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if not isinstance(row["object_id"], str) or not row["object_id"]:
            raise ValueError("object_id must be a nonempty string")
        if not isinstance(row["condition"], str) or not row["condition"]:
            raise ValueError("condition must be a nonempty string")
        split=row["split"]; oid=row["object_id"]
        identities[oid].add(split)
        qid=row.get("query_object_id")
        if qid is not None:
            if not isinstance(qid,str) or not qid:
                raise ValueError("query_object_id must be a nonempty string")
            identities[qid].add(split)
            if row["present"] != (oid==qid):
                raise ValueError("present disagrees with target and query object IDs")
        for field in ("source_id", "query_source_id"):
            if row.get(field) is not None:
                lineage[str(row[field])].add(split)
        paths={field:_resolve(root,row[field]) for field in ("reference","query")}
        for field,path in paths.items():
            if path not in hashes:
                hashes[path]=file_sha256(path)
                pixel_hashes[path]=pixel_sha256(load_rgb(path))
            declared=row.get(field+"_sha256")
            if declared is not None and declared != hashes[path]:
                raise ValueError(f"Hash mismatch: {row[field]}")
            image_splits[hashes[path]].add(split)
            image_splits[pixel_hashes[path]].add(split)
        ref_hash=hashes[paths["reference"]]
        if oid in reference_by_object and reference_by_object[oid] != ref_hash:
            raise ValueError(f"Object {oid} has more than one reference image")
        reference_by_object[oid]=ref_hash
        equal=hashes[paths["reference"]]==hashes[paths["query"]] or pixel_hashes[paths["reference"]]==pixel_hashes[paths["query"]]
        if equal and not (synthetic and row.get("derived_from_reference") is True and row["present"]):
            raise ValueError("Reference/query duplicates need explicit positive synthetic lineage")
        if row.get("derived_from_reference") is True and (not synthetic or not row["present"]):
            raise ValueError("derived_from_reference is only valid for positive synthetic pairs")
        if row.get("H_ref_to_query") is not None:
            h=np.asarray(row["H_ref_to_query"],dtype=float)
            if h.shape != (3,3) or not np.isfinite(h).all() or abs(np.linalg.det(h))<1e-12:
                raise ValueError("H_ref_to_query must be a finite nonsingular 3x3 matrix")
            if not row["present"]:
                raise ValueError("Target-absent pair cannot have a target homography")
        if row.get("target_bbox") is not None:
            box=np.asarray(row["target_bbox"],dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError("target_bbox must be finite ordered xyxy coordinates")
            if not row["present"]:
                raise ValueError("Target-absent pair cannot have a target bounding box")
        conditions[row["condition"]]+=1;split_counts[split]+=1;labels[str(row["present"]).lower()]+=1
    for name,groups in (("object identity",identities),("source lineage",lineage),("image content",image_splits)):
        leaked=[key for key,splits in groups.items() if len(splits)>1]
        if leaked:
            raise ValueError(f"Cross-split {name} leakage: {leaked[:3]}")
    return {"valid":True,"n_pairs":len(pairs),"n_target_objects":len(reference_by_object),
            "n_images":len(hashes),"splits":dict(split_counts),"labels":dict(labels),
            "conditions":dict(conditions),"synthetic_same_source":synthetic,
            "limitation":"File/lineage checks cannot establish physical-object identities in unannotated photographs."}


def _homography(src, dst):
    a=[];b=[]
    for (x,y),(u,v) in zip(src,dst):
        a.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        b.extend([u,v])
    return np.append(np.linalg.solve(np.asarray(a,float),np.asarray(b,float)),1).reshape(3,3)


def _transform(image, condition, rng):
    size=image.width;c=(size-1)/2
    h=np.eye(3);parameters={}
    if condition in {"rotation","combined"}:
        degrees=float(rng.choice([-40,-25,25,40]))
        angle=math.radians(degrees);co=math.cos(angle);si=math.sin(angle)
        rot=np.array([[co,-si,c-co*c+si*c],[si,co,c-si*c-co*c],[0,0,1.]])
        h=rot@h;parameters["rotation_degrees"]=degrees
    if condition in {"scale","combined"}:
        scale=float(rng.uniform(.58,.78))
        h=np.array([[scale,0,c*(1-scale)],[0,scale,c*(1-scale)],[0,0,1.]])@h
        parameters["scale"]=scale
    if condition in {"perspective","combined"}:
        corners=np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]],float)
        inset=size*.10
        target=corners.copy()
        target[0]+=[rng.uniform(0,inset*1.7),rng.uniform(0,inset)]
        target[1]+=[-rng.uniform(0,inset),rng.uniform(0,inset*1.8)]
        target[2]-=[rng.uniform(0,inset*1.8),rng.uniform(0,inset)]
        target[3]+=[rng.uniform(0,inset),-rng.uniform(0,inset*1.8)]
        h=_homography(corners,target)@h;parameters["perspective_corners"]=target.tolist()
    if not np.allclose(h,np.eye(3)):
        inverse=np.linalg.inv(h);inverse/=inverse[2,2]
        image=image.transform((size,size),Image.Transform.PERSPECTIVE,inverse.flatten()[:8],
                              resample=Image.Resampling.BICUBIC,fillcolor=(126,126,126))
    else:
        image=image.copy()
    if condition in {"lighting","combined"}:
        brightness=float(rng.choice([.48,.65,1.35]));contrast=float(rng.uniform(.65,1.3))
        image=ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(brightness)).enhance(contrast)
        parameters.update(brightness=brightness,contrast=contrast)
    if condition in {"occlusion","combined"}:
        side=int(size*(.50 if condition=="occlusion" else .35))
        x=int(rng.integers(int(size*.15),size-side+1));y=int(rng.integers(int(size*.15),size-side+1))
        fill=tuple(int(v) for v in rng.integers(70,180,size=3))
        ImageDraw.Draw(image).rectangle((x,y,x+side-1,y+side-1),fill=fill)
        parameters["occluder_bbox"]=[x,y,x+side,y+side]
        parameters["occluder_canvas_fraction"]=side*side/(size*size)
    return image,h,parameters


def _footprint_bbox(h,size):
    corners=np.array([[0,0,1],[size-1,0,1],[size-1,size-1,1],[0,size-1,1]],float)
    projected=(h@corners.T).T;xy=projected[:,:2]/projected[:,2:]
    return [float(np.clip(xy[:,0].min(),0,size)),float(np.clip(xy[:,1].min(),0,size)),
            float(np.clip(xy[:,0].max()+1,0,size)),float(np.clip(xy[:,1].max()+1,0,size))]


def build_synthetic_dataset(source_dir, out_dir, seed=17, max_objects=24, size=256):
    """Create seven controlled conditions with equally transformed absent pairs.

    Each target is ONE source photograph. Source photos are split before query
    generation. Absent queries come from another source within the same split;
    same-parent-folder negatives are preferred, then thumbnail-nearest selection.
    This is a proxy stress test, not a benchmark of new photographic viewpoints.
    """
    if max_objects<4 or size<32:
        raise ValueError("Need at least four source identities and size >= 32")
    source_dir=Path(source_dir).resolve();out_dir=Path(out_dir).resolve()
    if not source_dir.is_dir():raise ValueError("source_dir does not exist")
    if out_dir==source_dir or out_dir.is_relative_to(source_dir):
        raise ValueError("Place generated dataset outside the source tree")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Output directory must be empty to preserve existing data")
    candidates=sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"})
    rng=np.random.default_rng(seed)
    candidates=[candidates[i] for i in rng.permutation(len(candidates))]
    sources=[];seen=set();skipped=[]
    for p in candidates:
        try:
            original=load_rgb(p)
            if min(original.shape[:2])<32:continue
            image=ImageOps.fit(Image.fromarray(original),(size,size),method=Image.Resampling.LANCZOS)
            digest=pixel_sha256(np.asarray(image))
            if digest in seen:continue
            seen.add(digest)
            sources.append({"path":p,"image":image,"pixel_hash":digest,"raw_sha256":file_sha256(p),
                            "category_hint":p.parent.name})
            if len(sources)>=max_objects:break
        except (OSError,ValueError) as exc:
            skipped.append({"file":str(p.relative_to(source_dir)),"error":str(exc)[:100]})
    if len(sources)<4:raise ValueError("At least four distinct readable source images are required")
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"references").mkdir();(out_dir/"queries").mkdir()
    n_dev=max(2,min(len(sources)-2,int(round(len(sources)*.4))))
    objects=[]
    for i,source in enumerate(sources):
        oid=f"image_{i:03d}_{source['pixel_hash'][:10]}";split="dev" if i<n_dev else "test"
        relative=f"references/{oid}.png";source["image"].save(out_dir/relative)
        source.update(object_id=oid,split=split,reference=relative,reference_sha256=file_sha256(out_dir/relative))
        source["thumbnail"]=np.asarray(source["image"].resize((16,16)),dtype=float).reshape(-1)/255.
        objects.append({"object_id":oid,"split":split,"reference":relative,
                        "source_relative_path":str(source["path"].relative_to(source_dir)).replace("\\","/"),
                        "source_sha256":source["raw_sha256"],"source_id":source["pixel_hash"],
                        "category_hint":source["category_hint"],"target_observations":1})
    pairs=[]
    for i,source in enumerate(sources):
        candidates=[s for s in sources if s["split"]==source["split"] and s["object_id"]!=source["object_id"]]
        same_category=[s for s in candidates if s["category_hint"]==source["category_hint"]]
        candidates=same_category or candidates
        negative=min(candidates,key=lambda s:float(np.mean((s["thumbnail"]-source["thumbnail"])**2)))
        for c,condition in enumerate(CONDITIONS):
            transform_seed=seed*100003+i*101+c
            for present,query_source in ((True,source),(False,negative)):
                query,h,parameters=_transform(query_source["image"],condition,np.random.default_rng(transform_seed))
                suffix="present" if present else "absent"
                pid=f"{source['object_id']}_{condition}_{suffix}";relative=f"queries/{pid}.png"
                query.save(out_dir/relative)
                row={"id":pid,"object_id":source["object_id"],"reference":source["reference"],
                     "query":relative,"present":present,"condition":condition,"split":source["split"],
                     "reference_sha256":source["reference_sha256"],"query_sha256":file_sha256(out_dir/relative),
                     "source_id":source["pixel_hash"],"query_source_id":query_source["pixel_hash"],
                     "query_object_id":query_source["object_id"],"derived_from_reference":present,
                     "transform_seed":transform_seed,"transform_parameters":parameters,
                     "negative_selection":None if present else "thumbnail-nearest; same folder category preferred"}
                if present:
                    row["H_ref_to_query"]=h.tolist();row["target_bbox"]=_footprint_bbox(h,size)
                pairs.append(row)
    manifest={"schema_version":SCHEMA_VERSION,"name":"HyperNet same-source stress test",
              "protocol":"synthetic_same_source","seed":int(seed),"size":int(size),
              "identity_definition":"A source photograph is an image identity; physical object identities are not annotated.",
              "limitations":["All positive queries are transformations of the reference photograph.",
                  "Background pixels can identify the image; this is not foreground-only object recognition.",
                  "2D perspective transforms do not simulate independently observed 3D surfaces.",
                  "Absent means a different source photograph, not independently verified physical-object absence.",
                  "Target bounding boxes cover the transformed source-image footprint, not a segmented object."],
              "source_provenance":{"source_directory_name":source_dir.name,"source_root_runtime":str(source_dir),
                                   "selection":"Seeded permutation, distinct decoded/cropped image content", "skipped":skipped},
              "objects":objects,"pairs":pairs}
    manifest["validation"]=validate_manifest(manifest,out_dir)
    target=out_dir/"manifest.json"
    target.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return target
