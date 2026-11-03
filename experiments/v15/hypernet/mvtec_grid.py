"""Pure MVTec Grid data access; annotations are separate from model RGB inputs."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from PIL import Image

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pixels(array):return sha256(str(array.dtype).encode()+str(array.shape).encode()+np.ascontiguousarray(array).tobytes()).hexdigest()
def safe_path(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):raise ValueError('Dataset path escapes root')
    return path

class GridDataset:
    """One reference, separate normal calibration, and gated official TEST.

    load_rgb returns uint8 HWC original-resolution RGB. Grayscale inputs are
    repeated across channels; no resizing/cropping/normalization is performed.
    load_evaluation_target returns a boolean original-size mask separately.
    """
    def __init__(self,root,*,development_only=True,admission_path=None):
        self.root=Path(root).resolve();self.development_only=development_only
        self.manifest_path=self.root/('development_manifest.json' if development_only else 'manifest.json')
        self.manifest=read(self.manifest_path)
        if self.manifest.get('format')!='hypernet.mvtec_grid.manifest.v15' or not self.manifest.get('completed'):
            raise ValueError('Invalid Grid manifest')
        self.records={r['id']:r for r in self.manifest['records']}
        if len(self.records)!=len(self.manifest['records']):raise ValueError('Duplicate image IDs')
        self.admission_path=Path(admission_path).resolve() if admission_path else self.root/'test_admission.json'
        self._verified_admission=None

    def ids(self,role):return sorted(key for key,record in self.records.items() if record['role']==role)

    def _test_gate(self):
        if self.development_only:raise ValueError('Development loader cannot read TEST')
        if not self.admission_path.exists():raise ValueError('TEST requires a frozen protocol admission')
        admission=read(self.admission_path)
        if admission.get('test_pixels_authorized_for_frozen_inference') is not True or admission['manifest_sha256']!=digest(self.manifest_path):
            raise ValueError('Admission/manifest binding differs')
        protocol_path=Path(admission['protocol_path']);protocol=read(protocol_path)
        if digest(protocol_path)!=admission['protocol_sha256'] or not protocol.get('model_selection_complete') or protocol.get('test_scores_computed_before_freeze') is not False:
            raise ValueError('Protocol admission changed')
        if protocol['grid_manifest_sha256']!=digest(self.manifest_path):raise ValueError('Protocol/manifest differs')
        # The evaluator verifies all behavior/checkpoint hashes once before and
        # after scoring. Here every access rechecks the immutable gate itself.
        self._verified_admission=admission['protocol_sha256']

    def _record(self,image_id):
        if image_id not in self.records:raise KeyError(image_id)
        record=self.records[image_id]
        if record['role']=='test':self._test_gate()
        elif record['role'] not in ('reference','dev_normal'):raise ValueError('Unused normals are not part of this pilot')
        return record

    def load_rgb(self,image_id):
        record=self._record(image_id);path=safe_path(self.root,record['rgb_path'])
        if digest(path)!=record['sha256']:raise ValueError('Image raw hash differs')
        with Image.open(path) as image:
            if image.getexif().get(274,1)!=1:raise ValueError('Unexpected orientation')
            native=np.array(image);rgb=np.array(image.convert('RGB'),dtype=np.uint8)
        if pixels(native)!=record['native_pixel_sha256'] or pixels(rgb)!=record['pixel_sha256']:
            raise ValueError('Decoded image pixels differ')
        if rgb.shape!=(record['height'],record['width'],3):raise ValueError('Image dimensions differ')
        return rgb

    def load_evaluation_target(self,image_id):
        record=self._record(image_id)
        if record['role']!='test':raise ValueError('Target masks are for official TEST evaluation only')
        entry=record['mask']
        if entry is None:
            if record['is_anomaly']:raise ValueError('Missing anomaly mask')
            mask=np.zeros((record['height'],record['width']),dtype=bool)
        else:
            path=safe_path(self.root,entry['path'])
            if digest(path)!=entry['sha256']:raise ValueError('Mask file changed')
            with Image.open(path) as image:array=np.array(image)
            if pixels(array)!=entry['pixel_sha256'] or array.shape!=(record['height'],record['width']):raise ValueError('Mask pixels/shape changed')
            mask=array>0
        return {'is_anomaly':record['is_anomaly'],'mask':mask,'defect_type':record['defect_type']}
