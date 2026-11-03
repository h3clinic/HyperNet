"""Hash-checked plot-only RGB access, separating labels from model inputs."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image
from .gravity_spy_quality import is_constant_plot, RULE_ID


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GravitySpyCohort:
    def __init__(self, data_root, test_admission=None, *, apply_quality=True):
        self.root=Path(data_root).resolve()
        self.plan_path=self.root/'plan.json'
        self.raw_plan=json.loads(self.plan_path.read_text(encoding='utf-8'))
        self.plan=json.loads(self.plan_path.read_text(encoding='utf-8'))
        freeze=json.loads((self.root/'roster_freeze.json').read_text(encoding='utf-8'))
        if digest(self.plan_path)!=freeze['roster_sha256']:
            raise ValueError('Frozen roster mismatch')
        self.roster_sha256=freeze['roster_sha256']
        self.events={r['event_id']:r for r in self.plan['events']}
        self.apply_quality=apply_quality
        self.quality=None
        self.quality_sha256=None
        self.prior_exposure=None;self.prior_exposure_sha256=None
        prior_path=self.root/'prior_exposure_admission.json'
        if prior_path.exists():
            self.prior_exposure=json.loads(prior_path.read_text(encoding='utf-8'))
            if self.prior_exposure.get('roster_sha256')!=self.roster_sha256:
                raise ValueError('Prior-exposure amendment does not bind frozen roster')
            self.prior_exposure_sha256=digest(prior_path)
        quality_path=self.root/'quality_admission.json'
        if apply_quality and quality_path.exists():
            self.quality=json.loads(quality_path.read_text(encoding='utf-8'))
            if self.quality.get('roster_sha256')!=self.roster_sha256 or self.quality.get('rule')!=RULE_ID:
                raise ValueError('Quality amendment does not bind this roster/rule')
            source_path=Path(__file__).with_name('gravity_spy_quality.py')
            if self.quality.get('quality_source_sha256')!=digest(source_path):
                raise ValueError('Frozen quality rule changed')
            self.quality_sha256=digest(quality_path)
            self.plan['target_labels']=list(self.quality['eligible_target_labels'])
            self.plan['reference_by_label']={key:value for key,value in self.raw_plan['reference_by_label'].items()if key in self.plan['target_labels']}
        self.test_admission=None
        if test_admission is not None:
            from .gravity_spy_admission import verify_test_admission
            self.test_admission=verify_test_admission(self.root,test_admission)

    @property
    def target_labels(self):
        return tuple(self.plan['target_labels'])

    def query_events(self, split='calibration', *, include_prior_exposed=False):
        if split not in('calibration','test'):
            raise ValueError('Expected calibration or test')
        excluded=set(self.quality['excluded_event_ids'])if self.quality else set()
        test_quality=self.root/'quality_test_audit.json'
        if self.apply_quality and split=='test'and test_quality.exists():
            report=json.loads(test_quality.read_text(encoding='utf-8'))
            if report.get('roster_sha256')!=self.roster_sha256 or report.get('quality_admission_sha256')!=self.quality_sha256:
                raise ValueError('TEST quality report does not bind frozen quality policy')
            excluded.update(report['excluded_event_ids'])
        primary=None
        if split=='test'and not include_prior_exposed:
            if self.quality and self.prior_exposure is None:raise PermissionError('Prior event-exposure review is incomplete')
            if self.prior_exposure:primary=set(self.prior_exposure['primary_test_event_ids'])
        return tuple(dict(row) for row in self.plan['events']if row['split']==split and row['event_id']not in excluded and(primary is None or row['event_id']in primary))

    def _path(self, value):
        path=(self.root/value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image path escapes cohort')
        return path

    def load_rgb(self, event_id):
        row=self.events[event_id]
        if row['split']=='test'and self.test_admission is None:
            raise PermissionError('TEST pixels are sealed until calibration is frozen')
        if row['split']=='test'and(self.prior_exposure is None or event_id not in set(self.prior_exposure['primary_test_event_ids'])):
            raise PermissionError('Previously exposed TEST events are not authorized as fresh evaluation pixels')
        if row['split']not in('reference','calibration','test'):
            raise PermissionError('Excluded event is not a model input')
        sidecar=self._path('image_records/'+event_id+'.json')
        record=json.loads(sidecar.read_text(encoding='utf-8'))
        if any(record.get(k)!=row.get(k)for k in('event_id','image_url','image_path','split','group_id')):
            raise ValueError('Acquired image does not match frozen event metadata')
        if record['crop_xyxy']!=self.plan['plot_crop_xyxy']:
            raise ValueError('Plot crop differs from predeclared crop')
        image_path=self._path(row['image_path'])
        if digest(image_path)!=record['crop_png_sha256']:
            raise ValueError('Cropped PNG changed')
        with Image.open(image_path)as image:
            rgb=np.asarray(image.convert('RGB')).copy()
        if list(rgb.shape[:2][::-1])!=record['crop_dimensions']or hashlib.sha256(rgb.tobytes()).hexdigest()!=record['crop_rgb_sha256']:
            raise ValueError('Decoded RGB mismatch')
        if self.apply_quality and self.quality and is_constant_plot(rgb):
            raise ValueError('Excluded source image: exactly constant RGB plot')
        return rgb

    def reference(self,label):
        return self.load_rgb(self.plan['reference_by_label'][label])

    def provenance(self):
        return {'dataset':'Gravity Spy v1.1','roster_sha256':self.roster_sha256,
            'one_source_image_per_reference':True,'duration_seconds':self.plan['duration_seconds'],
            'crop_xyxy':self.plan['plot_crop_xyxy'],'real_weight_training':False,
            'additional_calibration_events':self.plan['counts']['split_events']['calibration'],
            'eligible_calibration_events':len(self.query_events('calibration')),
            'quality_admission_sha256':self.quality_sha256,
            'prior_exposure_admission_sha256':self.prior_exposure_sha256,
            'raw_target_count':len(self.raw_plan['target_labels']),'eligible_target_count':len(self.target_labels),
            'test_pixels_access_enabled':self.test_admission is not None,
            'source_url':self.plan['dataset_url'],'license':self.plan['license']}
