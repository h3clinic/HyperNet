import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_details import PixelLedger,save_torch


def test_ledger_distinguishes_ids_geometry_and_rgb_uniqueness(tmp_path):
    ledger=PixelLedger(tmp_path/'ledger',shard_size=2)
    canonical=np.zeros((3,8,8),np.uint8);canonical[2,3,3]=255
    a=np.zeros((3,8,8,3),np.uint8);b=a.copy();a[1,2,2]=100;b[2,4,4]=200
    ledger.append(np.arange(3),{'canonical_mask':canonical>0,'canonical_raster':canonical,'view_a':a,'view_b':b})
    result=ledger.finish()
    assert result['unique_family_ids']==3
    assert result['unique_canonical_raster_sha256']==2
    assert result['unique_canonical_mask_sha256']==2
    assert result['unique_training_rgb_sha256']==3
    assert result['rendered_training_views']==6


def test_atomic_checkpoint_uses_only_safe_primitive_metadata(tmp_path):
    import torch
    path=tmp_path/'weights.pt'
    save_torch(path,{'state_dict':{'weight':torch.ones(3)},'runtime':{'torch':str(torch.__version__)}})
    value=torch.load(path,weights_only=True)
    assert value['runtime']['torch']==str(torch.__version__)
    assert not (tmp_path/'weights.pt.writing').exists()
