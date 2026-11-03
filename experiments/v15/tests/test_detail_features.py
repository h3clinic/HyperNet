import numpy as np
import torch
import pytest
from hypernet.detail_encoder import DetailEncoder,FORMAT,state_hash
from hypernet.detail_features import FrozenDetails,tile_layout

def make_encoder(tmp_path):
    torch.manual_seed(5);model=DetailEncoder()
    checkpoint=tmp_path/'own_random.pt'
    torch.save({'format':FORMAT,'state_dict':model.state_dict(),'state_sha256':state_hash(model),
        'optimizer_steps':0,'natural_training_images':0,'borrowed_pretrained_weights':False},checkpoint)
    return FrozenDetails(checkpoint,'cpu',batch_size=8)

def test_native_tiles_cover_irregular_image_without_resizing(tmp_path):
    encoder=make_encoder(tmp_path);rng=np.random.default_rng(2);rgb=rng.integers(0,256,(101,145,3),dtype=np.uint8)
    before=encoder.provenance();features=encoder.encode(rgb)
    assert features['descriptors'].shape==(26,37,24)
    assert features['detail'].shape==(26,37,4)
    assert features['tile_count']==12
    assert features['source_pixels_resized']is False
    assert features['xy'][-1,-1,0]==pytest.approx(144.5/145)
    assert features['xy'][-1,-1,1]==pytest.approx(100.5/101)
    np.testing.assert_allclose(np.linalg.norm(features['descriptors'],axis=-1),1,atol=1e-6)
    assert encoder.provenance()==before

def test_selected_pixel_matches_exact_native_tile_forward(tmp_path):
    encoder=make_encoder(tmp_path);rng=np.random.default_rng(3);rgb=rng.integers(0,256,(96,96,3),dtype=np.uint8)
    features=encoder.encode(rgb);pad=np.pad(rgb,tile_layout(96,96)['pad'],mode='reflect')
    # Core pixel(52,56) belongs to tile(row1,col1), at local coordinate(12,16).
    tile=pad[48:112,48:112]
    with torch.inference_mode():encoded=encoder.model(torch.from_numpy(tile.copy()).permute(2,0,1)[None].float()/255)
    np.testing.assert_allclose(features['descriptors'][13,14],encoded['dense'][0,:,12,16].numpy(),atol=1e-6)

def test_rejects_borrowed_checkpoint(tmp_path):
    path=tmp_path/'bad.pt';torch.save({'format':FORMAT,'natural_training_images':0,'borrowed_pretrained_weights':True},path)
    with pytest.raises(ValueError):FrozenDetails(path,'cpu')
