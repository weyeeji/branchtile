import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sparse_budget_plan import parse_budgets,resolve_budget,load_full_sources,validate_reuse,reused_variant,digest,IDENTITY_KEYS


def test_absolute_grid_saturation_and_fraction_compatibility():
    grid=parse_budgets('4k,8K,16384,32768,full','tokens')
    assert grid==[4096,8192,16384,32768,'full']
    lengths=[6126,7476,13912,20896,7685,8965,8213,14773]
    rows=[resolve_budget(b,'tokens',n) for n in lengths for b in grid]
    assert sum(r['saturated_to_full'] for r in rows)*4==104
    assert sum(not r['saturated_to_full'] for r in rows)*4==56
    assert resolve_budget(.1,'fraction',6126)['budget_tokens']==613
    assert resolve_budget('full','tokens',20)['budget_tokens']==20
    for text,mode in [('35','tokens'),('1.1','fraction'),('4k,4096','tokens')]:
        with pytest.raises(AssertionError):parse_budgets(text,mode)


def source(tmp_path):
    mask=tmp_path/'mask.pt';mask.write_bytes(b'fixture')
    expected={key:key for key in IDENTITY_KEYS}
    expected['generation']={'seed':7}
    variant={'fraction':1.,'status':'completed','budget_tokens':100,'seed':9,
        'generated_ids':[1,2],'generated_tokens':2,'decode_seconds':3.,'gather_seconds':.2,
        'mask_artifact':str(mask),'mask_artifact_sha256':digest(mask)}
    question={'question_index':2,'status':'completed','question_sha256':'q',
        'input_ids_sha256':'ids','prompt_tokens':120,'variants':[variant]}
    group={'group_index':1,'document_sha256':'doc','document_range':[4,104],'questions':[question]}
    data={**expected,'status':'completed','mode':'natural','groups':[group]}
    path=tmp_path/'source.json';path.write_text(json.dumps(data))
    return path,expected,group,question


def test_reuse_validates_identity_and_separates_historical_cost(tmp_path):
    path,expected,group,question=source(tmp_path)
    records,sources=load_full_sources([path],expected)
    record=records[1,2];validate_reuse(record,group,question)
    value=reused_variant(record,resolve_budget(4096,'tokens',100))
    assert value['requested_budget']==4096 and value['budget_tokens']==100
    assert value['generated_ids']==[1,2] and not value['new_generation_performed']
    assert 'decode_seconds' not in value
    assert value['historical_reference_cost_seconds']['decode_seconds']==3.
    assert sources[0]['sha256']==digest(path)
    changed=copy.deepcopy(question);changed['input_ids_sha256']='changed'
    with pytest.raises(AssertionError):validate_reuse(record,group,changed)
    expected['model_revision']='changed'
    with pytest.raises(AssertionError):load_full_sources([path],expected)


def test_reuse_rejects_corrupted_artifacts_and_duplicate_sources(tmp_path):
    path,expected,group,question=source(tmp_path)
    with pytest.raises(AssertionError):load_full_sources([path,path],expected)
    (tmp_path/'mask.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError):load_full_sources([path],expected)
