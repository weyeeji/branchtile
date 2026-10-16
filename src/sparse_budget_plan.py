"""CPU budget resolution and explicit reuse provenance for the A4 reference."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_budgets(text, mode):
    values=[]
    for item in text.split(','):
        item=item.strip().lower()
        if mode=='fraction':
            value=float(item)
            assert 0<value<=1, 'Fraction outside (0,1]'
        elif item=='full':
            value='full'
        else:
            value=int(item[:-1])*1024 if item.endswith('k') else int(item)
            assert value>=36, 'Absolute document budget must include first4/last32'
        assert value not in values, 'Duplicate nominal budget'
        values.append(value)
    return values


def resolve_budget(requested, mode, document_tokens):
    assert document_tokens>0
    actual=(document_tokens if requested=='full' else
            math.ceil(document_tokens*requested) if mode=='fraction' else requested)
    actual=min(document_tokens,max(36,actual))
    return {'budget_mode':mode,'requested_budget':requested,'budget_tokens':actual,
            'fraction':requested if mode=='fraction' else actual/document_tokens,
            'actual_document_fraction':actual/document_tokens,
            'saturated_to_full':actual==document_tokens}


IDENTITY_KEYS=('manifest_sha256','model_revision','weights_manifest_sha256',
               'model_config_sha256','core_sha256','generation','selector')


def load_full_sources(paths, expected):
    """Strictly validate frozen inputs before reusing completed full generations."""
    records={};sources=[]
    for name in paths:
        path=Path(name)
        data=json.loads(path.read_text())
        assert data['status']=='completed' and data['mode']=='natural',name
        for key in IDENTITY_KEYS:
            assert data[key]==expected[key],f'Reuse identity mismatch {key}: {name}'
        source_sha=digest(path)
        sources.append({'path':str(path),'sha256':source_sha})
        for group in data['groups']:
            ds,de=group['document_range']
            for question in group['questions']:
                assert question['status']=='completed'
                full=[v for v in question['variants'] if v['budget_tokens']==de-ds]
                assert len(full)==1, 'Source must contain exactly one full-budget generation'
                variant=full[0]
                assert variant['status']=='completed'
                assert variant['seed']==expected['generation']['seed']+question['question_index']
                assert len(variant['generated_ids'])==variant['generated_tokens']
                key=(group['group_index'],question['question_index'])
                assert key not in records, f'Duplicate reuse source {key}'
                artifact=Path(variant['mask_artifact'])
                assert digest(artifact)==variant['mask_artifact_sha256'],'Source mask artifact changed'
                source_id=f'{source_sha}:g{key[0]}:q{key[1]}:b{de-ds}'
                records[key]={'variant':variant,'document_sha256':group['document_sha256'],
                    'document_range':group['document_range'],'question_sha256':question['question_sha256'],
                    'input_ids_sha256':question['input_ids_sha256'],'prompt_tokens':question['prompt_tokens'],
                    'provenance':{'source_path':str(path),'source_sha256':source_sha,
                        'group_index':key[0],'question_index':key[1],'source_fraction':variant['fraction'],
                        'source_budget_tokens':de-ds,'source_generation_id':source_id,
                        'source_generated_ids_sha256':hashlib.sha256(json.dumps(variant['generated_ids'],separators=(',',':')).encode()).hexdigest()}}
    return records,sources


def validate_reuse(record,group,question):
    for key in ('document_sha256','document_range'):
        assert record[key]==group[key],f'Reuse document mismatch: {key}'
    for key in ('question_sha256','input_ids_sha256','prompt_tokens'):
        assert record[key]==question[key],f'Reuse question mismatch: {key}'


def reused_variant(record,budget):
    value=copy.deepcopy(record['variant'])
    costs={key:value.pop(key) for key in list(value) if '_seconds' in key}
    value.update(budget,generation_origin='reused_prior_full_budget',new_generation_performed=False,
                 measurement_origin='historical_source_only; no new selector/gather/decode measurement',
                 historical_reference_cost_seconds=costs,**record['provenance'])
    return value
