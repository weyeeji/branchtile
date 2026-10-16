"""Exact CPU equivalence fixture on frozen suffix-only pooled scores; no model run."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.joint_support_exchange import (
    FROZEN_SOURCE_SHA256, PROTOCOL_ID, PROTOCOL_SHA256,
    JointSupportExchangeConfig, apply_joint_exchange,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    out=Path(args.output);assert not out.exists(),'Keep prior qualification evidence'
    protocol_path=ROOT/'results/joint_support_exchange_v1/protocol.json'
    reference_path=ROOT/'scripts/probe_joint_support_exchange.py'
    assert sha(protocol_path)==PROTOCOL_SHA256 and sha(reference_path)==FROZEN_SOURCE_SHA256
    protocol=json.loads(protocol_path.read_text())
    nodes=[n for n in ast.parse(reference_path.read_text()).body if isinstance(n,ast.FunctionDef)
           and n.name in ('score_sum','stable_order','exchange')]
    namespace={'np':np,'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(reference_path),'exec'),namespace)
    reference=namespace['exchange']
    started=time.perf_counter();cpu_started=time.process_time();records=[]
    for source in protocol['sources']:
        directory=ROOT/source['directory'];metadata_path=directory/'metadata.json'
        assert sha(metadata_path)==source['metadata_sha256']
        metadata=json.loads(metadata_path.read_text())
        start,end=metadata['document_token_range'];common_end=metadata['actual_shared_document_token_range'][1]
        scores=[];masks={k:[] for k in protocol['budgets']}
        for item in source['selector_files']:
            path=directory/item['name'];assert sha(path)==item['sha256']
            value=torch.load(path,map_location='cpu',weights_only=False)
            assert value['observation_is_question_suffix_only'] is True
            # Load only the pre-existing score/support fields into the fixture;
            # neither observation_q values nor future decode Q/O are inspected.
            scores.append(value['pooled_scores'].numpy().copy())
            for k in masks:masks[k].append(value['supports'].get(k,value['supports'].get(str(k))).numpy().copy())
            del value
        score=np.stack(scores);protected=np.zeros(end-start,dtype=bool)
        protected[:4]=True;protected[-32:]=True
        for budget,values in masks.items():
            baseline=np.stack(values)
            for epsilon in protocol['epsilon']:
                for order_name in protocol['request_orders']:
                    config=JointSupportExchangeConfig(PROTOCOL_ID,PROTOCOL_SHA256,epsilon,order_name,2,4,32)
                    independent,actual,record=apply_joint_exchange(score,baseline,config,
                        document_start=start,common_source_tokens=common_end-start)
                    assert np.array_equal(independent,baseline)
                    order=list(range(len(score)))
                    if order_name=='descending':order.reverse()
                    for head in range(score.shape[1]):
                        expected,events=reference(score[:,head].astype(np.float64),baseline[:,head],protected,epsilon,order,2)
                        assert np.array_equal(expected,actual[:,head])
                        assert events==record['records'][head]['events']
                    records.append({'source':source['id'],'metadata_sha256':source['metadata_sha256'],
                        'budget':budget,'exact_support_and_events_equal_frozen_reference':True,**record})
    assert len(records)==48 and not torch.cuda.is_initialized()
    result={'status':'cpu_algorithm_equivalence_qualified','protocol_sha256':PROTOCOL_SHA256,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'src/joint_support_exchange.py',ROOT/'tests/test_joint_support_exchange.py')},
        'uid':os.geteuid(),'python':sys.executable,'gpu_hours':0,'cuda_initialized':False,
        'cohort_configs':len(records),'head_configs':sum(len(r['records']) for r in records),
        'CPU_wall_seconds':time.perf_counter()-started,'CPU_process_seconds':time.process_time()-cpu_started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'records':records,
        'scope':'All frozen CPU protocol epsilon/order combinations retained. Exact algorithm port only; FrozenQCSelector integration and natural quality untested. No O/LSE or mass-loss guarantee from pooled-score epsilon.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cohort_configs','head_configs','CPU_wall_seconds','cuda_initialized')}))


if __name__=='__main__':main()
