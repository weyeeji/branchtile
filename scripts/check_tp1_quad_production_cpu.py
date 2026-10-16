"""CPU completed-summary provenance and common-warmup cost tests for TP1 production."""
import copy,hashlib,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from src.thinking_tp1_offline_qualification import load_tp1_qualification
from src.vllm_gqa4_raw_pair_setup import REQUIRED_QUALIFICATION_FIELDS
from scripts.benchmark_thinking_tp1_quad_production import component_first_use

def main():
    protocol={'test':'CPU_fixture_not_a_GPU_receipt'};sources={'fixture':'current'};fp={'TP':1,'attention_shape':[36,32,8,128,16]};checks=[]
    for b in range(4,9):
        c={k:True for k in REQUIRED_QUALIFICATION_FIELDS};c.update(batch=b,qualification_passed=True,layer_checks=[{'layer':i,'Q_bits_equal':True,'O_bits_equal':True} for i in range(36)],modes=[{'mode':m,'hidden_bits_equal':True} for m in ('checked','production')]);checks.append(c)
    record={'rank':0,'source_sha256':sources,'execution_fingerprint':fp,'phase_valid':True,'validation':True,'method':'native','setup_mode':'strict','qualification_passed':True,'qualification_steps':5,'checks':checks,'captured_layers_by_batch':{m:{str(b):list(range(36)) for b in range(4,9)} for m in ('witness','checked','production')}}
    flags={}
    with tempfile.TemporaryDirectory() as dirname:
        tmp=Path(dirname);rp=tmp/'raw.json';rp.write_text(json.dumps(record));sp=tmp/'summary.json';driver='scripts/qualify_thinking_tp1_quad_model.py'
        summary={'status':'completed','protocol':protocol,'source_sha256':{driver:hashlib.sha256((ROOT/driver).read_bytes()).hexdigest()},'runs':[{'validation':True,'method':'native','workers':[{'rank':0,'raw_pair_record':str(rp)}],'qualification_joint_audit':[{'rank':0,'sha256':hashlib.sha256(rp.read_bytes()).hexdigest()}],'answers':[{'tokens':[1,2]}]}]}
        sp.write_text(json.dumps(summary));loaded=load_tp1_qualification(sp,sources,protocol,list(range(4,9)));assert loaded['reference_tokens']==[[1,2]]
        flags['completed_exact_summary_source_and_receipt_bytes_bound']=True
        for label,mutate in [('incomplete',lambda x:x.update(status='initializing')),('wrong_protocol',lambda x:x.update(protocol={})),('wrong_rank',lambda x:x['runs'][0]['workers'][0].update(rank=1)),('wrong_receipt_digest',lambda x:x['runs'][0]['qualification_joint_audit'][0].update(sha256='wrong'))]:
            bad=copy.deepcopy(summary);mutate(bad);sp.write_text(json.dumps(bad))
            try:load_tp1_qualification(sp,sources,protocol,list(range(4,9)))
            except ValueError:flags['reject_'+label]=True
            else:raise AssertionError(label)
    rows=[{'phase':f'natural_{m}_{i}','method':m,'full_phase_control_wall_seconds':30 if m=='native' else 25,'cold_shared_document_total_wall_seconds':29 if m=='native' else 24,'candidate_graph_executed':m=='cascade'} for i,m in enumerate(('native','cascade','cascade','native'))]
    report={'runs':rows,'common_sampling_warmup_wall_seconds':100,'common_initial_precision_export_wall_seconds':2,'offline_preflight_wall_seconds':1,'candidate_total_setup_wall_seconds':3,'setup_receipt_export_wall_seconds':1,'input_preparation_wall_seconds':5,'native_initialization_and_graph_setup_wall_seconds':50}
    result=component_first_use(report,'natural');assert result['native_net_after_engine_seconds']==132 and result['candidate_net_after_engine_seconds']==132
    assert result['native_first_use_component_sum_seconds']==187 and result['candidate_first_use_component_sum_seconds']==187
    flags['shared_warmup_and_exports_charged_once_and_candidate_setup_not_hidden']=True
    source=(ROOT/'scripts/benchmark_thinking_tp1_quad_production.py').read_text()
    assert source.index('warm_start=time.perf_counter()')<source.index('        phases=[]')
    assert "for warm_b in range(1,9):" in source and "prompts[:warm_b]" in source
    assert "sampling=dict(protocol['sampling'])" in source and "assert llm.reset_prefix_cache()" in source
    flags['full_prompt_common_sampling_warmup_precedes_unchanged_formal_ABBA']=True
    import torch
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status':'passed','cuda_initialized':False,'checks':flags,'limitations':'Temporary mock receipts test CPU provenance/accounting only, not model correctness. Actual strict-summary preparation remains required.'},indent=2))
if __name__=='__main__':main()
