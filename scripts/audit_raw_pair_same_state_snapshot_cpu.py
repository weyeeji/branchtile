"""Independently inspect the first retained same-state numerical failure on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    torch.set_num_threads(4)
    path,output=Path(args.snapshot),Path(args.output);assert not output.exists()
    s=torch.load(path,map_location='cpu',weights_only=False)
    assert s['schema']=='native_raw_pair_same_state_v1'
    a,b=s['native_o'],s['candidate_o'];assert a.dtype==b.dtype==torch.bfloat16
    eq=lambda x,y:torch.equal(x.contiguous().view(torch.int16),y.contiguous().view(torch.int16))
    mismatches=[]
    for index in torch.nonzero(a.view(torch.int16)!=b.view(torch.int16)).tolist():
        t=tuple(index)
        mismatches.append({'coordinate':index,'question_index':s['questions'][index[0]],
            'native':a[t].item(),'candidate':b[t].item(),
            'native_bits_int16':a.view(torch.int16)[t].item(),'candidate_bits_int16':b.view(torch.int16)[t].item()})
    assert mismatches
    comparisons={key:{'bits_equal_native':eq(s[key],a),'bits_equal_candidate':eq(s[key],b)}
        for key in ('native_aa_o','native_pdl_off_o','witness_o','unshared_checkpoint_o','model_candidate_o','candidate_poison_o')}
    assert eq(s['q'],s['model_candidate_q'])
    assert all(comparisons[k]['bits_equal_native'] for k in ('native_aa_o','native_pdl_off_o','witness_o','unshared_checkpoint_o'))
    assert all(comparisons[k]['bits_equal_candidate'] for k in ('model_candidate_o','candidate_poison_o'))
    assert not torch.cuda.is_initialized()
    report={'status':'same_state_bitwise_failure_independently_confirmed_CPU',
        'snapshot':str(path),'snapshot_sha256':sha(path),'snapshot_bytes':path.stat().st_size,
        'audit_source_sha256':sha(Path(__file__)),
        **{k:s[k] for k in ('rank','layer','graph_index','questions','positions','prefix_tokens','S','SM','original_cache_shape','original_cache_stride','original_table_columns')},
        'Q_bits_equal':True,'differences':mismatches,'comparisons':comparisons,
        'max_abs_difference':float((a.float()-b.float()).abs().max()),
        'CUDA_initialized':torch.cuda.is_initialized(),
        'limits':['Direct same-Q attention output failure predates the later visible generated-token divergence; shared question identity is consistent with propagation but not a complete counterfactual causal proof.',
            'Native repeatability, PDL off, and unshared checkpoint match in this snapshot. This excludes those observed controls as explanations for these two output differences, not every possible runtime issue.',
            'The snapshot does not independently establish all previous generated tokens or all later layers; inspect diagnostic trajectory and full-model receipts separately.',
            'A very small absolute difference still falsifies the declared bitwise contract; no tolerance is substituted or failed heldout document discarded.']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'differences':mismatches,'CUDA_initialized':False},indent=2))


if __name__=='__main__':main()
