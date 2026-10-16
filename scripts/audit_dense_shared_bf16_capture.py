"""CPU provenance/layout/content audit for actual native BF16 capture artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def audit(path):
    torch.set_num_threads(4)
    summary = json.loads(path.read_text())
    assert summary['status'] == 'completed' and summary['args']['kv_dtype'] == 'auto'
    assert summary['engine']['kv_cache_dtype'] == 'auto'
    for source,expected in summary['source_sha256'].items():
        assert sha(ROOT/source) == expected, source
    assert sha(ROOT/summary['protocol']['manifest']) == summary['source_manifest_sha256']
    assert len(summary['workers']) == len(summary['native_receipts']) == 2
    result = {'status':'qualified', 'summary_sha256':sha(path), 'audit_source_sha256':sha(Path(__file__)),
              'source_hashes_verified':True,'actual_native_kv_dtype':'torch.bfloat16','ranks':[],
              'scope':'CPU artifact provenance and layout audit. Does not establish a new kernel numerical or performance result.'}
    for worker,native in zip(summary['workers'],summary['native_receipts']):
        capture = worker['dense_capture']
        assert worker['rank'] == native['rank'] == capture['rank']
        assert worker['final_inputs_tables_KV_hidden_unchanged']
        assert all(all(v for k,v in check.items() if k!='mode') for check in worker['checks'])
        assert capture['source_native_kv_dtype'] == 'torch.bfloat16' and not capture['payload_cast_performed']
        assert [r['layer'] for r in native['layers']] == list(range(48))
        q_path=ROOT/capture['q_file']
        assert sha(q_path)==capture['q_sha256']
        qo=torch.load(q_path,map_location='cpu',weights_only=True)
        assert sorted(qo['q'])==sorted(qo['o'])==list(range(48))
        assert qo['positions'].tolist()==worker['positions']
        for layer,receipt in enumerate(native['layers']):
            layout=capture['source_cache_layouts'][str(layer)]
            scales=capture['layer_scales'][str(layer)]
            assert receipt['storage_dtype']==layout['dtype']=='torch.bfloat16'
            assert receipt['shape']==layout['shape'] and receipt['stride']==layout['stride']
            assert scales['k_scale']==receipt['host_scales']['_k_scale_float']
            assert scales['v_scale']==receipt['host_scales']['_v_scale_float']
            for name in ('q','o'):
                tensor=qo[name][layer]
                assert tensor.shape==(8,16,128) and tensor.dtype==torch.bfloat16
                assert tensor_sha(tensor)==worker['layer_checks'][layer][name+'_sha256']
                assert torch.isfinite(tensor).all()
        assert [r['layer'] for r in capture['layer_files']]==[0,15,31,47]
        layers=[]
        for file in capture['layer_files']:
            layer=file['layer']
            artifact_path=ROOT/file['path']
            assert artifact_path.stat().st_size==file['bytes'] and sha(artifact_path)==file['sha256']
            artifact=torch.load(artifact_path,map_location='cpu',weights_only=True)
            assert artifact['rank']==worker['rank'] and artifact['layer']==layer
            assert artifact['source_native_kv_dtype']==artifact['original_cache_dtype']=='torch.bfloat16'
            assert not artifact['payload_cast_performed']
            receipt=native['layers'][layer]
            assert artifact['original_cache_shape']==receipt['shape']
            assert artifact['original_cache_stride']==receipt['stride']
            assert artifact['k_scale']==receipt['host_scales']['_k_scale_float']
            assert artifact['v_scale']==receipt['host_scales']['_v_scale_float']
            assert torch.equal(artifact['q'],qo['q'][layer]) and torch.equal(artifact['o'],qo['o'][layer])
            common=capture['common_complete_pages']*16
            assert artifact['common_complete_pages']*16==common
            prefix=artifact['prefix_kv']
            assert prefix.shape==(2,common,2,128) and prefix.dtype==torch.bfloat16
            assert torch.isfinite(prefix).all()
            table=artifact['original_block_table']
            assert (table[:,:common//16]==table[:1,:common//16]).all()
            positions=artifact['positions']
            assert torch.equal(positions,qo['positions'])
            slots=table[torch.arange(8),positions//16]*16+positions%16
            assert slots.tolist()==worker['physical_slots']
            tails=artifact['tail_kv']
            assert len(tails)==8
            for i,tail in enumerate(tails):
                assert tail.shape==(2,int(positions[i])+1-common,2,128)
                assert tail.dtype==torch.bfloat16 and torch.isfinite(tail).all()
            layers.append({'layer':layer,'artifact_sha256':file['sha256'],'prefix_tokens':common,
                           'private_tail_tokens':[t.shape[1] for t in tails],
                           'native_cache_shape':receipt['shape'],'native_cache_stride':receipt['stride'],
                           'k_scale':artifact['k_scale'],'v_scale':artifact['v_scale']})
        result['ranks'].append({'rank':worker['rank'],'all48_qo_hashes_match':True,'layers':layers})
    assert not torch.cuda.is_initialized()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    assert not output.exists(),output
    report=audit(Path(args.input))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
