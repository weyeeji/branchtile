"""CPU reconstruction of captured dense page16 inputs with auditable aliases.

Only active physical page IDs are compacted. KV values, page-level sharing and
original intra-page strides are preserved; GPU benchmarks restore page pitch.
"""
import hashlib
import json
from pathlib import Path


def load_dense_capture_fixture(capture_path, rows=None):
    import torch
    path=Path(capture_path).resolve();receipt_path=path.parent/'receipt.json'
    receipt=json.loads(receipt_path.read_text());assert receipt['status']=='completed'
    record=next(x for x in receipt['layer_files'] if Path(x['path']).name==path.name)
    payload_sha=hashlib.sha256(path.read_bytes()).hexdigest();assert payload_sha==record['sha256']
    summary_path=path.parents[2]/'summary.json';summary=json.loads(summary_path.read_text());assert summary['status']=='completed'
    rank=int(path.parent.parent.name.removeprefix('rank'));layer=record['layer']
    native=next(x for x in summary['native_receipts'] if x['rank']==rank)
    native_layer=next(x for x in native['layers'] if x['layer']==layer)
    data=torch.load(path,map_location='cpu',weights_only=True)
    assert data['original_cache_shape']==native_layer['shape'] and data['original_cache_stride']==native_layer['stride']
    scales=native_layer['scales'];host=native_layer['host_scales']
    assert all(host[k+'_float']==v==1. for k,v in scales.items())
    assert tuple(data['original_cache_shape'][1:])==(2,16,256), 'Only observed fused page16 ABI is qualified'
    batch=data['q'].shape[0];rows=list(range(batch)) if rows is None else list(map(int,rows))
    assert rows and len(rows)==len(set(rows)) and min(rows)>=0 and max(rows)<batch
    common=data['common_complete_pages'];prefix_tokens=common*16
    tables=data['original_block_table'];lens=(data['positions']+1).tolist()
    assert all(torch.equal(tables[r,:common],tables[rows[0],:common]) for r in rows)
    ids=[];mapping={}
    for r in rows:
        for original in tables[r,:(lens[r]+15)//16].tolist():
            if original not in mapping:mapping[original]=len(ids);ids.append(original)
    prefix=data['prefix_kv'];tails=data['tail_kv']
    if prefix.dtype==torch.float8_e4m3fn:
        prefix=prefix.view(torch.uint8);tails=[x.view(torch.uint8) for x in tails]
    cpu_cache=torch.zeros((len(ids),2,16,256),dtype=prefix.dtype)
    filled=torch.zeros((len(ids),16),dtype=torch.bool)
    for page in range(common):
        dest=mapping[int(tables[rows[0],page])]
        # [KV, token, head, D] -> [head, token, KV*D].
        source=prefix[:,page*16:(page+1)*16].permute(2,1,0,3).reshape(2,16,256)
        cpu_cache[dest].copy_(source);filled[dest]=True
    for r in rows:
        assert tails[r].shape==(2,lens[r]-prefix_tokens,2,128)
        for start in range(0,lens[r]-prefix_tokens,16):
            count=min(16,lens[r]-prefix_tokens-start)
            dest=mapping[int(tables[r,common+start//16])]
            source=tails[r][:,start:start+count].permute(2,1,0,3).reshape(2,count,256)
            live=filled[dest,:count]
            if bool(live.any()):assert torch.equal(cpu_cache[dest,:,:count][:,live,:],source[:,live,:]), 'Conflicting payload for shared physical page'
            cpu_cache[dest,:,:count].copy_(source);filled[dest,:count]=True
    width=max((lens[r]+15)//16 for r in rows)
    compact_tables=torch.zeros((len(rows),width),dtype=torch.int32)
    for i,r in enumerate(rows):
        pages=(lens[r]+15)//16
        compact_tables[i,:pages]=torch.tensor([mapping[x] for x in tables[r,:pages].tolist()],dtype=torch.int32)
    provenance={'capture':str(path),'capture_sha256':payload_sha,'receipt':str(receipt_path),
        'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),'model_summary':str(summary_path),
        'model_summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),'rank':rank,'layer':layer,
        'document_id':summary['args']['document_id'],'selected_original_rows':rows,'common_prefix_tokens':prefix_tokens,
        'original_cache_shape':data['original_cache_shape'],'original_cache_stride':data['original_cache_stride'],
        'active_physical_pages':len(ids),'original_physical_page_ids':ids,'scales':scales,'native_precision_source_sha256':summary['source_sha256']['scripts/vllm_native_precision_worker.py'],
        'layout_scope':'Offline active-page-ID compaction; source KV values and physical sharing preserved. Original page pitch will be restored on GPU. This is not the live engine allocation or an online zero-copy export.'}
    return {'cache_cpu':cpu_cache,'q_cpu':data['q'][rows].contiguous(),'original_o_cpu':data['o'][rows].contiguous(),
        'tables_cpu':compact_tables,'seq_lens':[lens[r] for r in rows],'prefix_tokens':prefix_tokens,
        'k_scale':scales['_k_scale'],'v_scale':scales['_v_scale'],'original_stride':tuple(data['original_cache_stride']),
        'semantic_fp8':data['kv_semantic_dtype'].endswith('float8_e4m3fn'),'provenance':provenance}
