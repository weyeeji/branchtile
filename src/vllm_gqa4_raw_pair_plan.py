"""One serial B8 workspace across all36 GQA4 layers; native metadata aliases."""
import torch
from src.xqa_raw_gqa4_pair_runtime import RawGQA4PairPlan


class LayerSharedGQA4RawPairBank:
    def __init__(self, modules, caches, native_table, native_lengths, prefix_tokens, enabled_batches):
        assert tuple(enabled_batches)==(8,) and len(caches)==36
        assert all(c.dtype==torch.bfloat16 and c.shape[1:]==(4,16,256) for c in caches)
        assert native_lengths.dtype==torch.int32 and native_lengths.is_contiguous()
        assert native_table.shape[0]==8 and native_table.dtype==torch.int32 and native_table.is_contiguous()
        self.prefix_tokens=prefix_tokens;self.caches=list(caches);self.enabled_batches=(8,)
        q=torch.empty((8,16,128),device=caches[0].device,dtype=torch.bfloat16)
        p=RawGQA4PairPlan(modules,q,caches[0],native_table,[prefix_tokens+1]*8,prefix_tokens)
        p.lengths=native_lengths[:8].view(torch.uint32).reshape(8,1)
        self.plans={8:p};self.workspace=p.workspace;self.raw_state=p.state
        self.views=[(c[...,:128],c[...,128:],c[...,:128].transpose(1,2),c[...,128:].transpose(1,2)) for c in caches]
        self.query_layouts={}

    def validate_aliases(self, table, lengths, b):
        assert b==8
        p=self.plans[b]
        assert table.shape==p.table.shape and table.stride()==p.table.stride() and table.data_ptr()==p.table.data_ptr()
        assert lengths.dtype==torch.int32 and lengths.is_contiguous() and lengths.numel()==8
        assert lengths.data_ptr()==p.lengths.data_ptr()

    def bind_prefix(self, table):
        prefix=table[:,:self.prefix_tokens//16]
        assert prefix.shape==(8,self.prefix_tokens//16)
        assert bool((prefix==prefix[:1]).all().item()),'Full cohort must share actual physical prefix pages'
        return {'physical_prefix_checked':True,'rows':8,'prefix_pages':self.prefix_tokens//16}

    def run_layer(self, layer, query, output):
        assert query.shape==output.shape==(8,16,128) and query.dtype==output.dtype==torch.bfloat16
        p=self.plans[8]
        self.query_layouts[layer]={'shape':list(query.shape),'stride':list(query.stride()),'output_stride':list(output.stride()),'Q_canonicalization_copy_charged':True}
        p.q.copy_(query);p.k,p.v,p.wk,p.wv=self.views[layer]
        output.copy_(p.run());return output

    def receipt(self):
        p=self.plans[8]
        return {'enabled_batches':[8],'layers':36,'Hq':16,'Hkv':4,'GQA':4,
                'workspace_allocation_sets':1,'raw_state_allocation_sets':1,'workspace_bytes':self.workspace.numel(),
                'raw_state_bytes':self.raw_state.numel()*4,'query_layouts':self.query_layouts,
                'plans':{8:{'native_batch':8,'native_table_shape':list(p.table.shape),'native_table_stride':list(p.table.stride()),
                    'native_launch_extent':p.max_len,'native_splits':p.splits,'prefix_SM':p.splits*4*4,
                    'tail_SM':p.splits*8*4,'native_SM':p.sm,'prefix_tiles':p.prefix//256}},
                'lengths_policy':'Alias native GPU int32 seq_lens as uint32; no host copy or extra metadata kernel.',
                'copy_policy':'Q canonicalization/packing,pair table+length refresh,O copy charged every layer. KV remains in original storage.',
                'runtime_candidate_PDL':False,'concurrency':'Single stream, shared state/workspace across36layers; no concurrent replay.'}
