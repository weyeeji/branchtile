"""GQA4 B6/7/8 bank: one workspace/raw state across all36 layers and batch graphs."""
import torch
from src.xqa_raw_gqa4_padded_shared_runtime import RawGQA4SharedPaddedPlan


class LayerSharedGQA4SurvivorBank:
    def __init__(self, modules, caches, native_table, native_lengths, prefix_tokens, enabled_batches):
        assert tuple(enabled_batches)==(6,7,8) and len(caches)==36
        assert all(c.dtype==torch.bfloat16 and c.shape[1:]==(4,16,256) for c in caches)
        assert native_lengths.dtype==torch.int32 and native_lengths.is_contiguous() and native_lengths.numel()>=8
        assert native_table.shape[0]==8 and native_table.dtype==torch.int32 and native_table.is_contiguous()
        self.prefix_tokens=prefix_tokens;self.caches=list(caches);self.enabled_batches=(6,7,8)
        self.workspace=torch.zeros(64*1024*1024,device=caches[0].device,dtype=torch.uint8)
        sm=torch.cuda.get_device_properties(caches[0].device).multi_processor_count
        extent=native_table.shape[1]*16
        self.capacity_by_batch={b:((b+1)//2)*4*min(max(1,sm//(b*4)),(extent+255)//256)*4*32*35 for b in self.enabled_batches}
        self.raw_state=torch.empty(max(self.capacity_by_batch.values()),device=caches[0].device,dtype=torch.float32)
        self.plans={}
        for b in self.enabled_batches:
            q=torch.empty((b,16,128),device=caches[0].device,dtype=torch.bfloat16)
            p=RawGQA4SharedPaddedPlan(modules,q,caches[0],native_table[:b],[prefix_tokens+1]*b,prefix_tokens,
                                    workspace=self.workspace,raw_state=self.raw_state)
            p.lengths=native_lengths[:b].view(torch.uint32).reshape(b,1)
            assert p.state.data_ptr()==self.raw_state.data_ptr() and p.workspace.data_ptr()==self.workspace.data_ptr()
            self.plans[b]=p
        self.views=[(c[...,:128],c[...,128:],c[...,:128].transpose(1,2),c[...,128:].transpose(1,2)) for c in caches]
        self.query_layouts={}

    def validate_aliases(self, table, lengths, b):
        assert b in self.plans
        p=self.plans[b]
        assert table.shape==p.table.shape and table.stride()==p.table.stride() and table.data_ptr()==p.table.data_ptr()
        assert lengths.dtype==torch.int32 and lengths.is_contiguous() and lengths.numel()==b
        assert lengths.data_ptr()==p.lengths.data_ptr()

    def bind_prefix(self, table):
        b=table.shape[0];prefix=table[:,:self.prefix_tokens//16]
        assert prefix.shape==(b,self.prefix_tokens//16)
        assert bool((prefix==prefix[:1]).all().item()),'Full cohort must share actual physical prefix pages'
        return {'physical_prefix_checked':True,'rows':b,'prefix_pages':self.prefix_tokens//16}

    def run_layer(self, layer, query, output):
        b=query.shape[0]
        assert b in self.plans and query.shape==output.shape==(b,16,128) and query.dtype==output.dtype==torch.bfloat16
        p=self.plans[b]
        self.query_layouts.setdefault(b,{})[layer]={'shape':list(query.shape),'stride':list(query.stride()),'output_stride':list(output.stride()),'Q_canonicalization_copy_charged':True}
        p.q.copy_(query);p.k,p.v,p.wk,p.wv=self.views[layer]
        output.copy_(p.run());return output

    def receipt(self):
        return {'enabled_batches':list(self.enabled_batches),'layers':36,'Hq':16,'Hkv':4,'GQA':4,
                'workspace_allocation_sets':1,'raw_state_allocation_sets':1,'workspace_bytes':self.workspace.numel(),
                'raw_state_bytes':self.raw_state.numel()*4,'query_layouts':self.query_layouts,'capacity_by_batch_floats':self.capacity_by_batch,
                'plans':{b:{'native_batch':b,'dummy_prefix_rows':b%2,'native_table_shape':list(p.table.shape),'native_table_stride':list(p.table.stride()),
                    'native_launch_extent':p.max_len,'native_splits':p.splits,'prefix_SM':p.prefix_sm,
                    'tail_SM':p.tail_sm,'native_SM':p.sm,'prefix_tiles':p.prefix//256} for b,p in self.plans.items()},
                'lengths_policy':'Alias native GPU int32 seq_lens as uint32; no host copy or extra metadata kernel.',
                'copy_policy':'Q canonicalization/odd padding/packing,pair metadata refresh,O copy charged. KV remains in original storage.',
                'runtime_candidate_PDL':False,'concurrency':'Single stream,shared state/workspace across36layers and3batchgraphs; no concurrent replay.'}
