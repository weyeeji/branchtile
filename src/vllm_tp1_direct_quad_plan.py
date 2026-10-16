"""TP1 Hq32/Hkv8 B4..8 quad qualification bank: one workspace/raw state across all36 layers and batch graphs."""
import torch
from src.xqa_raw_tp1_direct_quad_runtime import RawTP1DirectQuadPlan


class LayerSharedTP1DirectQuadBank:
    def __init__(self, modules, caches, native_table, native_lengths, prefix_tokens, enabled_batches):
        assert tuple(enabled_batches)==(4,5,6,7,8) and len(caches)==36
        assert all(c.dtype==torch.bfloat16 and c.shape[1:]==(8,16,256) for c in caches)
        assert native_lengths.dtype==torch.int32 and native_lengths.is_contiguous() and native_lengths.numel()>=8
        assert native_table.shape[0]==8 and native_table.dtype==torch.int32 and native_table.is_contiguous()
        self.prefix_tokens=prefix_tokens;self.caches=list(caches);self.enabled_batches=(4,5,6,7,8)
        self.workspace=torch.zeros(64*1024*1024,device=caches[0].device,dtype=torch.uint8)
        sm=torch.cuda.get_device_properties(caches[0].device).multi_processor_count
        extent=native_table.shape[1]*16
        self.capacity_by_batch={b:((b+3)//4)*8*min(max(1,sm//(b*8)),(extent+255)//256)*4*32*35 for b in self.enabled_batches}
        self.raw_state=torch.empty(max(self.capacity_by_batch.values()),device=caches[0].device,dtype=torch.float32)
        self.plans={}
        for b in self.enabled_batches:
            q=torch.empty((b,32,128),device=caches[0].device,dtype=torch.bfloat16)
            p=RawTP1DirectQuadPlan(modules,q,caches[0],native_table[:b],[prefix_tokens+1]*b,prefix_tokens,
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
        assert b in self.plans and query.shape==output.shape==(b,32,128) and query.dtype==output.dtype==torch.bfloat16
        assert query.is_contiguous() and output.is_contiguous(), 'Native Q/O address contract; noncontiguous layouts rejected'
        assert query.device==output.device==self.caches[layer].device
        p=self.plans[b]
        self.query_layouts.setdefault(b,{})[layer]={'shape':list(query.shape),'stride':list(query.stride()),'output_stride':list(output.stride()),'Q_canonicalization_copy_charged':False,'Q_native_alias':True,'O_native_alias':True}
        p.q,p.output=query,output
        p.k,p.v,p.wk,p.wv=self.views[layer]
        return p.run()

    def receipt(self):
        return {'enabled_batches':list(self.enabled_batches),'layers':36,'Hq':32,'Hkv':8,'GQA':4,
                'workspace_allocation_sets':1,'raw_state_allocation_sets':1,'workspace_bytes':self.workspace.numel(),
                'raw_state_bytes':self.raw_state.numel()*4,'query_layouts':self.query_layouts,'capacity_by_batch_floats':self.capacity_by_batch,
                'plans':{b:{'native_batch':b,'dummy_prefix_rows':(4-b%4)%4,'native_table_shape':list(p.table.shape),'native_table_stride':list(p.table.stride()),
                    'native_launch_extent':p.max_len,'native_splits':p.splits,'prefix_SM':p.prefix_sm,
                    'tail_SM':p.tail_sm,'native_SM':p.sm,'prefix_tiles':p.prefix//256} for b,p in self.plans.items()},
                'lengths_policy':'Alias native GPU int32 seq_lens as uint32; no host copy or extra metadata kernel.',
                'copy_policy':'Native contiguous Q/O aliases and direct native table/length addresses. No candidate Q canonicalization, packing, group metadata refresh or O copy launch. Raw checkpoint and original native tail/merge remain charged.',
                'address_contract':'Per-layer concrete Q/O/KV addresses fixed during capture; noncontiguous Q/O rejected; shared workspace/state serial only.',
                'runtime_candidate_PDL':False,'concurrency':'Single stream,shared state/workspace across36layers and5batchgraphs; no concurrent replay.'}
