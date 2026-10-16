"""Existing FA2 cascade with one-time scheduling and device-resident tail metadata.

No future-token KV padding. Root-controlled experiments must check device flags
at phase boundaries; an invalid phase is never a qualified output/performance run.
"""
import time
import torch
import triton
import triton.language as tl
from src.vllm_dense_cascade_plan import LayerSharedDenseCascadePlan


@triton.jit
def _update_tail(Table, Position, Indptr, Indices, Last, Flags,
                 TABLE_STRIDE:tl.constexpr,TABLE_COLS:tl.constexpr,B:tl.constexpr,
                 PREFIX:tl.constexpr,MAX_TOKENS:tl.constexpr,MAX_PAGES:tl.constexpr,
                 TILES:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    row=pid//TILES
    lane=tl.arange(0,R)
    tails=tl.load(Position+lane,lane<B,other=PREFIX)+1-PREFIX
    allowed=tl.minimum(MAX_TOKENS,(TABLE_COLS-PREFIX//16)*16)
    bad=(tails<=0)|(tails>allowed)
    safe=tl.minimum(tl.maximum(tails,1),allowed)
    counts=tl.where(lane<B,(safe+15)//16,0).to(tl.int32)
    offset=tl.sum(tl.where(lane<row,counts,0),0)
    row_count=tl.sum(tl.where(lane==row,counts,0),0)
    if pid==0:
        tl.store(Indptr,0)
        tl.store(Indptr+lane+1,tl.cumsum(counts),lane<B)
        tl.store(Last+lane,((safe-1)%16+1).to(tl.int32),lane<B)
        if tl.sum(((lane<B)&bad).to(tl.int32),0)>0:
            tl.atomic_or(Flags,1)
    page=(pid%TILES)*BLOCK+tl.arange(0,BLOCK)
    valid=(page<row_count)&(page<MAX_PAGES)
    physical=tl.load(Table+row*TABLE_STRIDE+PREFIX//16+page,valid,other=0)
    tl.store(Indices+offset+page,physical,valid)


@triton.jit
def _verify_prefix(Table,PrefixIndices,Flags,TABLE_STRIDE:tl.constexpr,
                   PREFIX_PAGES:tl.constexpr,TILES:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    row=pid//TILES
    page=(pid%TILES)*BLOCK+tl.arange(0,BLOCK)
    mask=page<PREFIX_PAGES
    actual=tl.load(Table+row*TABLE_STRIDE+page,mask,other=0)
    bound=tl.load(PrefixIndices+page,mask,other=0)
    if tl.sum((mask&(actual!=bound)).to(tl.int32),0)>0:
        tl.atomic_or(Flags,2)


class LayerSharedDeviceDenseCascadePlan(LayerSharedDenseCascadePlan):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.device_flags=torch.zeros(1,device=self.device,dtype=torch.int32)
        self.prefix_bound=False
        self.device_prepare_count=0
        self.bound_epoch=0
        self.initial_tail_plan_info=tuple(self.tail_wrapper._plan_info)

    def bind_prefix(self,block_tables):
        """Epoch/admission-only full physical check and copy; never per-token."""
        started=time.perf_counter()
        shared=block_tables[:self.batch,:self.prefix_pages]
        assert shared.shape==(self.batch,self.prefix_pages)
        if not bool((shared==shared[:1]).all().item()):
            raise ValueError('Cannot bind a physically divergent shared prefix')
        self.prefix_indices.copy_(shared[0])
        self.prefix_bound=True
        self.bound_epoch+=1
        self.device_flags.zero_()
        return {'epoch':self.bound_epoch,'physical_prefix_checked':True,
                'prefix_pages':self.prefix_pages,'host_elapsed_seconds':time.perf_counter()-started,
                'scope':'One-time cohort binding; this cost must be charged.'}

    def prepare_device(self,block_tables,positions,verify_prefix=True):
        """Capture this update once before the first layer; all48 layers reuse it."""
        assert self.prefix_bound
        assert positions.is_cuda and positions.numel()>=self.batch
        assert block_tables.is_cuda and block_tables.shape[0]>=self.batch
        assert block_tables.shape[1]>=self.prefix_pages+1 and block_tables.stride(1)==1
        tiles=triton.cdiv(self.max_tail_pages,256)
        _update_tail[(self.batch*tiles,)](block_tables,positions,self.tail_indptr,
            self.tail_indices,self.tail_last,self.device_flags,
            TABLE_STRIDE=block_tables.stride(0),TABLE_COLS=block_tables.shape[1],
            B=self.batch,PREFIX=self.common_prefix_tokens,MAX_TOKENS=self.max_tail_tokens,
            MAX_PAGES=self.max_tail_pages,TILES=tiles,R=triton.next_power_of_2(self.batch),BLOCK=256)
        if verify_prefix:
            prefix_tiles=triton.cdiv(self.prefix_pages,256)
            _verify_prefix[(self.batch*prefix_tiles,)](block_tables,self.prefix_indices,self.device_flags,
                TABLE_STRIDE=block_tables.stride(0),PREFIX_PAGES=self.prefix_pages,TILES=prefix_tiles,BLOCK=256)
        # No Tensor.item(), CPU sequence lengths, .plan() or host copy here.
        self.device_prepare_count+=1
        self.prepared=True

    def device_error_receipt(self):
        """Phase-boundary synchronization; any nonzero flag rejects the phase."""
        flags=int(self.device_flags.item())
        return {'flags':flags,'tail_bounds_error':bool(flags&1),'prefix_identity_error':bool(flags&2),
                'valid':flags==0,'epoch':self.bound_epoch,'tail_plan_info_unchanged':tuple(self.tail_wrapper._plan_info)==self.initial_tail_plan_info,
                'host_plan_calls_after_initialization':0,
                'scope':'Device flags accumulate across graph replays and are read only at an explicit validation/phase boundary.'}
