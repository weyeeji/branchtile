"""One grouped dynamic-cut graph bank across sequential document cohorts."""
import time
import torch
from src.vllm_native_raw_pair_general_plan import LayerSharedRawPairGeneralBank
from src.xqa_raw_pair_dynamic_cut_runtime import RawPairDynamicCutPlan

class LayerSharedRawPairMultiDocumentBank(LayerSharedRawPairGeneralBank):
    def __init__(self,modules,caches,native_table,native_lengths,prefix_tokens,enabled_batches):
        self.device_cut=torch.tensor([prefix_tokens//256],device=caches[0].device,dtype=torch.int32)
        self.uploaded_prefix_tokens=prefix_tokens
        def factory(*args,**kwargs):return RawPairDynamicCutPlan(*args,device_cut=self.device_cut,**kwargs)
        super().__init__(modules,caches,native_table,native_lengths,prefix_tokens,enabled_batches,plan_factory=factory)
        assert all(p.device_cut.data_ptr()==self.device_cut.data_ptr() for p in self.plans.values())
        self.switches=[]

    def activate_prefix(self,prefix_tokens,prepare_candidate):
        """Only between fully completed groups; never copy native live lengths/table.

        The caller discards ownership and binds current physical pages after the
        next document's prefill. Native-only activations do no device upload.
        """
        assert isinstance(prefix_tokens,int) and prefix_tokens>0 and prefix_tokens%16==0
        assert all(prefix_tokens<plan.max_len for plan in self.plans.values())
        assert not torch.cuda.is_current_stream_capturing()
        tick=time.perf_counter();sync=0.;upload=0.
        if prepare_candidate:
            t=time.perf_counter();torch.cuda.synchronize();sync=time.perf_counter()-t
            t=time.perf_counter();self.device_cut.fill_(prefix_tokens//256);torch.cuda.synchronize();upload=time.perf_counter()-t
            self.uploaded_prefix_tokens=prefix_tokens
        self.prefix_tokens=prefix_tokens
        for p in self.plans.values():p.prefix_tokens=prefix_tokens
        result={'nominal_prefix_tokens':prefix_tokens,'cut_tiles':prefix_tokens//256,'candidate_cut_uploaded':bool(prepare_candidate),
                'prior_work_synchronize_seconds':sync,'cut_upload_and_complete_seconds':upload,
                'complete_bank_activation_seconds':time.perf_counter()-tick,'native_lengths_and_tables_untouched':True,
                'shared_cut_ptr':self.device_cut.data_ptr()}
        self.switches.append(result);return result

    def receipt(self):
        result=super().receipt()
        result.update(dynamic_cut=True,shared_cut_ptr=self.device_cut.data_ptr(),device_cut_shape=list(self.device_cut.shape),
                      uploaded_prefix_tokens=self.uploaded_prefix_tokens,nominal_active_prefix_tokens=self.prefix_tokens,
                      document_switches=self.switches,cut_scope='One stable device int32 address across all batch/layer graphs; uploaded only outside replay on a synchronized stream.')
        return result
