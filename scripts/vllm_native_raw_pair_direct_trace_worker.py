"""CPU observers and profiler RPCs only; inherits the qualified direct execution path."""
from pathlib import Path
import torch
from scripts.vllm_native_raw_pair_direct_worker import NativeRawPairDirectWorker


class NativeRawPairDirectTraceWorker(NativeRawPairDirectWorker):
    def install_direct_capture_observer(self):
        state=self.raw_pair
        assert state['setup_mode']=='production_only' and not state['managers']
        assert not hasattr(self,'direct_capture_observations')
        self.direct_capture_observations=[]
        bank=state['plan'];original=bank.run_layer
        def observed(layer,query,output):
            self.direct_capture_observations.append({'capturing':torch.cuda.is_current_stream_capturing(),
                'mode':state['mode'],'batch':query.shape[0],'layer':layer,
                'Q_ptr':query.data_ptr(),'O_ptr':output.data_ptr(),'KV_ptr':bank.caches[layer].data_ptr(),
                'Q_shape':list(query.shape),'Q_stride':list(query.stride()),'O_stride':list(output.stride())})
            return original(layer,query,output)
        bank.run_layer=observed
        return {'rank':self.native_probe['rank'],'scope':'Host metadata observer only; invokes original bank exactly once, no tensor copy/allocation/mutation.'}

    def direct_capture_address_receipt(self):
        observations=self.direct_capture_observations
        actual=[x for x in observations if x['capturing']]
        for b in self.raw_pair['enabled_batches']:
            rows=[x for x in actual if x['batch']==b]
            assert {x['layer'] for x in rows}==set(range(48)) and all(x['mode']=='production' for x in rows)
        return {'rank':self.native_probe['rank'],'observations':observations,
                'scope':'Concrete arguments selected during capture; graph node parameter decoding is not performed.'}

    def start_direct_profile(self,tag):
        assert self.raw_pair['mode'] is None and not self.raw_pair['armed']
        assert getattr(self,'direct_profiler',None) is None
        assert tag.replace('_','').isalnum()
        torch.cuda.synchronize()
        self.direct_profile_path=self.native_probe['output']/f'{tag}.trace.json'
        assert not self.direct_profile_path.exists()
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False)
        self.direct_profiler=profiler;profiler.start()
        return {'rank':self.native_probe['rank'],'trace':str(self.direct_profile_path)}

    def stop_direct_profile(self):
        assert self.direct_profiler is not None
        torch.cuda.synchronize();self.direct_profiler.stop()
        self.direct_profiler.export_chrome_trace(str(self.direct_profile_path))
        self.direct_profiler=None
        return {'rank':self.native_probe['rank'],'trace':str(self.direct_profile_path),
                'scope':'Profiler perturbs execution. No latency or speedup claim from this run.'}
