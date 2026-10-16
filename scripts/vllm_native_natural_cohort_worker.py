"""CPU-only step records around unchanged native input preparation and graphs."""
import hashlib
import json
from pathlib import Path
import time

from scripts.vllm_native_precision_worker import NativePrecisionWorkerExtension


class NativeNaturalCohortWorker(NativePrecisionWorkerExtension):
    def configure_native_cohort_probe(self,config):
        result=self.configure_native_precision_probe(config)
        self.cohort_steps=[]
        runner=self.model_runner
        previous=runner.prepare_inputs
        def prepare(*args,**kwargs):
            started=time.perf_counter()
            batch=previous(*args,**kwargs)
            state=self.native_probe
            b=int(batch.num_reqs)
            questions=[state['requests'].get(rid) for rid in batch.req_ids[:b]]
            assert all(q is not None for q in questions)
            self.cohort_steps.append({'phase':state['phase'],'batch':b,
                'questions':questions,'positions':batch.num_computed_tokens_np[:b].tolist(),
                'num_scheduled_tokens':batch.num_scheduled_tokens[:b].tolist(),
                'is_prefilling':batch.is_prefilling_np[:b].tolist(),'has_prefill':bool(batch.has_prefill),
                'decode_graph_eligible':bool(batch.decode_graph_eligible),
                'num_tokens':int(batch.num_tokens),'num_tokens_after_padding':int(batch.num_tokens_after_padding),
                'prepare_inputs_started_monotonic_seconds':started,
                'prepare_inputs_finished_monotonic_seconds':time.perf_counter()})
            return batch
        runner.prepare_inputs=prepare
        result['cohort_worker_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    def set_native_cohort_phase(self,phase):
        self.cohort_steps=[]
        return self.set_native_phase(phase)

    def finish_native_cohort_phase(self):
        result=self.finish_native_phase()
        path=self.native_probe['output']/f'{self.native_probe["phase"]}_all_steps.json'
        path.write_text(json.dumps({'rank':self.native_probe['rank'],'phase':self.native_probe['phase'],
            'steps':self.cohort_steps,'scope':'CPU scheduler/input metadata for all prepared batches, including prefill and recomputation. No CUDA events or model alteration.'},indent=2)+'\n')
        result['all_steps_metadata']=str(path)
        result['all_steps_count']=len(self.cohort_steps)
        return result
