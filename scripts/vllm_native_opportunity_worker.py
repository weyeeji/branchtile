"""Native FI opportunity instrumentation; enabled only in a separate profile run."""
import hashlib
import json
from pathlib import Path

from scripts.vllm_native_precision_worker import NativePrecisionWorkerExtension


class NativeOpportunityWorkerExtension(NativePrecisionWorkerExtension):
    def configure_opportunity_probe(self, config):
        result = self.configure_native_precision_probe(config)
        self.opportunity_events = []
        self.opportunity_enabled = False
        previous = self.model_runner.cudagraph_manager.run_fullgraph

        def measured(desc):
            if not self.opportunity_enabled:
                return previous(desc)
            import torch
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            batch = self.native_probe['batch']
            row = {'batch': int(batch.num_reqs),
                   'positions': batch.num_computed_tokens_np[:batch.num_reqs].tolist(),
                   'graph_descriptor': str(desc), 'begin': begin, 'end': end}
            begin.record()
            result = previous(desc)
            end.record()
            self.opportunity_events.append(row)
            return result
        self.model_runner.cudagraph_manager.run_fullgraph = measured
        result['opportunity_worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    def set_opportunity_instrumentation(self, enabled):
        self.opportunity_enabled = bool(enabled)
        if enabled:
            self.opportunity_events = []
        return {'rank': self.native_probe['rank'], 'enabled': self.opportunity_enabled}

    def finish_opportunity_instrumentation(self):
        import torch
        self.opportunity_enabled = False
        torch.cuda.synchronize()
        rows = [{k: v for k, v in row.items() if k not in ('begin', 'end')} |
                {'model_graph_cuda_us': row['begin'].elapsed_time(row['end']) * 1000}
                for row in self.opportunity_events]
        result = {'rank': self.native_probe['rank'], 'events': rows,
                  'scope': 'Instrumented actual native whole-model graph only, excluding engine preparation/sampling; not unprofiled engine latency.'}
        path = self.native_probe['output'] / 'opportunity_graph_events.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        return result
