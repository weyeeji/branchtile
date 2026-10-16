"""Native-only finite-queue admission and scheduling observations, without CUDA edits."""
import json
import time
from scripts.vllm_native_precision_worker import NativePrecisionWorkerExtension


class NativeSelfConsistencyWorker(NativePrecisionWorkerExtension):
    def configure_self_consistency_probe(self, config):
        receipt = self.configure_native_precision_probe(config)
        state = self.native_probe
        state['sample_admissions'] = []
        state['all_steps'] = []
        runner = self.model_runner
        original_add = runner.add_requests
        def add(scheduler_output):
            result = original_add(scheduler_output)
            for request in scheduler_output.scheduled_new_reqs:
                state['sample_admissions'].append({
                    'phase': state['phase'], 'request_id': request.req_id,
                    'question_index': state['requests'][request.req_id],
                    'seed': request.sampling_params.seed,
                    'admitted_monotonic_seconds': time.perf_counter()})
            return result
        runner.add_requests = add
        original_prepare = runner.prepare_inputs
        def prepare(*args, **kwargs):
            started = time.perf_counter()
            batch = original_prepare(*args, **kwargs)
            b = batch.num_reqs
            state['all_steps'].append({'phase': state['phase'],
                'step_started_monotonic_seconds': started, 'batch': b,
                'num_tokens': batch.num_tokens, 'padded_tokens': batch.num_tokens_after_padding,
                'request_ids': list(batch.req_ids[:b]),
                'positions': batch.num_computed_tokens_np[:b].tolist(),
                'pure_single_token_decode_shape': b == batch.num_tokens})
            return batch
        runner.prepare_inputs = prepare
        return receipt

    def finish_self_consistency_phase(self):
        state = self.native_probe
        original = self.finish_native_phase()
        path = state['output'] / (state['phase'] + '_queue.json')
        rows = {key: [r for r in state[key] if r['phase'] == state['phase']]
                for key in ('sample_admissions', 'all_steps')}
        rows.update(rank=state['rank'], phase=state['phase'],
                    scope='Host scheduling intervals, not isolated GPU kernel timing. Native execution unchanged.')
        path.write_text(json.dumps(rows, indent=2) + '\n')
        return dict(original, queue_metadata=str(path))
