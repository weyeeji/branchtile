"""Frozen mask variants sharing exactly one native E0 graph and plan buffers.

Installing equal-length histories uses the existing explicit plan API. This
is an engineering control, not physical KV release or a new executor.
"""
import copy
import hashlib
import json
import re
import time

import torch

from scripts.vllm_sparse_production_worker import ProductionSparseWorkerExtension
from src.joint_support_exchange import JointSupportExchangeConfig


class JointSupportWorkerExtension(ProductionSparseWorkerExtension):
    def freeze_joint_variants(self, specifications):
        state=self.sparse_state
        assert not state['graph_enabled']
        base=state['selector'];expected={(l,q) for l in range(48) for q in range(len(base.prompts))}
        assert set(base.masks)==set(base.joint_pending_scores)==expected
        assert base.joint_report is None
        assert specifications[0]=={'name':'independent','configuration':None}
        names=[s['name'] for s in specifications]
        assert len(names)==len(set(names)) and all(re.fullmatch('[a-z0-9_]+',n) for n in names)
        started=time.perf_counter();variants={};reports={}
        for spec in specifications:
            selector=copy.copy(base)
            selector.joint_report=None
            if spec['configuration'] is None:
                assert spec['name']=='independent'
                selector.joint_config=None;selector.joint_pending_scores={}
                reports[spec['name']]={'enabled':False,'mask_identity':selector.mask_identity()}
            else:
                selector.joint_config=JointSupportExchangeConfig.from_mapping(spec['configuration'])
                reports[spec['name']]=selector.finalize_joint_supports(48,2)
            variants[spec['name']]=selector
        # Every variant was computed from the same untouched independent IDs.
        original=base.mask_identity()['sha256']
        assert all(not r['enabled'] or r['independent_mask_identity']['sha256']==original for r in reports.values())
        base.joint_pending_scores={}
        state['joint_variants']=variants;state['joint_active_variant']='independent'
        state['selector']=variants['independent'];state['joint_install_records']=[]
        result={'rank':state['rank'],'all_variants_frozen_before_any_candidate_graph':True,
            'specifications':specifications,'independent_mask_identity':original,
            'freeze_all_variants_wall_seconds':time.perf_counter()-started,'variants':reports,
            'shared_score_D2H_bytes':sum(r['GPU_to_CPU_bytes'] for r in base.joint_transfers),
            'shared_score_transfer_and_copy_wall_seconds':sum(r['transfer_and_owned_CPU_copy_wall_seconds'] for r in base.joint_transfers),
            'cost_accounting':'Score D2H occurs once in selector freeze; per-variant reports reference this shared cost. Add each CPU finalization once.'}
        (state['output']/'joint_variants.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    @torch.inference_mode()
    def install_joint_variant(self, name):
        state=self.sparse_state;started=time.perf_counter()
        assert name in state['joint_variants'] and state['graph_enabled']
        assert set(state['graph_managers'])=={'e0'},'Variant installation supports only native E0'
        selector=state['joint_variants'][name];selector.require_finalized_joint_supports()
        prompts=[len(x) for x in selector.prompts]
        histories=[[[torch.cat((torch.arange(selector.doc_start),selector.masks[l,q][h],
            torch.arange(selector.doc_end,prompts[q]))) for h in range(2)] for q in range(len(prompts))] for l in range(48)]
        identity=hashlib.sha256(json.dumps({'prompt_hashes':state['config']['prompt_token_hashes'],
            'active_masks':selector.mask_identity()['sha256']},sort_keys=True).encode()).hexdigest()
        records=[]
        for (method,b),plan in sorted(state['shared_plans'].items()):
            assert method=='e0'
            before=[t.data_ptr() for t in (plan.histories,plan.indices,plan.indptr,plan.outputs,plan.workspace)]
            previous_id=plan.history_id
            plan.set_histories(histories,identity)
            after=[t.data_ptr() for t in (plan.histories,plan.indices,plan.indptr,plan.outputs,plan.workspace)]
            assert before==after and not plan.prepared
            records.append({'batch':b,'previous_history_id':previous_id,'new_history_id':identity,
                'history_payload_sha256':plan.history_payload_sha256,'all_capture_buffer_addresses_unchanged':True,
                'history_H2D_bytes':plan.histories.numel()*plan.histories.element_size(),
                'prepared_false_until_next_checked_native_plan':True})
        torch.cuda.synchronize()
        previous=state['joint_active_variant'];state['selector']=selector
        state['joint_active_variant']=name;state['shared_history_id']=identity
        state['graph_pending']=None
        result={'rank':state['rank'],'from_variant':previous,'to_variant':name,
            'install_complete_wall_seconds':time.perf_counter()-started,'plans':records,
            'active_mask_identity':selector.mask_identity()['sha256'],
            'same_graph_manager_and_capture_count':True,'graph_capture_calls':dict(state['graph_capture_calls']),
            'scope':'Equal-length logical history installation only; no KV payload copy/release or graph recapture; normal prepare still validates history identity and topology.'}
        state['joint_install_records'].append(result)
        return result

    def set_joint_phase(self, phase, variant, teacher_tokens=None):
        if variant is None:
            return self.set_sparse_phase(phase,'dense',teacher_tokens)
        installed=self.install_joint_variant(variant)
        return {'installation':installed,**self.set_sparse_phase(phase,'e0',teacher_tokens)}
