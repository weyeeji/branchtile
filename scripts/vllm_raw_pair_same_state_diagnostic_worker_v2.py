"""Independent late-state diagnosis on the original native sampler trajectory.

Full-model same-state checks plus per-layer identical-Q native AA/PDL controls.
Does not authorize production or alter any frozen kernel/worker source.
"""
import hashlib
import json
from pathlib import Path
import torch
from scripts.vllm_native_raw_pair_general_policy_worker import NativeRawPairGeneralPolicyWorker,bits_equal


class RawPairSameStateDiagnosticWorker(NativeRawPairGeneralPolicyWorker):
    def configure_dense_cascade(self, config):
        result=super().configure_dense_cascade(config)
        self.same_state_diag={'start':int(config['diagnostic_start_graph']),'stop':int(config['diagnostic_stop_graph']),
                              'records':[],'snapshots':[],'workspace':torch.zeros(64*1024*1024,device=self.model_runner.device,dtype=torch.uint8)}
        return result

    @torch.inference_mode()
    def _qualify_raw_pair_state(self, desc, native_hidden):
        probe,state,diag=self.native_probe,self.raw_pair,self.same_state_diag
        graph_index=len(probe['records'])-1
        if not diag['start']<=graph_index<=diag['stop']:return
        # The frozen routine restores original native KV slots before returning.
        super()._qualify_raw_pair_state(desc,native_hidden)
        model_check=state['checks'][-1]
        from flashinfer.xqa import xqa
        b=desc.num_tokens;p=state['plan'].plans[b];nw=diag['workspace']
        record={'graph_index':graph_index,'batch':b,'S':p.splits,'prefix':p.prefix_tokens,
                'positions':state['pending']['positions'],'questions':probe['records'][-1]['questions'],
                'full_model_qualification_passed':model_check['qualification_passed'],
                'native_model_AA_bits_equal':model_check['native_AA_bits_equal'],'layers':[]}
        for layer in range(48):
            q=state['saved_q'][layer][:b].contiguous()
            witness_o=state['saved_o'][layer][:b]
            cache=state['caches'][layer]
            native_o,aa_o,off_o,candidate_o,candidate2=[torch.empty_like(q) for _ in range(5)]
            def native(output,pdl):
                xqa(q.unsqueeze(1),cache[...,:128],cache[...,128:],p.table,p.lengths,output.unsqueeze(1),
                    nw[8*1024*1024:],nw[:8*1024*1024],2,16,q_scale=1.,kv_scale=1.,kv_layout='HND',sm_count=p.sm,enable_pdl=pdl)
            native(native_o,True);native(aa_o,True);native(off_o,False)
            state['plan'].run_layer(layer,q,candidate_o)
            paired_state=p.state.clone() if not bits_equal(native_o,candidate_o) else None
            p.state.fill_(float('nan'))
            state['plan'].run_layer(layer,q,candidate2)
            checks={'layer':layer,'full_model_layer_Q_bits_equal':model_check['layer_checks'][layer]['Q_bits_equal'],
                    'full_model_layer_O_bits_equal':model_check['layer_checks'][layer]['O_bits_equal'],
                    'native_AA_bits_equal':bits_equal(native_o,aa_o),
                    'native_PDL_on_off_bits_equal':bits_equal(native_o,off_o),
                    'native_public_vs_witness_O_bits_equal':bits_equal(native_o,witness_o),
                    'candidate_same_Q_bits_equal':bits_equal(native_o,candidate_o),
                    'candidate_poison_replay_bits_equal':bits_equal(candidate_o,candidate2),
                    'candidate_max_abs':float((native_o.float()-candidate_o.float()).abs().max()),
                    'candidate_bit_mismatch_count':int((native_o.view(torch.int16)!=candidate_o.view(torch.int16)).sum())}
            record['layers'].append(checks)
            failed=not all(v for k,v in checks.items() if k.endswith('bits_equal'))
            if failed and not diag['snapshots']:
                snapshot=self._save_same_state_snapshot(layer,q,witness_o,native_o,aa_o,off_o,candidate_o,candidate2,
                                                        p,cache,paired_state,record,checks)
                diag['snapshots'].append(snapshot)
        diag['records'].append(record)
        self._save_same_state_report()

    def _save_same_state_snapshot(self,layer,q,witness,native,aa,off,candidate,candidate2,p,cache,paired_state,record,checks):
        b=len(q);lengths=p.lengths.flatten().cpu().long()
        table=p.table.cpu().long();counts=(lengths+15)//16
        used=torch.unique(torch.cat([table[i,:int(n)] for i,n in enumerate(counts)]),sorted=True)
        mapping={int(old):new for new,old in enumerate(used)}
        compact_table=torch.zeros_like(table)
        for i,n in enumerate(counts):compact_table[i,:int(n)]=torch.tensor([mapping[int(x)] for x in table[i,:int(n)]])
        compact_cache=cache.index_select(0,used.to(device=cache.device)).cpu()
        # Capture independent native-GQA8 checkpoint for raw fragment/hint comparison.
        raw_native=torch.empty(b*2*p.splits*4*32*35,device=q.device,dtype=torch.float32)
        unused=torch.empty_like(q);nw=self.same_state_diag['workspace']
        p.modules[8].xqa_wrapper(False,p.splits*b*2,2,0,1.,None,unused.unsqueeze(1),1.,q.unsqueeze(1),
            None,cache[...,:128].transpose(1,2),cache[...,128:].transpose(1,2),None,None,p.table,p.max_len,
            p.lengths,b,1.,None,1,None,None,nw[:8*1024*1024],nw[8*1024*1024:],False,raw_native,p.prefix_tokens//256,1)
        unshared=torch.empty_like(q)
        p.modules[8].xqa_wrapper(False,p.splits*b*2,2,0,1.,None,unshared.unsqueeze(1),1.,q.unsqueeze(1),
            None,cache[...,:128].transpose(1,2),cache[...,128:].transpose(1,2),None,None,p.table,p.max_len,
            p.lengths,b,1.,None,1,None,None,nw[:8*1024*1024],nw[8*1024*1024:],False,raw_native,p.prefix_tokens//256,2)
        path=self.native_probe['output']/f'same_state_first_failure_g{record["graph_index"]}_l{layer}.pt'
        torch.save({'schema':'native_raw_pair_same_state_v1','rank':self.native_probe['rank'],'layer':layer,
            'graph_index':record['graph_index'],'questions':record['questions'],'positions':record['positions'],
            'q':q.cpu(),'model_candidate_q':self.raw_pair['check_q'][layer][:b].cpu(),
            'model_candidate_o':self.raw_pair['check_o'][layer][:b].cpu(),'witness_o':witness.cpu(),'native_o':native.cpu(),'native_aa_o':aa.cpu(),'native_pdl_off_o':off.cpu(),
            'candidate_o':candidate.cpu(),'candidate_poison_o':candidate2.cpu(),'unshared_checkpoint_o':unshared.cpu(),
            'cache':compact_cache,'block_tables':compact_table,'seq_lens':lengths,'original_used_physical_pages':used,
            'original_cache_shape':list(cache.shape),'original_cache_stride':list(cache.stride()),'original_table_columns':p.table.shape[1],
            'prefix_tokens':p.prefix_tokens,'S':p.splits,'SM':p.sm,'k_scale':1.,'v_scale':1.,
            'paired_prefix_raw_state':(paired_state if paired_state is not None else p.state).cpu(),
            'native_prefix_raw_state':raw_native.cpu(),'checks':checks},path)
        return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'layer':layer,'graph_index':record['graph_index'],'unshared_checkpoint_bits_equal':bits_equal(native,unshared)}

    def _save_same_state_report(self):
        diag=self.same_state_diag
        path=self.native_probe['output']/'same_state_diagnostic.json'
        result={'rank':self.native_probe['rank'],'window':[diag['start'],diag['stop']],
                'records':diag['records'],'snapshots':diag['snapshots'],
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope':'Original native sampler trajectory. Window same-state fullmodel plus identical-Q perlayer controls; not performance.'}
        path.write_text(json.dumps(result,indent=2)+'\n')
        return {'path':str(path),'rank':self.native_probe['rank'],'diagnosed_states':len(diag['records']),
                'full_model_failures':sum(not x['full_model_qualification_passed'] for x in diag['records']),
                'same_Q_candidate_failures':sum(not l['candidate_same_Q_bits_equal'] for x in diag['records'] for l in x['layers']),
                'native_AA_failures':sum(not l['native_AA_bits_equal'] for x in diag['records'] for l in x['layers']),
                'snapshots':diag['snapshots']}

    def finish_same_state_diagnostic(self):
        return self._save_same_state_report()
