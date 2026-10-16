"""Versioned impact audit of the historical observation-window boundary bug."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['main30b_tp2_docfinqa256','main30b_tp2_publicdev1_256','main30b_tp2_publicdev6_256',
         'main30b_tp2_train1_b8_256','main30b_tp2_train7_b8_256']


def main():
    torch.set_num_threads(4)
    rows = []
    for name in NAMES:
        old_suffix = '_export_retry1' if name == 'main30b_tp2_train1_b8_256' else '_export'
        for rank in [0,1]:
            old = ROOT/f'results/captures/{name}{old_suffix}/rank{rank}'
            new = ROOT/f'results/captures/{name}_suffix_v2/rank{rank}'
            meta = json.loads((new/'metadata.json').read_text())
            assert meta['export_semantics_version'] == 2
            end = meta['document_token_range'][1]
            for layer in meta['layers']:
                for qi in range(meta['B']):
                    file = f'selector_q{qi}_l{layer}.pt'
                    a,b = [torch.load(p/file, map_location='cpu', weights_only=True) for p in [old,new]]
                    assert b['observation_is_question_suffix_only']
                    positions = b['observation_positions']
                    assert int(positions[0]) >= end
                    assert len(positions) == min(32,int(positions[-1])+1-end)
                    assert torch.equal(b['observation_q'], a['observation_q'][:,:,-len(positions):])
                    changed = {str(budget):int((a['supports'][budget] != b['supports'][budget]).sum()) for budget in meta['budgets']}
                    removed = len(a['observation_positions'])-len(positions)
                    if removed == 0:
                        assert torch.equal(a['pooled_scores'],b['pooled_scores'])
                        assert not any(changed.values())
                    rows.append(dict(capture=name,rank=rank,layer=layer,question=qi,
                        removed_document_Q=removed, new_suffix_observations=len(positions),
                        actual_suffix_Q_unchanged=True, changed_mask_entries_by_budget=changed,
                        original_selector_file=str((old/file).relative_to(ROOT)),
                        corrected_selector_file=str((new/file).relative_to(ROOT)),
                        original_sha256=hashlib.sha256((old/file).read_bytes()).hexdigest(),
                        corrected_sha256=hashlib.sha256((new/file).read_bytes()).hexdigest()))
    result = dict(scope='Boundary correction only, same actual Q/KV and question tokens; old artifacts preserved',
        selector_rows=len(rows), rows_with_document_Q_removed=sum(r['removed_document_Q']>0 for r in rows),
        corrected_new_Q_matches_original_suffix_exactly=all(r['actual_suffix_Q_unchanged'] for r in rows),
        all_already_suffix_only_scores_and_masks_unchanged=True, rows=rows)
    (ROOT/'results/selector_suffix_correction_impact.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'},indent=2))


if __name__ == '__main__':
    main()
