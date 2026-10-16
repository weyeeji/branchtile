"""Explicit training-shape transfer, separate from document-local qualification."""
import json
from pathlib import Path
from src.raw_pair_direct_multidoc_offline import validate_document_records,load_completed_qualification
ROOT=Path(__file__).resolve().parents[1]
TRAINING_PROTOCOL=ROOT/'configs/raw_pair_multidoc_training_v1.json'


def validate_training_shapes(records,training_protocol,strict_sources,fingerprint,enabled_batches):
    if training_protocol!=json.loads(TRAINING_PROTOCOL.read_text()):raise ValueError('Training protocol must be the frozen original two documents')
    documents=training_protocol['documents']
    if [d['document_id'] for d in documents]!=['docfinqa-c1647beac4edf6d2','docfinqa-1d5e740592615e73']:
        raise ValueError('Unexpected training document selection')
    validation=validate_document_records(records,documents,strict_sources,fingerprint,training_protocol['candidate_enabled_batches'])
    covered={rank:set() for rank in (0,1)}
    for record in records:covered[record['rank']].update(c['batch'] for c in record['checks'])
    jointly=sorted(covered[0]&covered[1]&set(enabled_batches))
    if not jointly:raise ValueError('No jointly qualified training shape applies to this bank')
    return {'qualification_kind':'explicit_training_shape_transfer','evaluation_document_locally_qualified':False,
            'training_document_ids':[d['document_id'] for d in documents],
            'training_document_audits':validation,'training_shapes_by_rank':{str(k):sorted(v) for k,v in covered.items()},
            'eligible_training_qualified_shapes':jointly,'unqualified_shapes_fallback_native':sorted(set(enabled_batches)-set(jointly)),
            'scope':'New-document output-equivalence experiment backed by prior training shapes and live ownership checks; not new-document per-layer strict qualification.'}


def load_training_shapes(summary_path,strict_sources,enabled_batches):
    protocol=json.loads(TRAINING_PROTOCOL.read_text())
    loaded=load_completed_qualification(summary_path,protocol,ROOT,strict_sources)
    contract=validate_training_shapes(loaded['records'],protocol,strict_sources,loaded['execution_fingerprint'],enabled_batches)
    return {'training_protocol':protocol,'contract':contract,'loaded':loaded}
