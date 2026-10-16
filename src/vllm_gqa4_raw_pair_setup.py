"""CPU-only setup and offline-qualification validation for the general adapter."""
from src.vllm_gqa4_raw_pair_contract import REQUIRED_QUALIFICATION_FIELDS as BASE_QUALIFICATION_FIELDS

REQUIRED_QUALIFICATION_FIELDS = BASE_QUALIFICATION_FIELDS + (
    'device_lengths_equal_native_and_positions', 'native_GPU_metadata_aliases_verified',
)


def capture_modes(setup_mode):
    if setup_mode == 'strict':
        return ('witness', 'checked', 'production')
    if setup_mode == 'production_only':
        return ('production',)
    raise ValueError('setup_mode must be strict or production_only')


def candidate_descriptor(desc, full_mode, enabled_batches):
    return (desc.cg_mode == full_mode and desc.num_tokens == desc.num_reqs
            and desc.num_tokens in enabled_batches and desc.uniform_token_count in (None, 1)
            and desc.num_active_loras == 0 and desc.num_ubatches == 1
            and desc.max_query_len in (None, 1))


def validate_offline_records(records, source_sha256, fingerprint, enabled_batches, tp_size):
    """Fail closed on stale code/model/device or missing actual model batch coverage.

    These are research provenance checks, not authentication. Receipts must come
    from completed strict qualification using this exact adapter/source set.
    """
    required = set(enabled_batches)
    covered = {rank: set() for rank in range(tp_size)}
    for record in records:
        rank = record['rank']
        if rank not in covered:
            raise ValueError('Unexpected TP rank in offline receipt')
        if record.get('source_sha256') != source_sha256:
            raise ValueError('Offline source fingerprint mismatch')
        if record.get('execution_fingerprint') != fingerprint:
            raise ValueError('Offline model/device/engine fingerprint mismatch')
        if record.get('setup_mode') != 'strict' or not record.get('qualification_passed'):
            raise ValueError('Offline receipt is not a passing strict model qualification')
        checks = record.get('checks', [])
        if not checks or len(checks) != record.get('qualification_steps'):
            raise ValueError('Offline checks missing or inconsistent')
        for check in checks:
            if not check.get('qualification_passed') or not all(check.get(k) is True for k in REQUIRED_QUALIFICATION_FIELDS):
                raise ValueError('Failed bitwise model check in offline receipt')
            if len(check.get('layer_checks', [])) != 36 or not all(
                    row['layer'] == i and row['Q_bits_equal'] is True and row['O_bits_equal'] is True
                    for i, row in enumerate(check['layer_checks'])):
                raise ValueError('Offline receipt misses an exact Q/O layer check')
            if [m.get('mode') for m in check.get('modes', [])] != ['checked', 'production']:
                raise ValueError('Offline receipt misses one candidate graph mode')
            if not all(v is True for m in check['modes'] for k, v in m.items() if k != 'mode'):
                raise ValueError('Offline candidate mode check failed')
            b = check['batch']
            for mode in ('witness', 'checked', 'production'):
                if record.get('captured_layers_by_batch', {}).get(mode, {}).get(str(b)) != list(range(36)):
                    raise ValueError('Offline receipt misses captured layer coverage for batch')
            covered[rank].add(b)
    if any(not required.issubset(batches) for batches in covered.values()):
        raise ValueError(f'Offline qualification does not cover every enabled batch on every TP rank: {covered}')
    return {'qualified_batches_by_rank': {str(rank): sorted(batches) for rank, batches in covered.items()},
            'records': len(records), 'source_model_device_match': True,
            'scope': 'Observed strict states only; no claim of exhaustive numerical proof.'}
