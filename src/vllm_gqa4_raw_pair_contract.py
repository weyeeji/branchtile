"""Separate36-layer exact qualification fields; never accepts48-layer receipts."""
REQUIRED_QUALIFICATION_FIELDS = (
    'all36_attention_Q_bits_equal', 'all36_attention_O_bits_equal',
    'hidden_bits_equal', 'local_vocab_logits_bits_equal',
    'gathered_logits_bits_equal_where_present', 'all36_current_KV_slot_bits_equal',
    'input_ids_positions_tables_bits_unchanged', 'native_AA_bits_equal',
    'candidate_graph_did_execute', 'same_state_replay_not_new_generated_input',
)


def qualify_same_state_receipt(receipt: dict) -> bool:
    """The driver checks both TP ranks after RPC; never strand one peer locally."""
    return all(receipt.get(field) is True for field in REQUIRED_QUALIFICATION_FIELDS)
