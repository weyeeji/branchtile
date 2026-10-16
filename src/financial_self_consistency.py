"""Gold-free finite-sample voting; preserves invalid samples in the denominator."""
from collections import Counter
from decimal import Decimal
import hashlib
import json

from src.financial_answer_metric import ParseError, extract_final, parse_value


def sample_seed(document_id, question_hash, sample_index, study_seed=20261017):
    value = json.dumps([document_id, question_hash, sample_index, study_seed], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'big') % (2**31)


def vote(final_responses):
    """No gold input; exact Decimal equality and earliest-sample tie breaking."""
    parsed, keys, errors = [], [], []
    for response in final_responses:
        try:
            item = parse_value(extract_final(response))
            value = Decimal(item.value) if item.kind == 'numeric' else item.value
            parsed.append(item)
            keys.append((item.kind, item.dimension, value))
            errors.append(None)
        except ParseError as exc:
            parsed.append(None); keys.append(None); errors.append(str(exc))
    counts = Counter(key for key in keys if key is not None)
    if not counts:
        return {'prediction': '', 'selected_sample': None, 'count': 0,
                'denominator': len(keys), 'valid_samples': 0, 'top_two_margin': 0,
                'strict_majority': False, 'parse_errors': errors}
    winner = max(counts, key=lambda key: (counts[key], -keys.index(key)))
    index = keys.index(winner)
    second = max((v for k, v in counts.items() if k != winner), default=0)
    return {'prediction': 'FINAL: ' + parsed[index].raw, 'selected_sample': index,
            'count': counts[winner], 'denominator': len(keys),
            'valid_samples': sum(key is not None for key in keys),
            'top_two_margin': counts[winner] - second,
            'strict_majority': counts[winner] > len(keys) / 2, 'parse_errors': errors}
