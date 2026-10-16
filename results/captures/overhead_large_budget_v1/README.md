# Natural-capture overlays for large-budget operator experiments

CPU-only derived masks, 2026-10-16. No new model generation, formal evaluation,
GPU measurement, changed Q/K/V, or changed selector scores. Original capture
files are used through symlinks; the generator never writes them. Symlinks do
not enforce immutable filesystem permissions.

| Directory under this folder | B | Full document tokens | 64K | 128K | Full integer CLI argument |
| --- | ---: | ---: | --- | --- | ---: |
| `publicdev6_256/rank{0,1}` | 4 | 173232 | 65536 actual | 131072 actual | 173232 |
| `docfinqa256/rank{0,1}` | 3 | 86784 | 65536 actual | clipped to full 86784 | 86784 |
| `publicdev1_256/rank{0,1}` | 3 | 89736 | 65536 actual | clipped to full 89736 | 89736 |

Every rank contains layers 0/15/31/47 and the unchanged naturally reached
same-forward snapshots at steps 16/64/256 (step 1 is also linked for original
compatibility). `masks_l{layer}_b65536.pt`, `..._b131072.pt`, `..._bfull.pt`, and
`..._b{full_document_tokens}.pt` are real generated files. Numeric full aliases
work with the current integer `--budget` CLI; the string `full` also works when
calling `compare_attention.load_capture` directly. The two full aliases have
identical support tensors. B3 `b131072` is explicitly recorded as clipped;
its filename is not a claim that the shorter document contains 131072 tokens.

Stable frozen suffix-Q score ordering and the original first4/last32 protection
are unchanged. Document budgets count the complete original document. Its
protected partial-page tail remains in each unchanged private suffix and is
omitted only from the shared-source mask. The system prefix and complete
private suffix are retained by the original loader. Under each rank,
`original_metadata.json` links the original metadata; `metadata.json` and
`overlay_receipt.json` are separate generated files with effective budgets,
input/output paths, SHA256 hashes, and preservation checks.

Reproduce and validate without CUDA:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/prepare_overhead_large_budget_captures.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/prepare_overhead_large_budget_captures.py --check
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_overhead_large_budget_captures.py
```

The generator compares all 96 old 4K–32K masks bitwise and exercises the exact
original `load_capture` AST on CPU for 24 large-budget cases, temporarily
replacing only tensor `.cuda()` with identity. This checks shape, prefix,
private-tail and request ownership preservation without importing GPU backends.
It is not numerical GPU qualification. The independent crosscheck compares all
96 new full-document support hashes with the previously generated capacity
audit, including effective-budget identity. Only scripts, metadata and small
receipts are committed; local tensor files and symlinks can be recreated.
