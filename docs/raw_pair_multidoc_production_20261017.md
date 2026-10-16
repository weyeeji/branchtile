# Production-only deployment measurement for the shared multi-document bank

This is a separate worker and driver. The strict multi-document qualification files remain frozen. `benchmark_raw_pair_multidoc_production.py` captures exactly one production graph bank for B6..21 and reuses it across the same two documents. It does not capture witness/checked graphs or run their shadows. Native and candidate both reset APC, execute the same cold anchor, and use the same completed-group document order under ABBA.

The driver requires a completed strict multi-document summary for the exact protocol. It rechecks the strict driver's current source hashes, each rank receipt's digest against the completed audit, exact document IDs and prefixes, both ranks' model/device fingerprints, and every required same-state numerical field. The runtime worker repeats validation against its actual model/device. Every document must have at least one jointly qualified batch; an entirely unqualified document is rejected. Only the intersection of observed passing B values for that document may dispatch the candidate. Other B values fall back to native; another document's coverage is never borrowed. All natural output token arrays must also match that document's original offline native trajectory.

The separate production worker changes setup and receipt validation only. The CPU checker compares its attention-hook configuration tail, activation, graph capture, qualification and export methods byte-for-byte with the frozen strict worker. The production phase API rejects validation and its qualification arming API rejects the production setup mode. New production source fingerprints and the accepted strict source fingerprints remain explicit. This is provenance validation, not cryptographic authentication.

Per-document switch/upload/synchronization, APC reset, anchor, graph control, natural group completion and export remain timed. Worker UUID-verified NVML counters and sampling overhead use the same boundaries on both arms. Initialization, offline preflight, candidate configuration and one production capture are separately recorded. The input preparation timer includes local tokenizer loading. The component sum excludes Python imports/process launch, the native precision receipt RPC, energy identity initialization, NVML sampling and post-run research export/audit. Energy initialization and sampling are reported separately. The first-two-document component sum is labeled an estimate assembled from measured components and ABBA means; it is not a matched independent cold-service trial. Comparison to the old per-document capture path can report its separately measured setup components, not attribute their whole timing difference causally to this change.

CPU validation: `results/raw_pair_multidoc_offline_cpu_v1.json` contains document-isolation/partial-coverage tests and rejects missing ranks, stale sources, changed model or prefix, failed layer checks and missing captured layers. These are validation fixtures, not GPU numerical evidence.

After root confirms the frozen strict two-document qualification has completed successfully:

```bash
CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/benchmark_raw_pair_multidoc_production.py --qualification-summary results/raw_pair_multidoc_model_qual_v1/summary.json --prepare-only --output results/raw_pair_multidoc_production_cpu_v1
```

Root may then schedule an exclusive TP2 allocation:

```bash
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH CUDA_VISIBLE_DEVICES=2,3 FLASHINFER_CUDA_ARCH_LIST=12.0 .venv-engine/bin/python scripts/benchmark_raw_pair_multidoc_production.py --qualification-summary results/raw_pair_multidoc_model_qual_v1/summary.json --output results/raw_pair_multidoc_production_v1
```

`--setup-only` measures configuration and single production capture without any document generation. The full default run executes eight complete natural document groups (two documents in each of four arms); suggested timeout is 1800 seconds. Existing result summaries cannot be overwritten. No GPU launch has been performed by this subagent.
