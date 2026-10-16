# Preserved eval2 failure and independent late-state localization

The frozen heldout production run `results/native_raw_pair_heldout_production_v1/eval2_bf16` fails exact-output parity. Per the independent failure audit,the first changed output is question6 at index1939: native3405 versus candidate8350. Graph index1938 has actual B9,questions[2,3,4,6,7,9,10,11,12],S9,table[9,10240],nominal prefix113280 and lengths115253–115264. The B9 phase began around graph1786; checking only its first8states was insufficient. The failure remains part of the evidence and is not removed from the dataset.

CPU boundary inspection does not identify a simple metadata mistake at this state. Prefix442complete256-token tiles ends at113152; all nine lengths have ceil(L/256)=451. Using the even member's total length in the shared prefix cannot mask any fully shared tile here. Native S remains9 from full table extent163840. Split0 resumes at450; splits1–8 resume at442–449. Every private/mixed-boundary tile still uses its original request's query,KV and length. This is a local structural check,not proof of universal numerical identity.

The new diagnostic preserves the original native sampler trajectory and never selects candidate hidden state. It enables only B9 candidate graphs initially and instruments native graph indices1760–1960; natural output is bounded at1970tokens to stop after the target window. The complete produced token prefix must match the previously recorded native output and is reported explicitly. Early EOS remains enabled. This capped run is diagnosis,not a quality or timing experiment.

At each selected state,the frozen strict routine runs original native AA,witness,checked and production graphs,checks hidden/local+gathered logits/all48layer Q/O/currentKV slots/metadata,and restores native KV. Then every layer executes on **identical witness Q and restored native KV**:

- Production public XQA twice for same-Q AA.
- Public XQA with PDL off.
- Complete raw-pair layer call.
- The same candidate after raw-state NaN poisoning.

Per-layer byte equality and mismatch counts distinguish native AA instability,PDL differences,full-model input drift and same-Q attention failure. The first failing layer per rank is exported with exact Q/output variants,remapped used physical KV pages,original layout/table width,lengths,S and both paired and unshared native raw prefix states. Unshared checkpoint→resume output is also recorded,so later analysis can distinguish prefix fragment/hint mismatch from the checkpoint boundary itself. Snapshot copying is outside any performance claim.

The independent files are `scripts/diagnose_raw_pair_late_same_state.py` and `scripts/vllm_raw_pair_same_state_diagnostic_worker.py`. They subclass the frozen worker without editing it or any CUDA source. Original failed results remain untouched. GPU assignment is exclusively root's responsibility:

```bash
CUDA_VISIBLE_DEVICES=0,1 FLASHINFER_CUDA_ARCH_LIST=12.0 .venv-engine/bin/python scripts/diagnose_raw_pair_late_same_state.py --output results/raw_pair_eval2_late_same_state_v1
```

Default B9/window targets the first known divergence. A fully passing window does not close the failure:earlier states or another cause must then be examined. The output explicitly distinguishes completion of diagnosis from successful numerical qualification.
