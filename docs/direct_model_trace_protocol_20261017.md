# Short full-model direct-input trace protocol

`scripts/profile_native_raw_pair_direct_model.py` prepares the original training train26 cohort, accepts only new direct-worker offline qualification, and captures a production-only graph bank. Its worker subclass adds only CPU capture-address observation and profiler RPCs; attention arithmetic and dispatch remain inherited from the qualified worker. These instrumentation sources are separately hashed. The observer does not retain tensors, allocate GPU buffers or modify operands.

Native and candidate each generate16 fixed tokens on the same real prompts. Their uncached one-token anchor, graph capture and setup occur outside profiling. Each TP rank saves a CUDA/CPU trace; exact output tokens, graph request/position trajectories, source provenance and candidate execution are checked. Profiling includes any remaining group prefill as well as decode. No profiled time is reported as an acceleration measurement.

After collection, `scripts/audit_direct_model_trace.py` checks that each candidate model-graph replay contains96 direct XQA nodes (two per48 layers), while each native graph contains48 native XQA nodes. Direct nodes must carry nonzero graph/node IDs and correlate to `cudaGraphLaunch`. It separately lists normal framework copy/cache kernels rather than claiming the model has no copies. The CPU bank review supplies the no-adapter-copy argument; CUPTI kernel names alone cannot determine arbitrary copy ownership.

The strict source is unchanged. A profiler/trace failure is retained and is not a mathematical rejection of a prior qualified kernel. Graph node argument memory is not independently decoded; the CPU observer records Q/O/cache addresses actually passed during capture as additional evidence.

The parent must schedule the manifest through `scripts/run_jobs_locked.py` after the new direct model qualification is complete. No GPU work was launched while preparing this harness.
