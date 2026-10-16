# PAT explicit stream dependency compatibility hypothesis

The untouched local PAT multistream artifact passed its first output check but
failed all 30 subsequent synchronized repetitions. Its official
`DISABLE_STREAM=1` control passed all 30. This establishes a dependence on the
execution path in this environment; it does not identify a unique cause.

The official dispatcher uses `stream=0`, creates blocking child streams, and
launches the gather on implicit stream0. Under documented legacy-default stream
semantics, that gather should already wait for blocking child streams. We found
no per-thread-default compiler flag or matching macro definition in the inspected
Torch CUDA headers. Therefore **missing implicit synchronization is not proven**.
A concurrent scratch-write issue, another runtime interaction, or an unchecked
stream API failure remains possible. [NVIDIA CUDA 13.2 stream semantics](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/stream-sync-behavior.html).

The independent copied-source variant uses the current PyTorch caller stream,
records a ready event there, makes each child wait before launching, records a
done event on each child, and makes the caller wait before its gather. All stream
and event API results are checked. Queued event waits retain their dependency
while event resources may be released asynchronously after completion; this
permits destruction after enqueueing the join. [NVIDIA CUDA 13.2 event API](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__EVENT.html).

This explicit contract is a reasonable compatibility adaptation for arbitrary
PyTorch caller streams and allocator ordering. It is a **testable hypothesis**
for the observed default-stream failure. If it fails the same numerical test,
we should reject that explanation and inspect concurrent writes; no arithmetic
change or tolerance relaxation is justified.

## Reproducibility and gate

The official checkout and original v2 shared object remain untouched. New build
scripts copy the pinned official source. Only the dispatch header and the already
needed obsolete, unused Thrust include change. All arithmetic kernels, tree
construction, tile selection and allocation strategy remain unchanged. The
receipt records original/copied source hashes, the patch and shared-object hash.
Per-call events/stream lifecycle remain charged in a future benchmark.

The first CPU-only build v3 failed before compilation because the shell PATH did
not include the existing venv ninja. That failed receipt is retained. Independent
v4 adds the existing venv bin directory to PATH; no packages are installed. Both
hide all GPUs before torch import, set both architecture variables to12.0, and
locate FlashInfer headers with `find_spec` without importing FlashInfer. Successful
completion must confirm `CUDA_initialized=false` and `flashinfer_imported=false`.

`qualify_pat_stream_compat.py` repeats the actual BF16 B8 state with an unchanged
FP64 tolerance. Separate default and nondefault caller-stream runs test the
explicit contract. Each repetition synchronizes and records any failure; native
AA and raw pair must remain valid. The parent schedules normal and memcheck runs.
Only after qualification may a new matched eager comparison cite performance.
