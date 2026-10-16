"""Exact spatial owner-task/packed-reduction control, not a novelty claim."""
from __future__ import annotations
import time
import torch
from .owner_tasks_plan import build_task_layout
from .triton_kernel_owner_tasks import producer, merge


class Attention:
    def __init__(self, plan, splits=64, block_n=64, threads=128, compact=True,
                 region_partials=False, chunk_tokens=512):
        if not compact or region_partials or threads != 128 or block_n not in (32, 64):
            raise ValueError("Owner tasks require compact=True, region_partials=False, threads=128, N32/N64")
        if splits <= 0:
            raise ValueError("splits must be positive")
        started = time.perf_counter()
        self.plan, self.splits = plan, splits
        self.last_producer_compiled = None
        self.last_merge_compiled = None
        self.block_n, self.threads = block_n, threads
        heads, tokens = plan.indices.shape
        if not plan.indices.is_cuda:
            raise ValueError("Construct GPU Attention with a CUDA plan; CPU planner is separate")
        offsets = plan.offsets.detach().cpu().tolist()
        counts = plan.counts.detach().cpu().tolist()
        row_map = plan.row_map.detach().cpu().tolist()
        layout = build_task_layout(offsets, row_map, counts, list(plan.labels),
                                   plan.batch, plan.gqa, tokens, chunk_tokens)
        self.metadata_ms = (time.perf_counter()-started)*1000
        self.layout = layout
        self.num_tasks = len(layout["tasks"])
        self.shape = heads, plan.rows, 128
        device = plan.indices.device
        self.tasks = torch.tensor(layout["tasks"], dtype=torch.int32, device=device).reshape(self.num_tasks, 6)
        self.merge_indices = torch.tensor(layout["merge_indices"], dtype=torch.int32, device=device).reshape(heads, plan.batch, layout["max_contributions"])
        self.contribution_counts = torch.tensor(layout["contribution_counts"], dtype=torch.int32, device=device)
        self.u = torch.empty((layout["packed_rows"], 128), dtype=torch.float32, device=device)
        self.m = torch.empty((layout["packed_rows"],), dtype=torch.float32, device=device)
        self.l = torch.empty_like(self.m)
        self.output = torch.empty((heads, plan.rows, 128), dtype=torch.bfloat16, device=device)
        self.lse = torch.empty((heads, plan.rows), dtype=torch.float32, device=device)
        self._freeze_plan_topology()
        buffers = {"packed_U": self.u, "packed_M": self.m, "packed_L": self.l,
                   "task_descriptors": self.tasks, "merge_indices": self.merge_indices,
                   "contribution_counts": self.contribution_counts, "output": self.output, "LSE": self.lse}
        self.workspace_bytes_by_buffer = {name: tensor.numel()*tensor.element_size() for name, tensor in buffers.items()}
        self.workspace_bytes = sum(self.workspace_bytes_by_buffer.values())
        self.payload_aliases_source = True
        self.logical_kv_token_visits = sum(task[3]-task[2] for task in layout["tasks"])
        self.metadata = {"mechanism": "spatial_owner_tasks_packed_reduction_control",
                         "chunk_tokens": chunk_tokens, "task_count": self.num_tasks,
                         "packed_partial_rows": layout["packed_rows"],
                         "requested_splits_compatibility_parameter": splits,
                         "splits_semantics": "Not used for partitioning; fixed chunk_tokens defines tasks. This is not the original equal-split scheme.",
                         "prepare_scope": "CPU metadata D2H, validation, task/table construction, H2D and owned-buffer allocation; first kernel JIT excluded.",
                         "workspace_bytes_by_buffer": self.workspace_bytes_by_buffer,
                         "workspace_bytes": self.workspace_bytes,
                         "max_contributions": layout["max_contributions"],
                         "merge_streaming_group": 16,
                         "structure_sha256": layout["structure_sha256"],
                         "payload_aliases_source": True,
                         "graph_topology_contract": "Q/K/V and original Indices values may change in place. Replacing the plan or any plan tensor, changing tensor storage/layout, or changing offsets/labels/counts/row_map requires rebuild and graph recapture. graph.replay bypasses host identity/version checks; callers must enforce this contract."}
        torch.cuda.synchronize(device)
        self.prepare_ms = (time.perf_counter()-started)*1000

    def _freeze_plan_topology(self):
        # Strong references prevent tensor-id/address reuse from masking replacement.
        self._original_plan = self.plan
        self._plan_tensors = {name: getattr(self.plan, name)
                              for name in ("indices", "offsets", "counts", "row_map")}
        self._plan_tensor_ptrs = {name: tensor.data_ptr() for name, tensor in self._plan_tensors.items()}
        self._plan_tensor_layouts = {
            name: (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), tensor.device, tensor.dtype)
            for name, tensor in self._plan_tensors.items()}
        self._metadata_versions = {name: self._plan_tensors[name]._version
                                   for name in ("offsets", "counts", "row_map")}
        self._labels = tuple(self.plan.labels)
        self._plan_static_layout = self.plan.batch, self.plan.gqa, self.plan.rows

    def validate_plan(self):
        message = "Owner-task topology metadata changed: rebuild Attention outside replay and recapture the graph"
        if self.plan is not self._original_plan:
            raise RuntimeError(message)
        if ((self.plan.batch, self.plan.gqa, self.plan.rows) != self._plan_static_layout
                or tuple(self.plan.labels) != self._labels):
            raise RuntimeError(message)
        for name, original in self._plan_tensors.items():
            tensor = getattr(self.plan, name, None)
            if (tensor is not original or tensor.data_ptr() != self._plan_tensor_ptrs[name]
                    or (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), tensor.device, tensor.dtype)
                    != self._plan_tensor_layouts[name]):
                raise RuntimeError(message)
        if any(self._plan_tensors[name]._version != version for name, version in self._metadata_versions.items()):
            raise RuntimeError(message)

    def __call__(self, q, k, v):
        self.validate_plan()
        heads, rows, dim = self.shape
        tokens = self.plan.indices.shape[1]
        if q.shape != (heads, rows, dim) or k.shape != (heads, tokens, dim) or v.shape != k.shape:
            raise ValueError("Q/K/V shape mismatch")
        if any(tensor.dtype != torch.bfloat16 or tensor.device != self.output.device or not tensor.is_contiguous()
               for tensor in (q, k, v)):
            raise ValueError("Q/K/V must be contiguous BF16 tensors on the plan device")
        if self.num_tasks:
            self.last_producer_compiled = producer[(self.num_tasks,)](
                q, k, v, self.plan.indices, self.plan.row_map, self.tasks,
                self.u, self.m, self.l, ROWS=rows, TOKENS=tokens,
                REGIONS=len(self.plan.labels), N=self.block_n, D=dim,
                num_warps=4, num_stages=1)
        self.last_merge_compiled = merge[(heads, rows)](
            self.u, self.m, self.l, self.merge_indices, self.contribution_counts,
            self.output, self.lse, BATCH=self.plan.batch, GQA=self.plan.gqa,
            ROWS=rows, MAX_CONTRIB=self.layout["max_contributions"], D=dim,
            GROUP=16, num_warps=4)
        return self.output, self.lse
