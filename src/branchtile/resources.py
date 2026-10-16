"""Extract compiler/launch resources without privileged GPU profiling."""
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile


def kernel_resources(kernel, artifact_prefix=None):
    """Inspect the actual cubin and the generated TVM launch arguments.

    The generated host stub supplies seven trailing integer launch parameters:
    grid xyz, block xyz, dynamic shared-memory bytes. We verify this shape and
    report both static (cubin) and dynamic (launch) shared memory separately.
    """
    executable = kernel.adapter.executable
    module = getattr(executable, 'mod', executable)
    queue = [module]
    cuda_modules = []
    while queue:
        module = queue.pop()
        if module.kind == 'cuda':
            cuda_modules.append(module)
        queue.extend(module.imports)
    assert len(cuda_modules) == 1, 'Expected exactly one producer CUDA module'
    source = kernel.get_kernel_source()
    host = kernel.adapter.host_kernel_source
    if host is None:
        host = kernel.adapter.rt_mod.inspect_source()
    # This specific generated host wrapper passes no scalar tensor arguments.
    # Its int64 casts therefore end in precisely the seven launch parameters.
    values = re.findall(r'v_int64\) = \(\(int64_t\)(\d+)\);', host)
    assert len(values) >= 7
    launch = list(map(int, values[-7:]))
    assert launch[3] in (32, 64, 128, 256) and launch[4:6] == [1, 1]
    with tempfile.TemporaryDirectory(prefix='branchtile-resources-') as tmp:
        cubin = Path(tmp)/'producer.cubin'
        cuda_modules[0].write_to_file(str(cubin), 'cubin')
        output = subprocess.run(['/usr/local/cuda-13.2/bin/cuobjdump', '--dump-resource-usage', str(cubin)],
                                check=True, capture_output=True, text=True).stdout
        cubin_sha256 = hashlib.sha256(cubin.read_bytes()).hexdigest()
        if artifact_prefix:
            dest = Path(artifact_prefix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.with_suffix('.cubin').write_bytes(cubin.read_bytes())
            dest.with_suffix('.cu').write_text(source)
            dest.with_suffix('.resources.txt').write_text(output)
    fields = {k.lower(): int(v) for k, v in re.findall(r'\b(REG|STACK|SHARED|LOCAL):([0-9]+)', output)}
    return dict(cubin_sha256=cubin_sha256, cubin_resources=fields, launch_grid=launch[:3], launch_block=launch[3:6],
                dynamic_smem_bytes=launch[6], total_smem_bytes=launch[6]+fields.get('shared', 0),
                generated_source_sha256=hashlib.sha256(source.encode()).hexdigest())
