"""Read-only NVML monitor. Never changes clocks, power limits or device state."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time
import pynvml as nv

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--seconds", type=int, default=3600)
p.add_argument("--interval", type=float, default=1)
a = p.parse_args()
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
nv.nvmlInit()
handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
end = time.monotonic() + a.seconds
with open(a.output, "a", buffering=1) as f:
    while time.monotonic() < end:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for i, h in enumerate(handles):
            u = nv.nvmlDeviceGetUtilizationRates(h)
            m = nv.nvmlDeviceGetMemoryInfo(h)
            try:
                energy = nv.nvmlDeviceGetTotalEnergyConsumption(h)
            except nv.NVMLError:
                energy = None
            f.write(json.dumps({"time_utc": timestamp, "index": i,
                "uuid": nv.nvmlDeviceGetUUID(h), "gpu_util": u.gpu,
                "memory_used": m.used, "power_mw": nv.nvmlDeviceGetPowerUsage(h),
                "total_energy_mj": energy,
                "temperature_C": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)}) + "\n")
        time.sleep(a.interval)
nv.nvmlShutdown()
