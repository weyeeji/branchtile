"""Read-only board-state samples; no clock, power, or policy changes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import pynvml as nv

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seconds',type=int,default=320);a=p.parse_args()
assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
nv.nvmlInit();handles=[nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
end=time.monotonic()+a.seconds
with open(a.output,'a',buffering=1) as stream:
    while time.monotonic()<end:
        now=datetime.datetime.now(datetime.timezone.utc).isoformat()
        for i,h in enumerate(handles):
            util=nv.nvmlDeviceGetUtilizationRates(h);mem=nv.nvmlDeviceGetMemoryInfo(h)
            row={'time_utc':now,'index':i,'uuid':nv.nvmlDeviceGetUUID(h),'gpu_util':util.gpu,
                 'memory_used':mem.used,'power_mw':nv.nvmlDeviceGetPowerUsage(h),
                 'power_limit_mw':nv.nvmlDeviceGetPowerManagementLimit(h),
                 'temperature_C':nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU),
                 'sm_clock_MHz':nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM),
                 'memory_clock_MHz':nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM),
                 'pstate':nv.nvmlDeviceGetPerformanceState(h)}
            stream.write(json.dumps(row)+'\n')
        time.sleep(1)
nv.nvmlShutdown()
