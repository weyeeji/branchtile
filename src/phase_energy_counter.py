"""Read-only NVML energy counters; unsupported devices remain null."""
from datetime import datetime,timezone
import os,time

class PhaseEnergyCounter:
    def __init__(self,expected_uuids=None):
        self.error=None;self.handles=[];self.nvml=None
        self.selectors=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        self.expected_uuids=list(expected_uuids or [None]*len(self.selectors))
        assert len(self.expected_uuids)==len(self.selectors)
        try:
            import pynvml
            self.nvml=pynvml;pynvml.nvmlInit()
            for physical,expected in zip(self.selectors,self.expected_uuids):
                selector=expected or physical
                handle=(pynvml.nvmlDeviceGetHandleByUUID(selector) if selector.startswith('GPU-')
                        else pynvml.nvmlDeviceGetHandleByIndex(int(selector)))
                self.handles.append((physical,handle))
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            self.handles=[(physical,None) for physical in self.selectors]

    def sample(self,label):
        tick=time.perf_counter();result={'label':label,'UTC':datetime.now(timezone.utc).isoformat(),
                                       'monotonic_seconds':tick,'devices':[],'initialization_error':self.error}
        for (physical,handle),expected in zip(self.handles,self.expected_uuids):
            entry={'physical_selector':physical,'total_energy_mJ':None,'error':None,'expected_worker_UUID':expected,'identity_verified':False}
            try:
                if handle is None:raise RuntimeError(self.error)
                uuid=self.nvml.nvmlDeviceGetUUID(handle)
                entry['UUID']=uuid.decode() if isinstance(uuid,bytes) else uuid
                entry['identity_verified']=expected is not None and entry['UUID'].lower()==expected.lower()
                entry['total_energy_mJ']=int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
            except Exception as exc:entry['error']=f'{type(exc).__name__}: {exc}'
            result['devices'].append(entry)
        result['NVML_sampling_wall_seconds']=time.perf_counter()-tick
        return result


def interval(begin,end):
    left={d['physical_selector']:d for d in begin['devices']};right={d['physical_selector']:d for d in end['devices']}
    rows=[]
    for key in sorted(left.keys()|right.keys()):
        a=left.get(key,{}).get('total_energy_mJ');b=right.get(key,{}).get('total_energy_mJ')
        valid=a is not None and b is not None and b>=a
        identity=bool(left.get(key,{}).get('identity_verified') and right.get(key,{}).get('identity_verified') and left[key].get('UUID')==right[key].get('UUID'))
        rows.append({'physical_selector':key,'joules':(b-a)/1000 if valid else None,'counter_valid':valid,'identity_verified':identity})
    return {'wall_boundary_seconds':end['monotonic_seconds']-begin['monotonic_seconds'],'devices':rows,
            'total_GPU_joules':sum(d['joules'] for d in rows) if rows and all(d['counter_valid'] and d['identity_verified'] for d in rows) else None,
            'scope':'Measured device cumulative counter difference, includes all GPU activity within sample boundaries; no power/time estimate.'}
