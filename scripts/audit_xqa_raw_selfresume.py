"""CPU-only receipt, boundary arithmetic and raw-register mapping audit."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=ROOT/'results/xqa_raw_selfresume_gpu_v1/qualification.json'; d=json.loads(p.read_text())
 assert d['status']=='qualified_self_resume_only' and len(d['checks'])==5
 for path,h in d['source_sha256'].items():assert sha(ROOT/path)==h
 receipt=d['build_receipt']; directory=ROOT/'build/xqa_raw_continuation'/receipt['identity']
 assert sha(Path(receipt['upstream'])/'xqa/mha.cu')==receipt['origin_mha_sha256']
 for path,h in receipt['generated_sha256'].items():assert sha(directory/path)==h
 assert receipt['state_floats_per_cta']==4*32*35
 required=['native_pdl_equal','private_unsplit_equal','native_AA_10','checkpoint_did_not_write_O','active_state_finite','resume_bitwise_equal','native_FP64_pass','resume_FP64_pass','graph_resume_bitwise_equal']
 assert all(all(case[k] for k in required) for case in d['checks'])
 # Exhaust all phase boundary/split geometries at small lengths, several native S.
 geometry=0
 for S in (1,2,3,10,14,21,42):
  for tiles in range(0,70):
   for prefix in range(0,tiles+2):
    for split in range(S):
     original=list(range(split,tiles,S))
     cut=split+(((prefix-split+S-1)//S)*S if prefix>split else 0)
     before=list(range(split,min(prefix,tiles),S))
     after=list(range(cut,tiles,S))
     assert before+after==original
     geometry+=1
 # Independent MMA m16n8k16 output-coordinate derivation. The raw WarpAcc is
 # [Ntile=8][Mrow=2][Ncol=2], per lane; row=lane//4+8*Mrow.
 mapping=0; observed=set()
 for member in (0,1):
  for warp in range(4):
   for lane in range(32):
    for n in range(8):
     for j in range(2):
      native_row=lane//4; paired_row=lane//4+8*member
      native_index=4*n+j; paired_index=4*n+2*member+j
      assert native_index//4==paired_index//4 and paired_index%4//2==member
      column=64*(warp%2)+8*n+2*(lane%4)+j
      partial=warp//2
      observed.add((member,partial,native_row,column))
      assert paired_row-8*member==native_row
      mapping+=1
 assert len(observed)==mapping==4096
 # Row-scalar/hint layout is deduplicated by lane, unlike accumulator layout.
 scalars=[{'member':m,'native_lane':lane,'paired_lane':8*m+lane} for m in (0,1) for lane in range(8)]
 report={'status':'audited_selfresume_only','result_sha256':sha(p),'generated_source_count_verified':len(receipt['generated_sha256']),
  'library_sha256_at_audit':sha(d['library']),'state_bytes_B8_Hkv2_S10':8*2*10*4*32*35*4,
  'all_reported_five_case_checks_pass':True,'exhaustive_CPU_resume_geometry_cases':geometry,
  'CPU_prospective_pair_MMA_active_mapping_count':mapping,'CPU_prospective_pair_row_scalar_map':scalars,
  'limitations':['GPU equality used torch.equal on BF16, which does not distinguish signed zero. Strict int16-view equality remains needed.',
  'Only B8/S10 and short ~4K histories; nonempty-checkpoint cases have at most two native sequence iterations per split (zero-checkpoint cases reach three).',
  'Prefix16 and prefix0 both checkpoint zero tiles; partial-empty-checkpoint geometry lacks a GPU case.',
  'Graph replay used unchanged inputs once; dynamic same-address replay not tested.',
  'No cross-request sharing, performance or whole-model evidence.',
  'Prospective pair mapping audit is symbolic; it does not validate the future generated pairing implementation.',
  'All generated source hashes match. Audited shared-library hash identifies the current artifact but is not a proof of reproducible compilation.']}
 (ROOT/'results/xqa_raw_selfresume_audit_v1/audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='CPU_prospective_pair_row_scalar_map'},indent=2))
if __name__=='__main__':main()
