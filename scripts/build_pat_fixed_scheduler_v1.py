"""CPU-only PAT compatibility build: repair split-slot numbering, preserve CUDA math."""
import difflib,hashlib,importlib.util,json,os,pathlib,shutil,sys,time,traceback
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ.update(TORCH_CUDA_ARCH_LIST='12.0',FLASHINFER_CUDA_ARCH_LIST='12.0',MAX_JOBS='1',CUDA_HOME='/usr/local/cuda-13.2')
os.environ['PATH']=str(R/'.venv-engine/bin')+os.pathsep+os.environ['PATH']
output=R/'results/pat_fixed_scheduler_build_v1.json';assert not output.exists()
basepath=R/'results/pat_scratch_debug_build_v1.json';base=json.loads(basepath.read_text())
assert base['status']=='built_unqualified_scratch_debug'
src=R/'build/pat_fixed_scheduler_v1';assert not src.exists()
for name,h in base['generated_sha256'].items():assert sha(R/'build/pat_scratch_debug_v1'/name)==h
shutil.copytree(R/'build/pat_scratch_debug_v1',src)
header=src/'prefix_tree.h';old=header.read_text()
needle='        packed_boxes = balancePack(packed_boxes, kvHead, HRatio);'
assert old.count(needle)==1
patch='''

        // Compatibility repair: original tree counters can differ between members
        // of a shared box after private descendants have received ranks. Preserve
        // every box and its original position, and number shared sets first.
        // Reject schedules for which a common contiguous rank cannot be assigned.
        const std::vector<int> original_counts = split_per_seq;
        std::vector<int> next_rank(original_counts.size(), 0);
        std::vector<size_t> order(packed_boxes.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const auto& qa = packed_boxes[a].q_table;
            const auto& qb = packed_boxes[b].q_table;
            if (qa.size() != qb.size()) return qa.size() > qb.size();
            return qa < qb;
        });
        for (size_t index : order) {
            auto& box = packed_boxes[index];
            TORCH_CHECK(!box.q_table.empty(), "PAT fixed scheduler: empty box");
            std::vector<bool> members(next_rank.size(), false);
            int common_rank = -1;
            for (int qid : box.q_table) {
                TORCH_CHECK(qid >= 0 && size_t(qid) < next_rank.size(),
                            "PAT fixed scheduler: request index out of bounds");
                TORCH_CHECK(!members[qid], "PAT fixed scheduler: duplicate request in box");
                members[qid] = true;
                if (common_rank < 0) common_rank = next_rank[qid];
                TORCH_CHECK(next_rank[qid] == common_rank,
                            "PAT fixed scheduler: nonuniform member next-rank; unsupported schedule");
            }
            box.CTA_rank = common_rank;
            for (int qid : box.q_table) ++next_rank[qid];
        }
        TORCH_CHECK(next_rank == original_counts,
                    "PAT fixed scheduler: altered per-request contributor counts");
        std::vector<std::vector<bool>> written(next_rank.size());
        for (size_t q = 0; q < next_rank.size(); ++q) written[q].resize(next_rank[q], false);
        for (const auto& box : packed_boxes) {
            for (int qid : box.q_table) {
                TORCH_CHECK(box.CTA_rank >= 0 && box.CTA_rank < next_rank[qid],
                            "PAT fixed scheduler: rank outside gather range");
                TORCH_CHECK(!written[qid][box.CTA_rank],
                            "PAT fixed scheduler: overlapping scratch writers");
                written[qid][box.CTA_rank] = true;
            }
        }
        for (const auto& ranks : written) {
            TORCH_CHECK(std::all_of(ranks.begin(), ranks.end(), [](bool x) { return x; }),
                        "PAT fixed scheduler: unwritten gather slot");
        }
        split_per_seq = next_rank;
'''
new=old.replace(needle,needle+patch);header.write_text(new)
diff=R/'results/pat_fixed_scheduler_v1.patch';assert not diff.exists()
diff.write_text(''.join(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='a/prefix_tree.h',tofile='b/prefix_tree.h')))
objs=[R/p for p in base['reused_cuda_objects']]
for p,h in base['reused_cuda_objects'].items():assert sha(R/p)==h
rep={'status':'building','scope':__doc__,'label':'PAT compat fixed scheduler (local), not unchanged upstream','base_receipt':{'path':str(basepath.relative_to(R)),'sha256':sha(basepath)},'commit':base['commit'],'changes':['Assign common CTA ranks by stable shared-set-first order; preserve original box and launch order.','Reject nonuniform-member schedules; prove per-request contiguous unique scratch slots before GPU upload.','Keep NaN scratch and metadata diagnostics from previous build.'],'unchanged':['All CUDA objects including arithmetic, gather, v4 explicit stream dependencies.','Query sets, KV token partitions, CTA count, tile choices, per-request split counts.'],'generated_sha256':{p.name:sha(p) for p in src.iterdir() if p.is_file()},'reused_cuda_objects':base['reused_cuda_objects'],'patch':str(diff.relative_to(R)),'patch_sha256':sha(diff),'source_sha256':{str(pathlib.Path(__file__).relative_to(R)):sha(__file__)}}
t=time.perf_counter();output.write_text(json.dumps(rep,indent=2)+'\n')
try:
 import torch
 from torch.utils.cpp_extension import load
 fi=pathlib.Path(importlib.util.find_spec('flashinfer').origin).parent/'data/cutlass'
 directory=R/'build/pat_fixed_scheduler_extension_v1';directory.mkdir()
 module=load(name='_prefix_attn',sources=[str(src/'bindings.cpp')],extra_include_paths=[str(src),'/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl',str(fi/'include'),str(fi/'tools/util/include')],extra_cflags=['-O3','-std=c++20'],extra_ldflags=[str(p) for p in objs],with_cuda=True,build_directory=str(directory),verbose=True)
 rep.update(status='built_unqualified_fixed_scheduler',module=module.__file__,module_sha256=sha(module.__file__),CUDA_initialized=torch.cuda.is_initialized(),flashinfer_imported='flashinfer' in sys.modules)
 assert not rep['CUDA_initialized'] and not rep['flashinfer_imported']
except Exception:
 rep.update(status='failed',error=traceback.format_exc());raise
finally:
 rep['CPU_build_seconds']=time.perf_counter()-t;output.write_text(json.dumps(rep,indent=2)+'\n')
