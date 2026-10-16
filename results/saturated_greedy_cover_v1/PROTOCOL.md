# Full-universe saturated greedy cover control

This protocol and script are committed before reading natural selector tensors
or new natural baseline results for this task. It is one classical greedy
cover arm, not a new algorithm, an exact natural-data oracle, or a claimed
constant-factor approximation.

The domain is the same three corrected official main-model `suffix_v2`
captures, rank 0, layer 15, both local KV heads, and document budgets 4096 and
16384. Epsilon is fixed to 0, 0.001, 0.005 and 0.01. Epsilon 0 returns the
original independent support bit for bit. Every positive-epsilon run starts
empty over **all nonmandatory tokens in the actual common document source**,
including positions outside the original masks' union. First4/last32 remain
mandatory; the document tail in private KV remains private and mandatory.
System prefix and other private KV remain fixed and outside this selection.

For request r, let k_r be its free budget, c_r its mandatory score, A_r its
original selected score, and tau_r = (1-epsilon) A_r - c_r. Let g_r(U) be the
sum of its top k_r scores available in U. Greedy adds the unit-cost token with
maximum exact marginal of

```text
F(U) = sum_{r: tau_r > 0} min(g_r(U)/tau_r, 1)
       + min(|U|/max_r(k_r), 1).
```

The cardinality term is omitted only if all free budgets are zero. Each
request uses its own normalization tau; there are no fitted affinity weights.
Original token ID, ascending, is the single tie-break. Lazy old marginals are
upper bounds by weighted-rank submodularity; candidates are recomputed until
the best exact current marginal dominates all remaining bounds.

Original stored binary scores are represented losslessly as per-request
integers. Epsilon uses its preregistered decimal rational (1/1000, 1/200,
1/100). Normalized marginal sums use an exact common denominator, allowing
integer comparisons and exact tie-breaking rather than rounded lazy bounds.
No score is quantized. This exact implementation's CPU cost is recorded.

Once every score and cardinality constraint passes, each request independently
takes stable top-k within U and all mandatory tokens, refilling to its exact
budget even when some scores are zero. Unused U entries disappear from the
reported final union. No reverse-delete postprocessing is included. The
original `math.fsum` loss and loss/original must both meet the original float
epsilon checks. A rare boundary-rounding failure is preserved as failure;
the threshold is never silently relaxed or repaired using held-out signals.

Before natural execution, qualification covers the review's multi-owner
triangle and outside-initial-union counterexamples (exact optima 2 and 1),
fixed common/private and zero-score cardinality cases, and 60 deterministic
random small cases (seed 17019). Exhaustive enumeration supplies the small
feasible-U optimum; lazy greedy must match complete greedy recomputation's
entire chosen sequence and exact marginals. Greedy is not required to equal
the global optimum, and doing so on fixtures would not prove it generally.

Natural traversal is fixed: source-list order, increasing budget, increasing
KV head, increasing epsilon. All 48 rows are requested (12 identity, 36 cover).
Each completed row flushes to disk. A process CPU soft/hard limit of 590/600
seconds and cooperative 575 CPU / 590 wall-second limits preserve a partial
prefix if necessary. The pending cell is explicitly named on timeout. No
post-hoc subset of successful configurations is selected.

Only metadata, suffix-only pooled scores and original supports are opened.
No decode Q/K/V, actual attention errors, model outputs, or GPU are used.
Reported outputs include per-request retained score and support hashes,
selected entries, actual-common-source logical union and owner/N64 tiles,
greedy candidate/chosen counts, exact marginal evaluations, CPU/wall time,
and process peak RSS. Full source KV remain allocated: physical capacity
released is zero. Feasibility in pooled score does not establish attention
or generation quality, and CPU selector timing is not kernel latency.

```bash
.venv/bin/python scripts/probe_saturated_greedy_cover.py
```

The script refuses existing output. Independent qualification can be run with
`--self-test`, which opens no natural data. Results from earlier methods may
be joined only after this protocol is committed and this arm is fixed.
