# ISCAS 2027 scope and submission constraints

Checked against the official submission system and IEEE CASS on 2026-10-17.

The [author instructions](https://epapers2.org/iscas2027/ESR/authorinfo1.php)
state October 22, 2026 at 23:59 UTC−05:00, equivalent to **October 23 at
12:59 Asia/Shanghai**. [IEEE CASS active calls](https://ieee-cas.org/active-calls)
also lists the October 22 extension. The conference's general call still
shows October 13 in its regular-paper section; do not infer a missed deadline
from that stale section or confuse the separately listed special-session date.

The limit is four pages of technical content with an optional fifth page
containing references only; author names are included. The PDF limit is 5 MB.
These are formatting requirements, not evidence that this project is accepted
or that a particular percentage improvement qualifies as a publishable result.

The [official topic list](https://epapers2.org/iscas2027/ESR/display_tracks.php)
includes 14.3, system architectures and hardware solutions for foundation
models and LLMs, and 14.6, hardware-aware learning and cross-layer optimization.
An implementation paper must substantiate its hardware mechanism and practical
effect. Merely renaming shared-prefix attention or presenting speedups over a
weak baseline does not establish a contribution. Special sessions are marked
invited-only; this work is not assumed to have an invitation.

For this project, a defensible candidate contribution is native-schedule
shared-prefix computation with explicit preservation of unreduced warp state
and hidden control history, plus measured integration cost. The controlled
counterexample and repair explain the technical issue. Native FlashInfer/XQA
is the primary deployed baseline; the locally repaired PAT comparison has
bounded operator-level scope. New documents, full natural trajectories,
both model families, setup costs, task quality and energy attribution must
remain visible. A microbenchmark ratio cannot substitute for end-to-end data.

The stable old six-document repeat currently suggests modest net benefit;
the direct-input/device-cut combination, TP1 configuration and finite
self-consistency workload are separate experiments, not gains already proved.
Close unsuccessful mechanisms rather than accumulating tuning results as
independent evidence. No submission, author identity, affiliation or acceptance
claim is authorized or implied by recording these requirements.
