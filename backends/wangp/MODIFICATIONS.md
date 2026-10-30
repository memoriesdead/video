# WanGP backend modifications

Modified distribution date: 2026-10-30 (UTC).

This is an independent, free patch and integration package for WanGP. It is not
an official WanGP release and does not imply upstream sponsorship or endorsement.
Upstream: https://github.com/deepbeepmeep/WanGP, revision
`2c67e50c8ed2c7074b4060bf6d6372af593a55ec` (2026-10-25).
The original WanGP and MMGP notices are preserved in `LICENSE.txt` and
`mmgp-LICENSE.md`. WanGP Community License 2.0 governs the redistributed WanGP
implementation and these derivative integration changes. Other backends and
separately installed dependencies retain their own licenses.

`patches/engine.patch` modifies only:

- `mmgp/cuda_graphs.py`: successful capture/replay counters and a statistics accessor.
- `models/minimax_h3/vsa/graphs.py`: new opt-in graph-state registration, warmed-cache admission, identity keys and fallback reporting.
- `models/minimax_h3/transformer.py`: graph-state preparation and eager input/PDD-head boundaries.
- `models/minimax_h3/pipeline.py`: bounded deterministic CPU image-reference latent reuse, version/configuration/precision invalidation, canonical CPU-state restoration and cloning on cache hits.
- `models/minimax_h3/minimax_h3_handler.py`: explicit graph opt-in and a restricted small-component residency group for the supported H3 backend.
- `wgp.py`: VSA graph admission and successful H3 task residency within an explicitly managed finite batch.

`run_batch.py` is a small integration adapter for the pinned `shared.api`
session contract. It validates all tasks before generation, supplies local-only
model-resolution settings and CPU thread caps, requires installed local FFmpeg,
and joins cancellation before explicit unload. The upstream generation lock
protects final session release. The adapter does not expose model conversion or
weight-saving commands.

The three H3 CPU fixture files are adapted to accept an explicit test-runtime
root. The adapter fixtures and test launcher are added here. No model weights,
encoders, projections, LoRAs, model conversions, private configuration, prompts,
generated media or operational receipts are distributed.

This export narrows the original local experiments to the standard supported
H3 residency admission. It excludes unrelated model-loader compatibility,
native activation and MLP scheduling experiments. Exported code has CPU fixture
coverage and clean-base patch-application verification; this export has not been
rerun on a GPU.
