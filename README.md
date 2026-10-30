# Local Video Engine

**9.5 seconds per warm video on an RTX 5090** in the measured local setup:
640x384, 107 frames, 24 FPS, eight steps, with audio. The first video task took
**25.11 seconds**. These are medians from four warm tasks and two first tasks.
Task timing includes processing and saving; process startup is separate.

The full coordinator failed its process identity guard before final exit.
Completed saved media passed a separate full video/audio decode and repeatability
audit. Those completed tasks supply the numbers above; the overall run remains
failed. They do not establish video quality or speed for every model.

The engine keeps stored model weights unchanged. It speeds execution through
reference caching, CUDA graph reuse and finite-batch residency. Ordinary batching
uses compatible models supported by WanGP; graph reuse and GPU retention have
specific H3 admission rules. No models, prompts, media or private configurations
are included. Inference runs locally with no hosted fallback.

## Setup

Python 3.10+ and Git are needed for the patch tool. Start with a separate runtime
checkout, then apply the pinned source patch:

```shell
git clone https://github.com/deepbeepmeep/WanGP.git video-runtime
git -C video-runtime checkout 2c67e50c8ed2c7074b4060bf6d6372af593a55ec
python tools/engine.py apply wangp --checkout ./video-runtime
```

[Install the runtime and run a batch](backends/wangp/README.md#run-a-finite-local-batch)
with your own compatible local models, config, manifest and FFmpeg installation.
The [backend instructions](backends/wangp/README.md) cover dependencies, a dry run,
optional speed settings and CPU tests. Check free GPU/CPU/RAM before generation.

## Checks and licenses

```shell
python -B -m unittest discover -s tests
python tools/check_release.py --check
```

[Benchmark details](benchmarks/reference.json) record the sample times and limits.
The backend README also describes the separate historical graph OFF/ON comparison.
One-second fresh finished video has not been achieved.

Original top-level tools use [Apache-2.0](LICENSE). WanGP-derived patches, adapter
and tests use [WanGP Community License 2.0](backends/wangp/LICENSE.txt), with its own
redistribution and commercial-use terms. Read [LICENSES.md](LICENSES.md) and
[the notices](THIRD_PARTY_NOTICES.md). Model licenses remain separate.
