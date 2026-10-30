# WanGP local batch backend

This source-only patch pack adds optional H3 VSA block-graph reuse, a bounded
reference-latent cache and finite-batch GPU residency to a pinned WanGP runtime.
It also supplies a portable finite-session runner with four CPU/compiler threads.
No model or runtime configuration is included. Use your existing, legally obtained,
compatible local models and your own exported task manifest.

WanGP Community License 2.0 applies to the WanGP implementation and derivative
patches/integration. Read `LICENSE.txt`, `mmgp-LICENSE.md` and `MODIFICATIONS.md`.
This is a free independent integration; upstream does not endorse it. Paid access,
embedding and redistribution have additional restrictions under the upstream
license. This backend's license does not replace another backend's or model's terms.

## Compatibility and source setup

- Upstream: https://github.com/deepbeepmeep/WanGP
- Exact source revision: `2c67e50c8ed2c7074b4060bf6d6372af593a55ec`.
- Start from that clean source revision; the patch targets exactly the six files
  listed in `backend.json`. The runner checks their canonical LF source hashes.
- Use an isolated runtime environment matching the upstream installation guide.
  The historical qualification used Python 3.11, PyTorch 2.10.0, torchvision 0.25.0,
  torchaudio 2.10.0, CUDA 13 and WGPKernels 1.4.2 on an NVIDIA SM120 device.
  Those are provenance details, not a claim that every combination is supported.
- Models, kernels and FFmpeg must already be installed locally. Downloads for
  initial dependency/model setup are the user's separate responsibility; the
  runner makes no setup download and has no hosted-inference fallback.
- Basic finite-session batching uses compatible models supported by upstream.
  H3 VSA graph capture is specific to the existing 128-token Triton INT8-QK route,
  with existing PV arithmetic and a complete eager cache warmup. Unsupported
  graph state stays eager. No universal-model or cross-hardware acceleration is claimed.

Apply using the package patch installer, or use Git directly from a shell. In
these commands `BACKEND` is this directory and `RUNTIME` is your separate checkout:

```powershell
git clone https://github.com/deepbeepmeep/WanGP.git "$RUNTIME"
git -C "$RUNTIME" checkout --detach 2c67e50c8ed2c7074b4060bf6d6372af593a55ec
git -C "$RUNTIME" apply --check "$BACKEND/patches/engine.patch"
git -C "$RUNTIME" apply "$BACKEND/patches/engine.patch"
```

Only Python source changes are needed for this patch; there is no custom native
build or bundled binary. Install backend dependencies separately using the
pinned upstream `docs/INSTALLATION.md`, selecting versions for your OS/GPU.
Keep the checkout and Python environment separate from other projects.

## CPU verification without models or GPU

The control-flow fixtures need PyTorch installed, but create only CPU tensors
and stand-ins for CUDA state. Pass the patched checkout explicitly:

```powershell
python -B "$BACKEND/tests/run_cpu_tests.py" --runtime "$RUNTIME"
```

Standard-library-only runner tests can be used without PyTorch or a checkout:

```powershell
python -B -m unittest discover -s "$BACKEND/tests" -p test_adapter.py -v
python -B "$BACKEND/run_batch.py" --help
```

CPU fixtures verify state admission/invalidation, bounded cache identity and
cloning, canonical CPU restoration, successful replay accounting, all-task
validation, cleanup order, cancellation joins, environment restoration and
session serialization. They do not validate CUDA capture, video quality or speed.

## Run a finite local batch

Use the Python interpreter from your WanGP environment. Provide your own
`wgp_config.json`, existing local FFmpeg directory, finite exported JSON manifest,
and a new output directory. Configuration must disable startup preloading and
automatic model unloading (`preload_model_policy` must contain neither `P` nor
`U`). The runner checks its pinned source before importing the runtime.

```powershell
python "$BACKEND/run_batch.py" --runtime "$RUNTIME" --config "$CONFIG" --manifest "$MANIFEST" --output "$NEW_OUTPUT" --ffmpeg-dir "$FFMPEG_DIRECTORY" --dry-run
python "$BACKEND/run_batch.py" --runtime "$RUNTIME" --config "$CONFIG" --manifest "$MANIFEST" --output "$NEW_OUTPUT" --ffmpeg-dir "$FFMPEG_DIRECTORY"
```

The manifest uses WanGP's exported task form: a nonempty JSON list of objects
with a unique `id` and a `params` object. Model definitions and all reference
paths must resolve locally through your runtime. No private prompts or example
media are supplied here. All tasks are normalized and passed through the pinned
upstream validator before the first generation submission. Dry-run initializes
the backend but rejects unexpected model loading; it does not start generation.

Each invocation owns one local in-process session, consumes progress events,
reuses that session across its finite task list and releases its models on exit.
No server or public endpoint is started by the adapter. Use one fresh command
process per batch; do not embed it in a concurrent application. Its lock prevents
two sessions in one process because VSA state and the upstream API are global.
The program does not stop other jobs or allocate workstation resource shares;
check available GPU/CPU/RAM capacity before actual inference.

Optional `--image-latent-cache` enables the CPU single-frame reference cache,
limited to one latent of at most 16 MiB. Input content, VAE identity/version,
configuration, training state, tiling and arithmetic policy participate in its
key. Unknown tensor versions bypass the cache. Hits return independent clones;
fresh task noise and cancellation checks remain active. If an external tool
changes tensor `.data` without a version update, it must call
`clear_image_latent_cache()` before reuse.

For the specifically supported H3 backend, add all three explicit options:

```powershell
python "$BACKEND/run_batch.py" --runtime "$RUNTIME" --config "$CONFIG" --manifest "$MANIFEST" --output "$NEW_OUTPUT" --ffmpeg-dir "$FFMPEG_DIRECTORY" --image-latent-cache --h3-vsa-graphs --keep-gpu-resident
```

Residency accepts only `minimax_h3_fasth3_omni_nvfp4` tasks with identical settings,
prompt and references; only `seed` and `client_id` may differ. It requires the
existing `nvfp4_awq,int8_convrot` backend configuration, eight Euler steps, one
guidance phase, video output, and literal overrides selecting `memory_profile=0`
and `compilation='cudagraph'` with VSA attention. Other models can use ordinary
batching; they cannot opt into this H3 residency contract. Large text/vision
encoders and latent upscalers are excluded from the retained group. Retention
is cleared before cancellation, an incomplete job is joined before explicit
unload, and final close uses the upstream generation lock.

## Evidence and limits

Before this public export, a matched local experiment at unchanged 640x384,
107 frames, 24 FPS and eight Euler steps compared retained graphs OFF/ON/ON/OFF.
All twelve clips matched their paired full decoded video and audio hashes.
Fully warm third-clip medians were 12.213 seconds OFF and 9.002 seconds ON
(26.29% less time), with only two observations per arm for the same prompt and
reference. The two ON three-clip processes took 62.984 and 67.859 seconds including
startup. Cold startup differences prevent attributing a whole-process percentage
to graph reuse. These are historical measurements, not a benchmark rerun of this
export or an arbitrary-video latency guarantee. Source weights, arithmetic,
resolution and frame count were held fixed in that experiment.

This export has clean pinned-base patch-application verification and CPU fixture
coverage. No new GPU qualification, broad visual/audio-quality study, other-model
speedup or one-second fresh-video result is claimed. Private receipts, logs,
prompts and generated media are intentionally outside this source package.

To remove the changes on an otherwise unchanged patched checkout:

```powershell
git -C "$RUNTIME" apply --reverse --check "$BACKEND/patches/engine.patch"
git -C "$RUNTIME" apply --reverse "$BACKEND/patches/engine.patch"
```

Omitting the optional flags keeps graph/cache/retention opt-ins off for the runner.
