# How to compare local video speed

Compare the same finished clip. Changing the model, resolution, frame count,
steps or audio requirement changes the job.

## Record the setup

- GPU, RAM, driver, runtime version and engine commit.
- DiT, VAE and text/vision encoder revisions, checksums and precision.
- Width, height, frame count, FPS, steps, solver, guidance and seed.
- Audio settings, interpolation, upscaling, previews and final encoding settings.
- Prompt/reference checksums and whether their encoded features are cached.
- Batch size, residency settings and other active workloads.

## Keep the timing boundaries separate

| Timing | What it includes |
| --- | --- |
| Process wall time | Command start through the completed run and its stated cleanup boundary |
| First task | First generation call, including its model/shape preparation, processing and saving |
| Warm task | Later generation call with the stated models/caches already loaded, including processing and saving |
| Fresh-input task | A new prompt/reference under the stated residency conditions |

A preview or first frame is not a finished video. Confirm that the complete file
decodes, has the expected frame count and duration, and includes the requested
audio. Separate these checks from visual/audio quality evaluation.

## Run and report

Keep model weight files unchanged. Validate all inputs before loading models.
Check available resources and preserve other jobs. Use repeated A/B/B/A runs,
with the same settings and seeds, to reduce order effects. Report first-task
and warm medians separately, with samples, failures and the exact timing scope.
Exclude an interrupted process wall time from completed-process comparisons.

Repeated decoded hashes can show that one setup is repeatable. They do not
prove that two different models have the same quality. A faster model or fewer
steps needs its own quality check.

Our 9.5-second reference is a warm-task result with repeated input. It is not
process startup or a measured latency for arbitrary fresh inputs. The saved
clips passed a separate media audit while the original coordinator run remained
failed. The [comparison](comparison.md) keeps published claims and our local
measurements separate. Share licensed public test media or checksums, not private
inputs, credentials or local machine paths.
