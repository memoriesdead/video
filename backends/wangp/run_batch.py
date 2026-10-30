"""Finite local WanGP batches against the pinned patched backend; no setup downloads.

Modified 2026-10-30. Integration uses the pinned WanGP session API and preserves
its WanGP Community License 2.0 terms; see LICENSE.txt and MODIFICATIONS.md.
"""
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
RESIDENCY_ENV = "WANGP_LOCAL_BATCH_GPU_RESIDENCY"
H3_MODEL = "minimax_h3_fasth3_omni_nvfp4"
_RUN_LOCK = threading.Lock()


@contextmanager
def serial_session():
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("Only one WanGP session may run in this process")
    try:
        yield
    finally:
        _RUN_LOCK.release()


@contextmanager
def local_environment(graphs=False, image_cache=False):
    values = {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "GRADIO_ANALYTICS_ENABLED": "False",
        "DO_NOT_TRACK": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "RAYON_NUM_THREADS": "4",
        "TORCHINDUCTOR_COMPILE_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
        "WGP_H3_VSA_GRAPHS": "1" if graphs else "0",
        "WGP_H3_IMAGE_LATENT_CACHE": "1" if image_cache else "0",
        "WGP_VSA_KERNEL": "triton" if graphs else None,
        "WGP_VSA_PV_ACCUM": "fp32" if graphs else None,
        RESIDENCY_ENV: None,
    }
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_tasks(path):
    tasks = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest must be a nonempty list of tasks")
    ids = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("params"), dict):
            raise ValueError("Each task needs id and params")
        task_id = str(task.get("id", "")).strip()
        if not task_id or task_id in ids:
            raise ValueError("Task ids must be nonempty and unique")
        ids.add(task_id)
    return tasks


def validate_retention(tasks, keep, graphs):
    if not keep:
        return
    if not graphs or any(task["params"].get("model_type") != H3_MODEL for task in tasks):
        raise ValueError("Retention requires explicit H3 VSA graphs and the supported H3 backend")
    first = {k: v for k, v in tasks[0]["params"].items() if k not in ("seed", "client_id")}
    for task in tasks:
        params = task["params"]
        if {k: v for k, v in params.items() if k not in ("seed", "client_id")} != first:
            raise ValueError("Retained tasks may differ only in seed and client_id")
        try:
            call = ast.parse("f(" + params.get("overrides", "") + ")", mode="eval").body
            if call.args or any(key.arg is None for key in call.keywords):
                raise ValueError
            options = {key.arg: ast.literal_eval(key.value) for key in call.keywords}
            if len(options) != len(call.keywords):
                raise ValueError
        except (SyntaxError, ValueError, TypeError):
            raise ValueError("Retention requires literal generation overrides") from None
        if (params.get("config") != "nvfp4_awq,int8_convrot"
                or options.get("memory_profile") != 0 or options.get("compilation") != "cudagraph"
                or options.get("attention_mode", "vsa") != "vsa"
                or params.get("num_inference_steps") != 8
                or params.get("sample_solver", "euler") != "euler"
                or params.get("guidance_phases", 1) != 1 or params.get("image_mode", 0) != 0):
            raise ValueError("Unsupported retained H3 generation settings")


def verify_runtime(runtime):
    metadata = json.loads((HERE / "backend.json").read_text(encoding="utf-8"))
    revision = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    if revision != metadata["revision"]:
        raise ValueError("WanGP revision differs from this backend pin")
    for relative, expected in metadata["patched_source_sha256"].items():
        source = (runtime / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
            raise ValueError("Patched runtime source differs: " + relative)


def cleanup(session, runtime, job, keep):
    os.environ.pop(RESIDENCY_ENV, None)
    if runtime is None:
        return  # close() initializes the runtime; never retry failed initialization.
    try:
        if job is not None and not job.done:
            job.cancel()
            job.result()  # Join before unloading resources used by the worker.
        if keep and runtime.module.offloadobj is not None:
            runtime.module.offloadobj.unload_all()
    finally:
        session.close()  # Upstream releases under its generation lock, including failures.


def run_session(session, entries, manifest_base, *, dry_run=False, keep=False):
    runtime = job = None
    try:
        runtime = session._ensure_runtime()
        if runtime.module.wan_model is not None:
            raise RuntimeError("Batch must start with no model loaded")
        tasks = session._normalize_source(copy.deepcopy(entries), caller_base_path=manifest_base)
        session._ensure_task_client_ids(tasks)
        errors = []
        for task in tasks:
            try:
                valid, error = runtime.module.validate_task(copy.deepcopy(task), session._state)
                if valid is None:
                    errors.append(str(error))
            except Exception as error:
                errors.append(str(error))
        if runtime.module.wan_model is not None:
            raise RuntimeError("Validation unexpectedly loaded a model")
        if errors:
            raise ValueError("Task validation failed: " + "; ".join(errors))
        if dry_run:
            return None
        if keep:
            os.environ[RESIDENCY_ENV] = "1"
        job = session.submit_manifest(tasks)
        for event in job.events.iter(timeout=0.2):
            if event.kind == "error":
                os.environ.pop(RESIDENCY_ENV, None)
                job.cancel()
        return job.result()
    finally:
        cleanup(session, runtime, job, keep)


def local_ffmpeg_setup(directory):
    """Replace upstream auto-download setup with existing local binaries only."""
    names = ["ffmpeg", "ffprobe"] + (["ffplay"] if os.name == "nt" else [])
    binaries = {name: directory / (name + ".exe" if os.name == "nt" else name) for name in names}
    if any(not path.is_file() for path in binaries.values()):
        raise FileNotFoundError("Install the local FFmpeg binaries first; no download attempted")
    handles = []

    def configure(*args, **kwargs):
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        for name, path in binaries.items():
            os.environ[name.upper() + "_BINARY"] = str(path)
        if os.name == "nt" and hasattr(os, "add_dll_directory") and not handles:
            handles.append(os.add_dll_directory(str(directory)))

    return configure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--image-latent-cache", action="store_true")
    parser.add_argument("--h3-vsa-graphs", action="store_true")
    parser.add_argument("--keep-gpu-resident", action="store_true")
    args = parser.parse_args(argv)
    runtime, config, manifest, output = [p.resolve() for p in
                                         (args.runtime, args.config, args.manifest, args.output)]
    entries = load_tasks(manifest)
    validate_retention(entries, args.keep_gpu_resident, args.h3_vsa_graphs)
    settings = json.loads(config.read_text(encoding="utf-8-sig"))
    if config.name != "wgp_config.json" or any(
            flag in settings.get("preload_model_policy", []) for flag in ("P", "U")):
        raise ValueError("Use wgp_config.json with startup preloading and automatic unloading disabled")
    if output.exists():
        raise ValueError("Output directory must be new")
    verify_runtime(runtime)
    setup_ffmpeg = local_ffmpeg_setup((args.ffmpeg_dir or runtime / "ffmpeg_bins").resolve())
    with serial_session(), local_environment(args.h3_vsa_graphs, args.image_latent_cache):
        sys.path.insert(0, str(runtime))
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        setup = importlib.import_module("shared.ffmpeg_setup")
        if Path(setup.__file__).resolve() != runtime / "shared/ffmpeg_setup.py":
            raise RuntimeError("A different WanGP backend is already imported")
        setup.download_ffmpeg = setup_ffmpeg
        api = importlib.import_module("shared.api")
        session = api.WanGPSession(root=runtime, config_path=config, output_dir=output,
            cli_args=["--profile", "0", "--offline", "--no-mcp-webui"], console_output=False)
        result = run_session(session, entries, manifest.parent, dry_run=args.dry_run,
                             keep=args.keep_gpu_resident)
    if result is None:
        print("Validated all tasks; no model weights loaded and no rendering started")
        return 0
    print(json.dumps({"success": result.success, "total_tasks": result.total_tasks,
                      "successful_tasks": result.successful_tasks, "failed_tasks": result.failed_tasks,
                      "generated_files": result.generated_files}))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
