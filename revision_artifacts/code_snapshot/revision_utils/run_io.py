import json
import os
import shutil


def resolve_config_path(cfg_name, script_file=None):
    """Find a config file from common local/server launch locations."""
    if os.path.isabs(cfg_name) and os.path.exists(cfg_name):
        return cfg_name

    script_dir = os.path.dirname(os.path.abspath(script_file or __file__))
    project_dir = os.path.dirname(script_dir)
    cwd = os.getcwd()
    candidates = [
        cfg_name,
        os.path.join(cwd, cfg_name),
        os.path.join(cwd, "cfgs", cfg_name),
        os.path.join(script_dir, cfg_name),
        os.path.join(script_dir, "cfgs", cfg_name),
        os.path.join(project_dir, "cfgs", cfg_name),
        os.path.join(os.path.dirname(cwd), "cfgs", cfg_name),
    ]

    seen = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.append(candidate)
        if os.path.exists(candidate):
            return candidate

    message = "Cannot find config file '{}'. Tried:\n{}".format(
        cfg_name, "\n".join("  - {}".format(path) for path in seen)
    )
    raise FileNotFoundError(message)


def copy_sources(snapshot_path, source_paths):
    """Copy source files into the run folder when they exist."""
    os.makedirs(snapshot_path, exist_ok=True)
    copied = []
    skipped = []
    for source_path in source_paths:
        if source_path and os.path.exists(source_path):
            shutil.copy(source_path, snapshot_path)
            copied.append(source_path)
        else:
            skipped.append(source_path)
    return {"copied": copied, "skipped": skipped}


def write_run_metadata(snapshot_path, args, outputs=None, source_snapshot=None):
    os.makedirs(snapshot_path, exist_ok=True)
    meta_path = os.path.join(snapshot_path, "run_meta.json")
    metadata = {
        "args": args,
        "purpose": "Major-revision run metadata for reproducibility and reviewer response.",
        "outputs": outputs or {},
        "source_snapshot": source_snapshot or {},
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return meta_path
