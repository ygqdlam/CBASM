#!/usr/bin/env python3
"""Run the corrected passive feature-similarity trainings for Fig. 5.

CBASM and AD-MT are launched concurrently on two separate GPUs.  The only
training-code change is the feature-similarity monitor: networks are temporarily
placed in eval mode under inference_mode(), and BatchNorm buffers are checked.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parent
ORIG_REPO = Path("/home/ygq/cq/code/AD-MT-revision")
BASE_CFG = ORIG_REPO / "cfgs" / "config_3d_la_aut.yml"
OUT_ROOT = Path("/home/ygq/cq/code0915/results_fig5_passive")
CFG_DIR = OUT_ROOT / "cfgs"
LOG_DIR = OUT_ROOT / "launcher_logs"
DATA_PATH = Path("/home/ygq/cq/dataset/LA/Left_Atrium/data")
ROOT_PATH = ORIG_REPO / "data_split" / "LA"

SEED = int(os.environ.get("FIG5_SEED", "2023"))
LABELED_NUM = int(os.environ.get("FIG5_LABELED_NUM", "4"))
MAX_ITERATIONS = int(os.environ.get("FIG5_MAX_ITERATIONS", "25000"))
TEST_INTERVAL_EP = int(os.environ.get("FIG5_TEST_INTERVAL_EP", "4"))

RUNS = [
    {
        "method": "CBASM",
        "exp": f"fig5_passive_cbasm_seed{SEED}",
        "script": "train_post_3d_aut_addema.py",
        "gpu": os.environ.get("FIG5_CBASM_GPU", "0"),
        "run_tag": "Fig5_passive_CBA_SM_eval_mode_no_buffer_update",
    },
    {
        "method": "AD-MT",
        "exp": f"fig5_passive_admt_seed{SEED}",
        "script": "train_post_3d_aut.py",
        "gpu": os.environ.get("FIG5_ADMT_GPU", "1"),
        "run_tag": "Fig5_passive_ADMT_eval_mode_no_buffer_update",
    },
]


def log(message: str) -> None:
    print(f"[{datetime.now():%F %T}] {message}", flush=True)


def write_cfg(run: dict) -> Path:
    with BASE_CFG.open("r") as handle:
        cfg = yaml.safe_load(handle)
    cfg.update(
        {
            "root_path": str(ROOT_PATH),
            "data_path": str(DATA_PATH),
            "res_path": str(OUT_ROOT / "LA"),
            "seed": SEED,
            "labeled_num": LABELED_NUM,
            "max_iterations": MAX_ITERATIONS,
            "test_interval_ep": TEST_INTERVAL_EP,
            "gpu_id": [int(run["gpu"])],
        }
    )
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_DIR / f"la_{run['exp']}.yml"
    with cfg_path.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg_path


def launch(run: dict) -> subprocess.Popen:
    cfg_path = write_cfg(run)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run['exp']}_gpu{run['gpu']}.log"
    cmd = [
        sys.executable,
        run["script"],
        "--cfg",
        str(cfg_path),
        "--exp",
        run["exp"],
        "--model",
        "vnet",
        "--res_path",
        str(OUT_ROOT / "LA"),
        "--run_tag",
        run["run_tag"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(run["gpu"])
    env.setdefault("MPLCONFIGDIR", str(OUT_ROOT / ".matplotlib"))
    log(f"START {run['method']} on GPU {run['gpu']}: {' '.join(cmd)}")
    log_file = log_path.open("a")
    log_file.write(f"\n===== {datetime.now():%F %T} START {run['method']} =====\n")
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(CODE_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    proc._codex_log_file = log_file  # keep file handle alive until wait
    proc._codex_log_path = log_path
    proc._codex_run = run
    return proc


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    procs = [launch(run) for run in RUNS]
    failed = []
    for proc in procs:
        code = proc.wait()
        proc._codex_log_file.close()
        run = proc._codex_run
        if code == 0:
            log(f"DONE {run['method']} ({run['exp']})")
        else:
            failed.append((run, code, proc._codex_log_path))
            log(f"FAIL {run['method']} ({run['exp']}), returncode={code}, log={proc._codex_log_path}")
    if failed:
        raise SystemExit(1)
    log("Both Fig.5 passive trainings completed.")
    log(f"Results root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
