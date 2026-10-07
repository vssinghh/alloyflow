#!/usr/bin/env bash
set -euo pipefail

# Unified AlloyFlow training launcher: runs on either Colab GPU (--colab) or
# local Mac MPS (--local), and always leaves checkpoints in checkpoints/<run_name>/
# so standard local evaluation (`uv run python -m evaluation ...`) works directly.
#
# Usage:
#   ./scripts/train.sh <run_name> [--colab | --local] [python -m training args...]
#
# Examples:
#   ./scripts/train.sh exp08_dart_full_900 --colab --mode sim_only
#   ./scripts/train.sh exp08_dart_full_900 --local --mode sim_only
#   ./scripts/train.sh exp08_smoke --colab --mode sim_only --epochs 5

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <run_name> [--colab | --local] [python -m training args...]"
    echo ""
    echo "Targets:"
    echo "  --colab    Train on a remote Google Colab GPU and pull checkpoints locally (default)"
    echo "  --local    Train on local Mac GPU (MPS) into checkpoints/<run_name>/"
    echo ""
    echo "Examples:"
    echo "  $0 exp08_dart_full_900 --colab --mode sim_only"
    echo "  $0 exp08_dart_full_900 --local --mode sim_only --epochs 40"
    exit 0
fi

RUN_NAME="$1"
shift

TARGET="colab"
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --colab)
            TARGET="colab"
            shift
            ;;
        --local)
            TARGET="local"
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
    TRAIN_ARGS=(--mode sim_only)
fi

if [[ "$TARGET" != "colab" && "$TARGET" != "local" ]]; then
    echo "[train] Error: Invalid target '$TARGET'. Must be 'colab' or 'local'."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCAL_SAVE_DIR="$REPO_ROOT/checkpoints/$RUN_NAME"
CREATED_NEW_DIR=0
if [[ ! -d "$LOCAL_SAVE_DIR" ]]; then
    CREATED_NEW_DIR=1
fi
mkdir -p "$LOCAL_SAVE_DIR"

# 1. Snapshot local code state (including uncommitted edits) into checkpoints/<run_name>/
echo "[train] Saving code snapshot to $LOCAL_SAVE_DIR/code_snapshot.tgz"
{
    echo "run_name: $RUN_NAME"
    echo "target: $TARGET"
    echo "timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo ""
    echo "=== git status ==="
    git status --short 2>/dev/null || true
    echo ""
    echo "=== git diff HEAD ==="
    git diff HEAD 2>/dev/null || true
} > "$LOCAL_SAVE_DIR/CODE_VERSION.txt"

tar czf "$LOCAL_SAVE_DIR/code_snapshot.tgz" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    pyproject.toml README.md DESIGN.md EXPERIMENT_LOG.md \
    assets/so101/scene.xml assets/so101/so101_arm.xml \
    envs collection training evaluation

# ==============================================================================
# LOCAL TARGET (Mac MPS / CPU)
# ==============================================================================
if [[ "$TARGET" == "local" ]]; then
    echo "[train] Running locally on Mac into checkpoints/$RUN_NAME..."
    uv run python -u -m training "${TRAIN_ARGS[@]}" --save-dir "checkpoints/$RUN_NAME" 2>&1 | tee "$LOCAL_SAVE_DIR/train.log"
    echo ""
    echo "[train] Done. Checkpoints saved to $LOCAL_SAVE_DIR:"
    ls -lh "$LOCAL_SAVE_DIR"
    echo ""
    echo "[train] Run evaluation with:"
    echo "  uv run python -m evaluation --checkpoint checkpoints/$RUN_NAME/best_policy.pt --benchmark --episodes 20"
    exit 0
fi

# ==============================================================================
# COLAB TARGET (Remote GPU -> Auto-Download to checkpoints/<run_name>/)
# ==============================================================================
GPU_TYPE="${COLAB_GPU:-T4}"
KEEP_VM="${COLAB_KEEP_VM:-0}"
DRIVE_FOLDER_ID="${ALLOYFLOW_DRIVE_FOLDER_ID:-1W7jPSb39ncSq3AF9a9YXTNI8PYQEGhCS}"
GDRIVE_BIN="$HOME/.gemini/config/plugins/gdrive/skills/gdrive/scripts/gdrive"
GDRIVE_TOKEN_JSON="$HOME/.gemini/antigravity/plugin_data/gdrive/token.json"

SESSION_CLEAN="$(echo "af-${RUN_NAME}" | tr -cd '[:alnum:]-' | cut -c1-20)"
REMOTE_CKPT_DIR="/content/alloyflow/checkpoints/$RUN_NAME"
SESSION_CREATED=0
TRAINING_STARTED=0
DOWNLOADED_FINAL=0

pull_remote_checkpoints() {
    echo "[train] Pulling checkpoints from Colab VM into $LOCAL_SAVE_DIR..."
    colab download -s "$SESSION_CLEAN" /content/train.log "$LOCAL_SAVE_DIR/train.log" >/dev/null 2>&1 || true
    for fname in train_config.json training_history.json best_policy.pt latest.pt latest_policy.pt pretrained_vision.pt epoch_010.pt epoch_020.pt epoch_030.pt epoch_040.pt; do
        colab download -s "$SESSION_CLEAN" "$REMOTE_CKPT_DIR/$fname" "$LOCAL_SAVE_DIR/$fname" >/dev/null 2>&1 || true
    done
}

print_local_fallback_hint() {
    echo ""
    echo "[train] Error: Colab GPU ($GPU_TYPE) is unavailable right now."
    echo "[train] You can attempt to train locally on your Mac GPU (MPS) with the --local flag:"
    echo "  ./scripts/train.sh $RUN_NAME --local ${TRAIN_ARGS[*]}"
}

cleanup() {
    rm -f "$LOCAL_SAVE_DIR/.remote_train.log" "$LOCAL_SAVE_DIR/.remote_train.exit"
    if [[ "$TRAINING_STARTED" == "1" && "$DOWNLOADED_FINAL" == "0" ]]; then
        pull_remote_checkpoints || true
    fi
    if [[ "$SESSION_CREATED" == "1" ]]; then
        if [[ "$KEEP_VM" == "1" ]]; then
            echo "[train] COLAB_KEEP_VM=1 set; keeping Colab session '$SESSION_CLEAN' alive."
        else
            echo "[train] Stopping Colab session '$SESSION_CLEAN'..."
            colab stop -s "$SESSION_CLEAN" >/dev/null 2>&1 || true
        fi
    fi
    if [[ "$TRAINING_STARTED" == "0" && "$CREATED_NEW_DIR" == "1" && -d "$LOCAL_SAVE_DIR" ]]; then
        rm -rf "$LOCAL_SAVE_DIR"
    fi
}
trap cleanup EXIT INT TERM

if [[ -x "$GDRIVE_BIN" ]]; then
    "$GDRIVE_BIN" readonly ls "$DRIVE_FOLDER_ID" >/dev/null 2>&1 || true
fi

if [[ ! -f "$GDRIVE_TOKEN_JSON" ]]; then
    echo "[train] Error: Drive token file not found at $GDRIVE_TOKEN_JSON"
    exit 1
fi

DRIVE_ACCESS_TOKEN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["token"]["access_token"])' "$GDRIVE_TOKEN_JSON")"

# Inspect TRAIN_ARGS for dataset files (--sim-data, --real-data), --pretrained-checkpoint, and --resume
PARSED_META="$(python3 - "${TRAIN_ARGS[@]}" <<'PY'
import argparse
import json
import os
import sys

p = argparse.ArgumentParser(add_help=False)
p.add_argument("--mode", default="sim_only")
p.add_argument("--sim-data", default="data/sim_demos_v2_dart_full_900.h5")
p.add_argument("--real-data", default="data/real_demos.h5")
p.add_argument("--pretrained-checkpoint", default="")
p.add_argument("--resume", action="store_true")
p.add_argument("--resume-from", default="")
args, _ = p.parse_known_args(sys.argv[1:])

req_data = ["data/loc_layouts_3000.npz"]
if args.mode in ("sim_only", "cotrain"):
    req_data.append(args.sim_data)
if args.mode in ("real_only", "finetune", "cotrain"):
    req_data.append(args.real_data)

print(json.dumps({
    "mode": args.mode,
    "required_data": req_data,
    "required_basenames": [os.path.basename(x) for x in req_data],
    "pretrained_checkpoint": args.pretrained_checkpoint,
    "resume": bool(args.resume),
    "resume_from": args.resume_from,
}))
PY
)"

PRETRAINED_CKPT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["pretrained_checkpoint"])' "$PARSED_META")"
RESUME_FLAG="$(python3 -c 'import json,sys; print("1" if json.loads(sys.argv[1])["resume"] else "0")' "$PARSED_META")"
RESUME_FROM_CKPT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["resume_from"])' "$PARSED_META")"
REQUIRED_BASENAMES_JSON="$(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1])["required_basenames"]))' "$PARSED_META")"

if [[ -n "$PRETRAINED_CKPT" && ! -f "$PRETRAINED_CKPT" ]]; then
    echo "[train] Error: --pretrained-checkpoint file not found locally: $PRETRAINED_CKPT"
    exit 1
fi

# Ensure all required dataset files for this run exist in the Google Drive folder
python3 - "$DRIVE_FOLDER_ID" "$DRIVE_ACCESS_TOKEN" "$GDRIVE_BIN" "$PARSED_META" <<'PY'
import json
import os
import subprocess
import sys
import urllib.request

folder_id, token, gdrive_bin, meta_json = sys.argv[1:5]
meta = json.loads(meta_json)

list_url = (
    f"https://www.googleapis.com/drive/v3/files"
    f"?q='{folder_id}'+in+parents+and+trashed=false"
    f"&fields=files(id,name,size)"
)
req = urllib.request.Request(list_url, headers={"Authorization": f"Bearer {token}"})
with urllib.request.urlopen(req) as resp:
    drive_files = {f["name"]: int(f.get("size", 0)) for f in json.loads(resp.read().decode("utf-8")).get("files", [])}

for rel_path in meta["required_data"]:
    bname = os.path.basename(rel_path)
    if bname in drive_files:
        continue
    if not os.path.isfile(rel_path):
        raise SystemExit(f"[train] Error: Required dataset file '{rel_path}' not found locally or in Drive.")
    print(f"[train] Uploading new dataset file '{rel_path}' to Google Drive folder...", flush=True)
    subprocess.run([gdrive_bin, "mutate", "upload", rel_path, "--parent", folder_id], check=True)
PY

echo "[train] Provisioning Colab session '$SESSION_CLEAN' with GPU=$GPU_TYPE..."
if ! colab new -s "$SESSION_CLEAN" --gpu "$GPU_TYPE"; then
    print_local_fallback_hint
    exit 1
fi
SESSION_CREATED=1

echo "[train] Uploading local code snapshot and Drive auth to Colab VM..."
colab upload -s "$SESSION_CLEAN" "$LOCAL_SAVE_DIR/code_snapshot.tgz" /content/code_snapshot.tgz
colab upload -s "$SESSION_CLEAN" "$GDRIVE_TOKEN_JSON" /content/.gdrive_auth.json

if [[ -n "$PRETRAINED_CKPT" ]]; then
    echo "[train] Uploading pretrained checkpoint ($PRETRAINED_CKPT) to Colab VM..."
    colab upload -s "$SESSION_CLEAN" "$PRETRAINED_CKPT" /content/pretrained_input.pt
fi

if [[ -n "$RESUME_FROM_CKPT" && -f "$RESUME_FROM_CKPT" ]]; then
    echo "[train] Uploading resume checkpoint ($RESUME_FROM_CKPT) to Colab VM..."
    colab upload -s "$SESSION_CLEAN" "$RESUME_FROM_CKPT" /content/resume_input.pt
elif [[ "$RESUME_FLAG" == "1" && -f "$LOCAL_SAVE_DIR/latest.pt" ]]; then
    echo "[train] Uploading local latest.pt for resume to Colab VM..."
    colab upload -s "$SESSION_CLEAN" "$LOCAL_SAVE_DIR/latest.pt" /content/resume_input.pt
    if [[ -f "$LOCAL_SAVE_DIR/best_policy.pt" ]]; then
        colab upload -s "$SESSION_CLEAN" "$LOCAL_SAVE_DIR/best_policy.pt" /content/resume_best_input.pt
    fi
    if [[ -f "$LOCAL_SAVE_DIR/training_history.json" ]]; then
        colab upload -s "$SESSION_CLEAN" "$LOCAL_SAVE_DIR/training_history.json" /content/resume_history_input.json
    fi
fi

TRAIN_ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${TRAIN_ARGS[@]}")"

echo "[train] Bootstrapping VM environment and downloading dataset..."
if ! colab exec -s "$SESSION_CLEAN" --timeout 600 <<EOF
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this Colab runtime.")
print(f"[VM] GPU verified: {torch.cuda.get_device_name(0)}", flush=True)

repo_dir = "/content/alloyflow"
if not os.path.exists(repo_dir):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/vssinghh/alloyflow.git", repo_dir],
        check=True,
    )

for pkg in ("envs", "collection", "training", "evaluation"):
    pkg_path = os.path.join(repo_dir, pkg)
    if os.path.exists(pkg_path):
        shutil.rmtree(pkg_path)

with tarfile.open("/content/code_snapshot.tgz", "r:gz") as tar:
    tar.extractall(repo_dir)
print("[VM] Applied local code snapshot.", flush=True)

pretrained_rel = "$PRETRAINED_CKPT"
if pretrained_rel and os.path.exists("/content/pretrained_input.pt"):
    dest_ckpt = pretrained_rel if os.path.isabs(pretrained_rel) else os.path.join(repo_dir, pretrained_rel)
    os.makedirs(os.path.dirname(dest_ckpt), exist_ok=True)
    shutil.move("/content/pretrained_input.pt", dest_ckpt)
    print(f"[VM] Placed pretrained checkpoint at {dest_ckpt}", flush=True)

remote_ckpt_dir = os.path.join(repo_dir, "checkpoints", "$RUN_NAME")
os.makedirs(remote_ckpt_dir, exist_ok=True)

resume_from_rel = "$RESUME_FROM_CKPT"
resume_flag = "$RESUME_FLAG"
folder_id = "$DRIVE_FOLDER_ID"
token = "$DRIVE_ACCESS_TOKEN"

if os.path.exists("/content/resume_input.pt"):
    if resume_from_rel:
        dest_res = resume_from_rel if os.path.isabs(resume_from_rel) else os.path.join(repo_dir, resume_from_rel)
        os.makedirs(os.path.dirname(dest_res), exist_ok=True)
        shutil.move("/content/resume_input.pt", dest_res)
    else:
        shutil.move("/content/resume_input.pt", os.path.join(remote_ckpt_dir, "latest.pt"))
        shutil.copyfile(os.path.join(remote_ckpt_dir, "latest.pt"), os.path.join(remote_ckpt_dir, "latest_policy.pt"))
    if os.path.exists("/content/resume_best_input.pt"):
        shutil.move("/content/resume_best_input.pt", os.path.join(remote_ckpt_dir, "best_policy.pt"))
    if os.path.exists("/content/resume_history_input.json"):
        shutil.move("/content/resume_history_input.json", os.path.join(remote_ckpt_dir, "training_history.json"))
    print(f"[VM] Placed local resume checkpoint in {remote_ckpt_dir}", flush=True)
elif resume_flag == "1":
    # Fallback: fetch latest.pt from Google Drive ckpt_<run_name> folder if not uploaded from local
    q_dir = (
        f"'{folder_id}' in parents and name = 'ckpt_${RUN_NAME}' "
        f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    u_dir = f"https://www.googleapis.com/drive/v3/files?{urllib.parse.urlencode({'q': q_dir, 'fields': 'files(id,name)'})}"
    req_d = urllib.request.Request(u_dir, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req_d) as resp_d:
        d_folders = json.loads(resp_d.read().decode("utf-8")).get("files", [])
    if d_folders:
        sub_id = d_folders[0]["id"]
        q_files = f"'{sub_id}' in parents and trashed = false"
        u_files = f"https://www.googleapis.com/drive/v3/files?{urllib.parse.urlencode({'q': q_files, 'fields': 'files(id,name)'})}"
        req_f = urllib.request.Request(u_files, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req_f) as resp_f:
            for f_item in json.loads(resp_f.read().decode("utf-8")).get("files", []):
                if f_item["name"] in ("latest.pt", "best_policy.pt", "training_history.json"):
                    dl_u = f"https://www.googleapis.com/drive/v3/files/{f_item['id']}?alt=media"
                    dl_r = urllib.request.Request(dl_u, headers={"Authorization": f"Bearer {token}"})
                    dest_f = os.path.join(remote_ckpt_dir, f_item["name"])
                    with urllib.request.urlopen(dl_r) as rr, open(dest_f, "wb") as out_f:
                        shutil.copyfileobj(rr, out_f)
                    print(f"[VM] Pulled {f_item['name']} from Drive folder ckpt_${RUN_NAME}", flush=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "mujoco>=3.1.0"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", repo_dir],
    check=True,
)
print("[VM] Installed alloyflow and mujoco.", flush=True)

required_basenames = set(json.loads('''$REQUIRED_BASENAMES_JSON'''))
data_dir = os.path.join(repo_dir, "data")
os.makedirs(data_dir, exist_ok=True)

list_url = (
    f"https://www.googleapis.com/drive/v3/files"
    f"?q='{folder_id}'+in+parents+and+trashed=false"
    f"&fields=files(id,name,size)"
)
req = urllib.request.Request(list_url, headers={"Authorization": f"Bearer {token}"})
with urllib.request.urlopen(req) as resp:
    drive_files = json.loads(resp.read().decode("utf-8")).get("files", [])

if not drive_files:
    raise RuntimeError(f"No dataset files found in Drive folder {folder_id}")

for df in drive_files:
    fname = df["name"]
    if fname not in required_basenames:
        continue
    fid = df["id"]
    expected_size = int(df.get("size", 0))
    dest = os.path.join(data_dir, fname)
    if os.path.exists(dest) and expected_size > 0 and os.path.getsize(dest) == expected_size:
        print(f"[VM] Dataset file {fname} already present ({expected_size} bytes).", flush=True)
        continue
    print(f"[VM] Downloading {fname} ({expected_size / 1e6:.1f} MB) from Drive...", flush=True)
    dl_url = f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media"
    dl_req = urllib.request.Request(dl_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(dl_req) as r, open(dest, "wb") as out_f:
        shutil.copyfileobj(r, out_f, length=16 * 1024 * 1024)
    print(f"[VM] Finished downloading {fname}.", flush=True)

for p in ("/content/train.log", "/content/train.exit", "/content/train.pid"):
    if os.path.exists(p):
        os.remove(p)

train_args = json.loads('''$TRAIN_ARGS_JSON''')
remote_save_dir = f"checkpoints/${RUN_NAME}"
cmd = [
    sys.executable,
    "-u",
    "-m",
    "training",
    *train_args,
    "--save-dir",
    remote_save_dir,
]
runner_code = (
    "import os, subprocess, sys\n"
    f"os.chdir({repo_dir!r})\n"
    "os.environ['MUJOCO_GL'] = 'egl'\n"
    f"os.environ['ALLOYFLOW_DRIVE_FOLDER_ID'] = {folder_id!r}\n"
    "os.environ['ALLOYFLOW_GDRIVE_AUTH_JSON'] = '/content/.gdrive_auth.json'\n"
    "with open('/content/train.log', 'w') as logf:\n"
    f"    proc = subprocess.Popen({cmd!r}, stdout=logf, stderr=subprocess.STDOUT)\n"
    "    rc = proc.wait()\n"
    "with open('/content/train.exit', 'w') as ef:\n"
    "    ef.write(str(rc))\n"
)
with open("/content/run_training_bg.py", "w") as rf:
    rf.write(runner_code)

bg_proc = subprocess.Popen(
    [sys.executable, "/content/run_training_bg.py"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
    start_new_session=True,
)
with open("/content/train.pid", "w") as pf:
    pf.write(str(bg_proc.pid))
print(f"[VM] Started background training (PID={bg_proc.pid}): {' '.join(cmd)}", flush=True)
EOF
then
    print_local_fallback_hint
    exit 1
fi
TRAINING_STARTED=1

echo "[train] Streaming remote training logs..."
PRINTED_LINES=0
TMP_LOG="$LOCAL_SAVE_DIR/.remote_train.log"
TMP_EXIT="$LOCAL_SAVE_DIR/.remote_train.exit"
rm -f "$TMP_LOG" "$TMP_EXIT"

FAIL_COUNT=0
POLL_ITER=0
while true; do
    POLL_ITER=$((POLL_ITER + 1))
    # Refresh Colab tunnel assignment and proxy token every 6th poll (~30s)
    if (( POLL_ITER % 6 == 0 )); then
        colab status -s "$SESSION_CLEAN" >/dev/null 2>&1 || true
    fi
    if colab download -s "$SESSION_CLEAN" /content/train.log "$TMP_LOG" >/dev/null 2>&1; then
        FAIL_COUNT=0
        TOTAL_LINES="$(wc -l < "$TMP_LOG" | tr -d ' ')"
        if [[ "$TOTAL_LINES" -gt "$PRINTED_LINES" ]]; then
            tail -n +"$((PRINTED_LINES + 1))" "$TMP_LOG"
            PRINTED_LINES="$TOTAL_LINES"
            cp "$TMP_LOG" "$LOCAL_SAVE_DIR/train.log"
            colab download -s "$SESSION_CLEAN" "$REMOTE_CKPT_DIR/latest.pt" "$LOCAL_SAVE_DIR/latest.pt" >/dev/null 2>&1 || true
            colab download -s "$SESSION_CLEAN" "$REMOTE_CKPT_DIR/best_policy.pt" "$LOCAL_SAVE_DIR/best_policy.pt" >/dev/null 2>&1 || true
            colab download -s "$SESSION_CLEAN" "$REMOTE_CKPT_DIR/training_history.json" "$LOCAL_SAVE_DIR/training_history.json" >/dev/null 2>&1 || true
            for snap in epoch_010.pt epoch_020.pt epoch_030.pt epoch_040.pt; do
                if [[ ! -f "$LOCAL_SAVE_DIR/$snap" ]]; then
                    colab download -s "$SESSION_CLEAN" "$REMOTE_CKPT_DIR/$snap" "$LOCAL_SAVE_DIR/$snap" >/dev/null 2>&1 || true
                fi
            done
        fi
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if (( FAIL_COUNT >= 4 )); then
            echo "[train] Error: Lost connection to Colab session '$SESSION_CLEAN' after $FAIL_COUNT failed polls."
            exit 1
        fi
    fi
    if colab download -s "$SESSION_CLEAN" /content/train.exit "$TMP_EXIT" >/dev/null 2>&1; then
        if colab download -s "$SESSION_CLEAN" /content/train.log "$TMP_LOG" >/dev/null 2>&1; then
            TOTAL_LINES="$(wc -l < "$TMP_LOG" | tr -d ' ')"
            if [[ "$TOTAL_LINES" -gt "$PRINTED_LINES" ]]; then
                tail -n +"$((PRINTED_LINES + 1))" "$TMP_LOG"
                PRINTED_LINES="$TOTAL_LINES"
            fi
            cp "$TMP_LOG" "$LOCAL_SAVE_DIR/train.log"
        fi
        EXIT_STATUS="$(tr -d '[:space:]' < "$TMP_EXIT")"
        rm -f "$TMP_LOG" "$TMP_EXIT"
        if [[ "$EXIT_STATUS" != "0" ]]; then
            echo "[train] Remote training exited with status $EXIT_STATUS"
            exit "$EXIT_STATUS"
        fi
        break
    fi
    # Sleep 5s inside the remote Jupyter kernel via stdin pipe so the kernel WebSocket stays open and busy
    echo "import time; time.sleep(5)" | colab exec -s "$SESSION_CLEAN" --timeout 30 >/dev/null 2>&1 || sleep 5
done

pull_remote_checkpoints
DOWNLOADED_FINAL=1

echo "[train] Done. Files in $LOCAL_SAVE_DIR:"
ls -lh "$LOCAL_SAVE_DIR"
echo ""
echo "[train] Run evaluation with:"
echo "  uv run python -m evaluation --checkpoint checkpoints/$RUN_NAME/best_policy.pt --benchmark --episodes 20"
