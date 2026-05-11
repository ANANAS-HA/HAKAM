# Rent → Running Runbook

End-to-end commands to take a freshly-rented GPU pod from "just booted" to "Recipe A extraction running." Optimised for copy-paste. Set the placeholders at the top once, then nothing else needs editing.

---

## Before you click "Rent"

1. **Pick a GPU.** Recommended for 27B-AWQ-4bit:
   - **Best $/perf**: Vast.ai H100 NVL (~$1.33/hr) — 94 GB VRAM, plenty of headroom
   - **Backup**: Vast A100 80 GB (~$0.67/hr) — slower but cheaper
   - **Avoid**: anything < 80 GB VRAM (won't fit 27B with activations)

2. **Storage settings on the rental form:**
   - Container disk: **50 GB** (system + venv)
   - Persistent volume: **200 GB** (model weights ~14 GB + videos ~25 GB + logits ~500 MB + room)

3. **HuggingFace token** — sign in at huggingface.co → Settings → Access Tokens → create a `read` token. Have it ready to paste; you'll need it both to download the videos and (later) to upload the logits cache.

4. **Verify SSH key** is registered with the rental provider (one-time setup, you've done it).

---

## Step 0 — Set your variables (do this once when the pod boots)

After clicking Rent, the provider gives you an SSH command like
`ssh -p 19318 root@79.117.54.182`. Pull the IP and port out and define them in your **WSL terminal**:

```bash
# === EDIT THESE 3 LINES ===
export POD_PORT=19318
export POD_IP=79.117.54.182
export HF_VIDEOS_REPO="<YOUR_USERNAME>/<YOUR_VIDEOS_REPO>"      # e.g. accy/sports-qa-videos
# ==========================

export POD_DEST="root@${POD_IP}:/workspace/Granduation Project"
export SSH_OPTS="-p ${POD_PORT} -i ~/.ssh/id_ed25519"

eval "$(ssh-agent)" && ssh-add ~/.ssh/id_ed25519
```

`HF_VIDEOS_REPO` is the dataset repo your `download_subset_videos.py` will pull from. The script just needs `--repo-id` matching this.

---

## Step 1 — One-time: install rsync on the pod

The default Vast/Runpod PyTorch image is minimal. From WSL:

```bash
ssh ${SSH_OPTS} root@${POD_IP} 'apt update && apt install -y rsync tmux'
```

`tmux` is for the long extraction run later (survives SSH drops).

---

## Step 2 — Push code + subset metadata from WSL (≈ 30 sec)

Pushes scripts, requirements, `data/subsets/`, anything small. Excludes videos (HF will provide), venv, caches, and the 7 GB `features/` directory.

```bash
rsync -avz --progress \
  --exclude='.venv' \
  --exclude='videos/' \
  --exclude='frames/' \
  --exclude='features/' \
  --exclude='logits_cache_*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.txt' \
  -e "ssh ${SSH_OPTS}" \
  "/home/accy/Granduation Project/" \
  "${POD_DEST}/"
```

Total payload: ~30–50 MB (code + 11 MB subset JSONs).

Verify on the pod side:

```bash
ssh ${SSH_OPTS} root@${POD_IP} 'ls "/workspace/Granduation Project"'
# Should list: extract_logits_subset.py, download_subset_videos.py,
#   requirements.txt, data/, _smoke_*.py, etc.
```

---

## Step 3 — Connect VS Code Remote-SSH (so you have a terminal on the pod)

In VS Code:

- F1 → **Remote-SSH: Add New SSH Host** → paste:
  `ssh -p <POD_PORT> root@<POD_IP> -i ~/.ssh/id_ed25519`
- F1 → **Remote-SSH: Connect to Host** → pick the new entry
- File → Open Folder → `/workspace/Granduation Project`
- Open a terminal (Ctrl + `)

All commands from Step 4 onward run in that pod terminal.

---

## Step 4 — Pod bootstrap (≈ 5–10 min)

```bash
cd "/workspace/Granduation Project"

# Persistent HF cache — survives Stop/Start, model weights don't re-download
export HF_HOME=/workspace/hf_cache
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

# Fresh venv on the persistent volume
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# All deps including huggingface_hub for the downloader
pip install -r requirements.txt
pip install huggingface_hub                  # in case requirements.txt didn't include it
pip install autoawq compressed-tensors       # required for 27B-AWQ-4bit

# Sanity: torch + GPU
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `torch 2.4.x+ cuda True NVIDIA H100 ...` (or whichever GPU you picked).

If `transformers` complains about torch version: `pip install --upgrade torch`.

---

## Step 5 — HuggingFace login (≈ 30 sec)

```bash
huggingface-cli login
# Paste your read token when prompted. Choose 'Y' if it asks to add it as a git credential.
```

Verify:

```bash
huggingface-cli whoami
# should print your username
```

---

## Step 6 — Download the 2080 subset videos from HF (≈ 15–30 min depending on dataset size)

```bash
python -u download_subset_videos.py \
  --repo-id "${HF_VIDEOS_REPO}" \
  --splits train val \
  --out-dir videos \
  --workers 16
```

(`HF_VIDEOS_REPO` from Step 0 must be set in your WSL env; on the pod just substitute the literal repo name, e.g. `--repo-id accy/sports-qa-videos`.)

The script will:
1. List all files in the HF repo (~10 sec)
2. Match each subset video to a repo path
3. Print `matched: 2080, missing: 0` (ideally)
4. Download in parallel into `videos/` preserving the `<sport>/<stem>.<ext>` layout

If `missing > 0`, those videos didn't exist in the HF repo. The list goes to `subset_missing_videos.txt`. You'd skip them in extraction (the extractor handles missing videos gracefully — manifest records `"error": "video missing"`).

Verify after:

```bash
ls videos/                              # should show all 8 sport folders
find videos -type f -name '*.avi' -o -name '*.mp4' | wc -l   # should be ~2080
du -sh videos/                          # ~20-30 GB total expected
```

---

## Step 7 — Prefetch the 27B-AWQ-4bit weights (≈ 5–10 min)

Downloads to `$HF_HOME` so it persists across Stop/Start.

```bash
huggingface-cli download cyankiwi/Qwen3.5-27B-AWQ-4bit
```

Without this step the first extractor call would hit a ~10-min stall during the first inference. Better to do it explicitly so progress is visible.

---

## Step 8 — Smoke test (≈ 3 min)

Validate the pipeline on the rented hardware before committing 40+ hours.

```bash
# Use a separate manifest+out_dir so the smoke doesn't pollute the production cache
python -u extract_logits_subset.py \
  --split train --limit 7 \
  --out_dir logits_cache_subset_TEST \
  --manifest logits_cache_subset_TEST.manifest.jsonl \
  2>&1 | tee smoke_27b_int4_run.log
```

Expected lines at the end:

```
Done. answered=7 avg_topk_mass=0.98xx avg_chosen_in_topk=1.0000
peak VRAM: ~14-18 GB
```

If `avg_topk_mass` < 0.95 or `chosen_in_topk` < 0.99, halt and inspect — something is mis-stacked. If the smoke passes, clean up:

```bash
rm -rf logits_cache_subset_TEST logits_cache_subset_TEST.manifest.jsonl
```

---

## Step 9 — Production run (≈ 40–60 hours)

Inside `tmux` so SSH disconnects don't kill the job:

```bash
tmux new -s extract
# inside tmux:
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache

python -u extract_logits_subset.py --split both 2>&1 | tee extract_full.log
# Detach: Ctrl-b then d
# Reattach later: tmux attach -t extract
```

Live progress shows on the tqdm bar:
```
videos: 12/2080 [04:32<13:02:11, 22.7s/vid] n=68 mass=0.987 chosen_in=1.000 rate=0.25/s eta_min=2480
```

Watch values:
- `mass` should stay > 0.95
- `chosen_in` should be 1.000 ± 0.001
- `rate` ~0.15–0.30 samples/sec on H100-class hardware
- `eta_min` is the live ETA

If a row errors, the script logs to manifest and moves on. To check for errors:

```bash
grep -c '"error"' logits_cache_subset.manifest.jsonl
```

Resumable: re-running the same command picks up at the next pending qa_id.

---

## Step 10 — Pull the cache back / push to HF (≈ 10 min)

When extraction finishes, you have ~400–500 MB of `.pt` files. Two storage options:

### Option A — push to HF Hub as a private dataset (recommended; survives pod death)

```bash
# on the pod
huggingface-cli upload "<YOUR_USERNAME>/qwen35-27b-logits-subset" \
  /workspace/Granduation\ Project/logits_cache_subset \
  --repo-type dataset --private --commit-message "Recipe A extraction"

# also upload the manifest
huggingface-cli upload "<YOUR_USERNAME>/qwen35-27b-logits-subset" \
  /workspace/Granduation\ Project/logits_cache_subset.manifest.jsonl \
  --repo-type dataset --path-in-repo manifest.jsonl
```

Then on WSL or any future student-training machine:

```bash
huggingface-cli download "<YOUR_USERNAME>/qwen35-27b-logits-subset" \
  --repo-type dataset \
  --local-dir "/home/accy/Granduation Project/logits_cache_subset"
```

### Option B — rsync directly back to WSL

```bash
# from WSL
rsync -avz --progress \
  -e "ssh ${SSH_OPTS}" \
  "${POD_DEST}/logits_cache_subset/" \
  "/home/accy/Granduation Project/logits_cache_subset/"

rsync -avz --progress \
  -e "ssh ${SSH_OPTS}" \
  "${POD_DEST}/logits_cache_subset.manifest.jsonl" \
  "/home/accy/Granduation Project/logits_cache_subset.manifest.jsonl"
```

~500 MB at typical home-internet download speed: 5–15 min.

I'd do **both** — HF for durability, rsync as immediate-access backup.

---

## Step 11 — Stop the pod (don't Destroy)

In the rental provider's web UI:
- **Stop** the pod → GPU billing pauses, persistent disk persists
- Cost while stopped: ~$0.04/hr (storage only) ≈ $1/day

Only **Destroy** when you're certain you don't need anything from the pod again. Destroy wipes the persistent volume, so re-pulling weights and re-uploading code would be needed for a future session.

---

## Failure-recovery cheat sheet

| Problem | Action |
|---|---|
| SSH disconnect mid-run | `tmux attach -t extract` to reconnect |
| Pod dies / needs restart | Click Start in the UI, then `tmux new -s extract`, re-run the same command — the manifest auto-resumes |
| `mass < 0.95` consistently | Halt with Ctrl-c; something's wrong with logit capture. Inspect a few `.pt` files |
| `chosen_in < 0.99` | Halt; bug in `out.logits` stacking. Check transformers version compatibility |
| Cap-hit rate > 50 % | Generations are running into max_new_tokens=256. Either accept (and filter at training time) or bump cap to 384 |
| Out of disk | Push completed logits to HF, free up the volume, resume |
| Out of VRAM | Shouldn't happen on 80 GB; if it does, set `device_map="auto"` with `max_memory={0: "70GiB", "cpu": "60GiB"}` to spill |

---

## Total cost estimate at this configuration

| GPU | Wall time | Cost |
|---|---|---|
| Vast H100 NVL @ $1.33/hr | ~50 h | ~$67 |
| Vast H100 PCIe @ $1.73/hr | ~50 h | ~$87 |
| Vast A100 80 GB @ $0.67/hr | ~80 h | ~$54 |

Plus minor overhead:
- Stopped-disk fee: ~$1 / day until you Destroy
- Bandwidth (download videos + upload logits): ~$0.50 total

Total budget for the full run: **~$60–90 depending on GPU**.

---

## Quick reference — the entire flow as a single block

If you want to glance at one block instead of scrolling:

```bash
# --- WSL: set vars + push code ---
export POD_PORT=<port>; export POD_IP=<ip>; export HF_VIDEOS_REPO="<user>/<repo>"
export POD_DEST="root@${POD_IP}:/workspace/Granduation Project"
export SSH_OPTS="-p ${POD_PORT} -i ~/.ssh/id_ed25519"
eval "$(ssh-agent)" && ssh-add ~/.ssh/id_ed25519
ssh ${SSH_OPTS} root@${POD_IP} 'apt update && apt install -y rsync tmux'
rsync -avz --progress --exclude='.venv' --exclude='videos/' --exclude='frames/' \
  --exclude='features/' --exclude='logits_cache_*' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='*.txt' \
  -e "ssh ${SSH_OPTS}" "/home/accy/Granduation Project/" "${POD_DEST}/"

# --- POD: bootstrap ---
cd "/workspace/Granduation Project"
export HF_HOME=/workspace/hf_cache
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
pip install huggingface_hub autoawq compressed-tensors
huggingface-cli login    # paste token

# --- POD: pull data + model ---
python -u download_subset_videos.py --repo-id "${HF_VIDEOS_REPO}" --workers 16
huggingface-cli download cyankiwi/Qwen3.5-27B-AWQ-4bit

# --- POD: smoke ---
python -u extract_logits_subset.py --split train --limit 7 \
  --out_dir logits_cache_subset_TEST \
  --manifest logits_cache_subset_TEST.manifest.jsonl
rm -rf logits_cache_subset_TEST logits_cache_subset_TEST.manifest.jsonl

# --- POD: production run in tmux ---
tmux new -s extract
source .venv/bin/activate; export HF_HOME=/workspace/hf_cache
python -u extract_logits_subset.py --split both 2>&1 | tee extract_full.log
# Ctrl-b d to detach; tmux attach -t extract to reconnect

# --- POD post-run: push to HF + back to WSL ---
huggingface-cli upload "<user>/qwen35-27b-logits-subset" logits_cache_subset \
  --repo-type dataset --private
# (or rsync from WSL — see Step 10 Option B)
```
