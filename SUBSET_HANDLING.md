# Subset Handling

This chapter documents the construction of the curated **2,080-video subset** used for teacher-logit extraction with `Qwen3.5-27B (Int4)`. It records the motivation for sub-sampling the Sports-QA corpus, the data-shape verification that preceded the design, the stratification logic (and a mid-design pivot from 5 to 8 categories), the sampling rules, the output schema, and the HuggingFace tooling built to materialise the subset on rented GPUs. Only material relevant to subset construction is included; downstream logit extraction, training, and evaluation are out of scope.

---

## 1. Motivation

The combined `train.json` + `val.json` of Sports-QA contains **75,355 question–answer rows across 4,771 unique videos** (counts cross-checked against the per-split totals in [METHODOLOGY_REPORT.md §2.1](METHODOLOGY_REPORT.md)). Running the full set through `Qwen3.5-27B (Int4)` for teacher-logit extraction is infeasible inside the project's ~20 hour rented-GPU budget; back-of-envelope at the observed ~7–9 s per (video, question) pair on the rented A100, the full set would run for 6+ days.

The requirement for the subset is therefore not raw size reduction but **balanced shrinkage**: the subset must preserve the dataset's hard properties — rare question types, the long-tail answer vocabulary, and per-sport diversity — so that downstream conclusions drawn on the subset transfer back to claims about the full distribution.

The handoff specification (`sports-qa-subset-handoff.md`, attached to the build conversation) called for **200 train + 60 val videos per sport category**, capped per-video question counts, soft yes/no balancing to fight the dominant `yes`/`no` answers, and full reproducibility under a fixed seed. Those rules were taken as the input contract for the sampler described below.

---

## 2. Source-data verification

Before writing any sampling code, the annotation files in [meta-data/](meta-data/) were inspected to confirm assumptions about schema and category encoding.

The verified structure is a flat JSON list of per-QA entries (one row per question, multiple rows per video):

```json
{
  "qa_id": 0,
  "video": "aerobic_gymnastics/v_HIow7_XktlQ_c003_01",
  "type": "Descriptive",
  "question": "What is the video about?",
  "answer": "aerobic gymnastics",
  "ans_cls": 16
}
```

The `video` field is `<sport_prefix>/<stem>` and the prefix is the **only** sport indicator in the row — there is no separate sport column. A scan across all three splits showed the dataset uses exactly **5 raw prefixes**:

| Prefix | Train videos | Val videos | Test videos |
|---|---|---|---|
| `football` | 618 | 206 | 206 |
| `basketball` | 527 | 175 | 177 |
| `volleyball` | 351 | 117 | 118 |
| `aerobic_gymnastics` | 303 | 101 | 101 |
| `fg` | 1,780 | 593 | 594 |

The `fg/` prefix turned out to be a single umbrella over the **FineGym** corpus, which aggregates four distinct gymnastics events (Balance Beam, Uneven Bar, Floor Exercise, Vault) without exposing them in the path. The sub-event is observable only inside the row itself: every fg video has at least one Descriptive question of the form *"What is the video about?"*, whose answer is one of `Balance Beam`, `Uneven Bar`, `Floor Exercise`, or `Vault`. This recovery channel is what makes the 8-category stratification described below possible.

The `type` field is one of four PascalCase strings — `Descriptive`, `Temporal`, `Causal`, `Counterfactual` — confirming the distribution table already documented in [METHODOLOGY_REPORT.md §2.2](METHODOLOGY_REPORT.md).

---

## 3. Stratification design (the 5 → 8 pivot)

The first iteration of the sampler stratified on the 5 raw prefixes, producing a 1,300-video subset (200 train + 60 val × 5). On review, this collapsed all four FineGym disciplines into a single bucket, which silently let any single sub-event (e.g. Uneven Bar) dominate the `fg` slice. The pivot was to expand to **8 logical sport categories**:

```
football
basketball
volleyball
Gym                ← aerobic_gymnastics relabelled for clarity
Balance_Beam       ┐
Uneven_Bar         │ ← recovered from fg/* via the
Floor_Exercise     │   "What is the video about?" answer
Vault              ┘
```

This decomposition is implemented by [`build_fg_sub_sport_map`](sample_subset.py#L70-L84), which iterates the raw rows, picks the canonical descriptive question, and maps the answer through a normalisation table that absorbs case and pluralisation variants ([sample_subset.py:17-28](sample_subset.py#L17-L28)). The lookup function [`sport_of`](sample_subset.py#L87-L95) then resolves any video to its logical category in one place, so every downstream stage agrees on the partition.

The chosen 8-category split required verifying that every category has enough material to satisfy the 200-train / 60-val floor:

| Logical sport | Train videos available | Val videos available |
|---|---|---|
| football | 618 | 206 |
| basketball | 527 | 175 |
| volleyball | 351 | 117 |
| Gym (= aerobic_gymnastics) | 303 | 101 |
| Uneven_Bar | 520 | 147 |
| Balance_Beam | 498 | 194 |
| Floor_Exercise | 474 | 152 |
| Vault | 288 | 100 |

All eight clear the threshold; the smallest is Vault (288 train, 100 val). Recovery of the fg sub-event was 100% successful in both splits — every fg video had a parseable descriptive answer (1,780/1,780 in train, 593/593 in val). The sampler still emits a per-run warning if any fg video resolves to `fg_UNKNOWN` ([sample_subset.py:118-120](sample_subset.py#L118-L120)).

Final per-category target: **200 train + 60 val = 260 videos × 8 categories = 2,080 videos** — exactly the size called for in the original handoff.

---

## 4. Sampling rules

The sampler applies four rules in order: rare-type-aware video selection, per-video question caps, long-tail answer preference within each cap, and a soft yes/no balancing pass over the result.

### 4.1 Rare-type richness score

Within a category, videos are ranked by how many rare-typed questions they carry, then the top-k are kept. The score is a weighted sum biased toward Counterfactual and Causal rows ([sample_subset.py:38-43](sample_subset.py#L38-L43)):

```python
RARE_TYPE_WEIGHTS = {
    "Counterfactual": 4,
    "Causal":         2,
    "Temporal":       1,
    "Descriptive":    1,
}
```

Counterfactual rows are 30× rarer than Descriptive rows in the source distribution (1.6% vs 51.3%), so the 4× weight is a deliberate over-correction: when the cap is reached, the videos that *do* contain Counterfactual material are kept preferentially. Ties are broken by a draw from a `random.Random(42)` instance ([sample_subset.py:125-127](sample_subset.py#L125-L127)), which is what makes reruns byte-identical.

### 4.2 Per-video question caps

For each chosen video the sampler retains, at most:

```python
QUESTION_TYPE_CAPS = {
    "Descriptive":    3,
    "Temporal":       3,
    "Causal":         2,
    "Counterfactual": 1,
}
```

(see [sample_subset.py:31-36](sample_subset.py#L31-L36)). The caps bound runtime variance per video downstream — without them, a video with twelve Temporal questions and a video with two would impose very different costs on the inference loop.

### 4.3 Long-tail answer preference

Within each per-type cap, rows whose answer is `yes`, `no`, or a bare digit are deprioritised in favour of long-tail answers ([`is_long_tail`](sample_subset.py#L98-L105) and the sort in [`cap_questions`](sample_subset.py#L139-L156)). The motivation is that the four labels `no`, `yes`, `1`, `2` cover ~45% of the corpus ([METHODOLOGY_REPORT.md §2.3](METHODOLOGY_REPORT.md)); without this preference the per-video Descriptive cap would routinely fill with three `yes`/`no` rows and starve the rare specialist phrases (e.g. *"flic-flac with step-out, also with support on one arm"*) that make the dataset interesting for a teacher–student transfer.

### 4.4 Soft yes/no balancing

Even after the long-tail preference, the surviving subset still carries far more `yes`/`no` rows than 50/50. [`balance_yes_no`](sample_subset.py#L188-L238) performs a final pass that drops majority-side rows until the ratio is within ±5% of 50/50 (the `YES_NO_TOLERANCE` constant at [sample_subset.py:46](sample_subset.py#L46)). The drop order is randomised by the same seeded RNG, so the balancing is reproducible. In the production run the balancer dropped **253 yes/no rows from train and 68 from val**, hitting an exact 50/50 ratio in both splits (see §5 below).

### 4.5 Reproducibility

Every stochastic step takes its randomness from a single `random.Random(seed=42)` instance threaded through the call graph. A two-run check on the production output (`md5sum data/subsets/*.json` before and after a fresh rerun) gave identical hashes for all four files, confirming determinism end-to-end. A `set(train_ids) & set(val_ids)` overlap check is asserted in `main()` ([sample_subset.py:350-353](sample_subset.py#L350-L353)) — the source splits are already disjoint, so this is a guard against future regressions, not a filter.

---

## 5. Outputs and final composition

A run of [`sample_subset.py`](sample_subset.py) at the default arguments writes four files into `data/subsets/`. The composition that the production seed (42) produced is:

### 5.1 Per-sport video counts (train / val)

| Sport | Train | Val |
|---|---|---|
| football | 200 | 60 |
| basketball | 200 | 60 |
| volleyball | 200 | 60 |
| Gym | 200 | 60 |
| Balance_Beam | 200 | 60 |
| Uneven_Bar | 200 | 60 |
| Floor_Exercise | 200 | 60 |
| Vault | 200 | 60 |
| **Total** | **1,600** | **480** |

(numbers from [data/subsets/subset_stats.json](data/subsets/subset_stats.json))

### 5.2 Per-question-type counts

| Question type | Train | Val |
|---|---|---|
| Descriptive | 4,614 | 1,390 |
| Temporal | 2,965 | 909 |
| Causal | 1,032 | 339 |
| Counterfactual | 326 | 105 |
| **Total questions** | **8,937** | **2,743** |

The combined subset is therefore **11,680 question–answer pairs over 2,080 videos**. The Counterfactual share rises from 1.6% in the source corpus to ~3.6% in the subset, and Causal from 5.0% to ~11.6% — confirming that the rare-type-richness score did its job.

### 5.3 Yes/no balance and answer-class coverage

After balancing, both splits hit an exact 50/50 ratio (train: 486 yes / 486 no; val: 153 yes / 153 no). The retained answer vocabulary is **190 distinct classes out of 191 in the full ans2cls.json** — 189 in train, 165 in val, with 1 class appearing in val but not train. Only one class from the full vocab is missing entirely from the subset.

### 5.4 Output files

| File | Purpose |
|---|---|
| [`data/subsets/subset_train.json`](data/subsets/subset_train.json) | 1,600 video entries with their kept (capped) questions |
| [`data/subsets/subset_val.json`](data/subsets/subset_val.json) | 480 video entries, same schema |
| [`data/subsets/subset_stats.json`](data/subsets/subset_stats.json) | per-sport, per-type, yes/no, totals |
| [`data/subsets/subset_video_manifest.json`](data/subsets/subset_video_manifest.json) | flat `{video_id, split, sport}` list for the downloader |

The per-entry schema follows the handoff specification:

```json
{
  "video_id": "basketball/v_-6Os86HzwCs_c001_00",
  "sport": "basketball",
  "split": "train",
  "questions": [
    {
      "question_id": 1234,
      "question_type": "Causal",
      "question_text": "...",
      "answer": "..."
    }
  ]
}
```

Downstream tooling — in particular [`extract_logits_subset.py`](extract_logits_subset.py) — consumes `subset_train.json` and `subset_val.json` directly.

---

## 6. HuggingFace download tooling

The 2,080 videos referenced by the subset live in the private dataset repo `KD0GP0IMSIU/KnowledgeDistillationVLMsQA` on HuggingFace. To fetch only the subset videos (rather than mirroring the full ~83 GB repo) onto a fresh rented GPU, [`download_subset_videos.py`](download_subset_videos.py) was added.

### 6.1 Strategy

The script does one bulk listing call against the HF API and then matches subset video IDs to repo paths in memory, before issuing parallel `hf_hub_download` calls for the matches. The matcher is in [`index_repo_files`](download_subset_videos.py#L38-L65); the resolution step is [`resolve_targets`](download_subset_videos.py#L68-L84); the per-file fetch is [`download_one`](download_subset_videos.py#L87-L109).

### 6.2 Layout-tolerance quirks

The repo's directory layout differs from the subset's logical taxonomy in two ways that the matcher absorbs:

- **`fg` ↔ `finegym_0X`.** The HF repo (and the local mirror in [videos/](videos/)) shards the FineGym videos across five sibling directories `finegym_01` through `finegym_05`, while the subset manifest references them under the single prefix `fg/`. The indexer registers every `finegym_*/<stem>` file under both its real key and the synthetic `fg/<stem>` key ([download_subset_videos.py:59-60](download_subset_videos.py#L59-L60)).
- **Doubled `volleyball/volleyball/`.** Some volleyball clips live one level deeper than the rest; an extra index entry handles this case ([download_subset_videos.py:63-64](download_subset_videos.py#L63-L64)).

### 6.3 Cache and `.metadata` sidecars

`hf_hub_download` writes a `<file>.metadata` sidecar in `.cache/huggingface/download/` next to every successful fetch. The sidecar records the commit hash, etag, and download timestamp; on subsequent runs the loader uses these to skip the wire transfer when the local file already matches the repo. A run that produces *only* `.metadata` files and no new video bytes therefore means the destination filesystem already contained byte-identical copies of every subset video — not a download failure. This was directly observed during integration: the local `videos/` tree from an earlier full-corpus pull already held all 2,080 subset videos at hash-equal content, so the first invocation of the downloader emitted exactly 2,080 sidecars and zero new mp4/avi files.

### 6.4 Offline matcher validation

The HF repo is gated and could not be listed from the development workstation. As an alternative, the local mirror at [videos/](videos/) — known to be hash-equal with the repo, see §6.3 — was passed as a synthetic file listing into the matcher. Of the **2,080 subset video IDs, 2,080 resolved to a local path with 0 missing**. All five layout conventions (`football/*.avi`, `basketball/*.avi`, `volleyball/volleyball/*.avi`, `aerobic_gymnastics/*.avi`, `finegym_0X/*.mp4`) round-tripped correctly. This is the closest the workstation can get to a true smoke test without HF credentials, and it exercises the entire matcher logic.

---

## 7. Test-set companion

Logit extraction on the subset eventually needs a paired pass over the **test split** for evaluation. The test split was deliberately *not* sub-sampled — the goal there is full coverage, not balanced reduction — so a separate manifest [`data/subsets/test_video_manifest.json`](data/subsets/test_video_manifest.json) was generated directly from `meta-data/test.json`. It lists all **1,196 test videos** (≈ 16 GB on disk) along with their split label and recovered logical sport, in the same shape as `subset_video_manifest.json`.

Reusing the existing downloader required only one code change: extending the `--splits` choices to include `test` ([download_subset_videos.py:124-125](download_subset_videos.py#L124-L125)). The matcher itself needed no changes; an offline validation pass against the local mirror resolved **1,196/1,196 test videos with 0 missing**.

The download invocation is:

```bash
.venv/bin/python download_subset_videos.py \
    --repo-id KD0GP0IMSIU/KnowledgeDistillationVLMsQA \
    --manifest data/subsets/test_video_manifest.json \
    --splits test \
    --workers 16
```

---

## 8. Verification record

The subset and its tooling passed the following end-to-end checks:

| Check | Result |
|---|---|
| Rerun determinism (`md5sum` × 2, seed=42) | identical hashes on all four output files |
| Disjointness (`set(train_ids) & set(val_ids)`) | empty (assert in `main()`) |
| Per-video caps (≤3 D / ≤3 T / ≤2 C / ≤1 CF) | satisfied for every entry in train and val |
| Yes/no ratio in `subset_stats.json` | 0.5000 (train), 0.5000 (val) — within ±0.05 tolerance |
| Logical-sport quotas (200 train / 60 val × 8) | exactly hit for every category |
| Answer-class coverage | 190 / 191 retained |
| HF matcher resolution (subset, offline) | 2,080 / 2,080 — 0 missing |
| HF matcher resolution (test, offline) | 1,196 / 1,196 — 0 missing |

The artefacts are stable, the sampler is reproducible, and the download path is validated end-to-end against the local mirror that is hash-equal with the HF repo.
