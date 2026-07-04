# RESUME — kawiki-g4e4b contextual embed (paused 2026-07-05)

Stage-1c contextual embed (`src/embed_contextual.py`) **paused mid-run**, checkpointed. Pick up here.

## State

- Config: `kawiki-g4e4b` — corpus `kawiki`, model `google/gemma-4-E4B` (text tower, `d_model=2560`, `n_layers=42`, `layer_idx=27`).
- Progress: **12000 / 18401 lemmas** embedded. ~6401 remain.
- Container `anbani-we-g4e4b` **stopped** (Exit 137 — no SIGTERM handler, killed after clean checkpoint; not a crash). GPU idle.
- Corpus + vocab already built on node (`work/kawiki/sentences.jsonl`, vocab). Only `embed` remains.

## Checkpoint (on GPU node — bind mount, survives container)

- Host: `george@100.88.125.127` (`exce`), dir `~/word-embeddings-work` (== `/work` in container).
- `work/kawiki-g4e4b/vectors.f32.npy` (188 MB, 18401×2560 fp32, preallocated) + `vectors.progress` (=`12000`).
- `load_checkpoint()` auto-resumes from row 12000 on next `embed` run; no re-embed of done rows. `vectors.progress` is deleted on clean completion.

## Resume (from local repo)

Embed only — checkpoint resumes, corpus/vocab untouched:

```
ssh george@100.88.125.127 "cd ~/word-embeddings-work && \
  docker run -d --name anbani-we-g4e4b --gpus all \
    -v \$PWD:/work -w /work \
    -v ~/.local/share/kiwix-desktop:/zim:ro \
    -v ~/.hfcache:/root/.cache/huggingface \
    -e HF_TOKEN -e WE_ZIM_DIR=/zim \
    anbani-we-generate make embed DS=kawiki-g4e4b CORPUS=kawiki"
# follow: ssh george@100.88.125.127 "docker logs -f anbani-we-g4e4b"
```

`HF_TOKEN` must be exported in the node env (gated Gemma weights). ~6401 lemmas left; checkpoints every 2000.

After embed completes → `make gpu-stage2 DS=kawiki-g4e4b` (reduce+dist), then `make gpu-fetch`.
