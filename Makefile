# Anbani/word-embeddings — deterministic orchestration.
#
# Three stages (mirrors Anbani/homoglyph):
#   stage 1  corpus -> vocab -> embed   HEAVY. GPU forward passes. Node Docker only.
#   stage 2  reduce -> dist             CPU. Runs on the node right after embed
#            (raw f32 vectors are large and live in work/ there), or locally from
#            a rsync'd work/ backup. Emits the committed dist/.
#   stage 3  evaluate + tests           CPU. The gate; also runs in CI against the
#            committed dist (probe metrics come from dist/neighbors.bin — no raw
#            vectors needed, so CI can validate without the node artifacts).
#
# Parameterize per dataset/corpus:  make embed DS=kawiki-eg   make vocab CORPUS=kawiki

PYTHON ?= python3
export PYTHONHASHSEED = 0
export PYTHONUTF8 = 1

CORPUS ?= kawiki
DS ?= kawiki-eg

.PHONY: all corpus vocab embed reduce dist evaluate test stage1 stage2 clean \
        docker-build-generate docker-build-build \
        gpu-sync gpu-stage1 gpu-stage2 gpu-logs gpu-stop gpu-fetch

all: evaluate

# ---- stage 1 (heavy, GPU node) ---------------------------------------------

## corpus: ZIM / poem -> work/<corpus>/sentences.jsonl
corpus:
	$(PYTHON) src/extract_corpus.py --corpus $(CORPUS)

## vocab: sentences -> work/<corpus>/{vocab.jsonl, form2lemma.json, occ.jsonl}
vocab:
	$(PYTHON) src/build_vocab.py --corpus $(CORPUS)

## embed: model forward passes -> work/<DS>/vectors.f32.npy  (needs GPU)
embed:
	$(PYTHON) src/embed_contextual.py --config $(DS)

# ---- stage 2 (CPU) ----------------------------------------------------------

## reduce: full-dim vectors -> UMAP-2D + alignment -> work/<DS>/layout.npz
reduce:
	$(PYTHON) src/reduce_layout.py --config $(DS)

## dist: quantize + neighbors + pack -> dist/<DS>/*
dist:
	$(PYTHON) src/build_dist.py --config $(DS)

stage1: corpus vocab embed
stage2: reduce dist

# ---- stage 3 (CPU, the gate) ------------------------------------------------

## evaluate: probe eval + schema/vocab_hash/displacement/blocklist asserts on dist/
evaluate:
	$(PYTHON) src/evaluate.py

## test: stdlib unit suite (pure-python logic; artifact tests skip if dist absent)
test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf src/__pycache__ tests/__pycache__ src/vendor/__pycache__

# ---- docker images ----------------------------------------------------------

docker-build-generate:
	docker build -f docker/Dockerfile.generate -t anbani-we-generate .

docker-build-build:
	docker build -f docker/Dockerfile.build -t anbani-we-build .

# ---- GPU node ("exce", Tailscale RTX 3090) ----------------------------------
# Docker-only there; never install on the host. Every target checks reachability
# first and fails fast. The kawiki ZIM is mounted read-only from the node's
# kiwix-desktop dir — never re-downloaded.

GPU_HOST ?= george@100.88.125.127
GPU_DIR ?= ~/word-embeddings-work
GPU_HFCACHE ?= ~/.hfcache
GPU_ZIM_DIR ?= ~/.local/share/kiwix-desktop
SSH_OPTS = -o ConnectTimeout=10 -o BatchMode=yes
GPU_ENV ?=
IMG_GEN = anbani-we-generate
IMG_BUILD = anbani-we-build
CONTAINER = anbani-we-stage1

## gpu-sync: rsync this repo to the node (excludes heavy/gitignored dirs)
gpu-sync:
	@ssh $(SSH_OPTS) $(GPU_HOST) true || (echo "gpu-sync: $(GPU_HOST) unreachable" >&2; exit 1)
	ssh $(SSH_OPTS) $(GPU_HOST) 'mkdir -p $(GPU_DIR) $(GPU_HFCACHE)'
	rsync -az --delete \
		--exclude .git --exclude work --exclude dist --exclude __pycache__ \
		--exclude '*.npy' \
		-e "ssh $(SSH_OPTS)" ./ $(GPU_HOST):$(GPU_DIR)/

## gpu-stage1: build the CUDA image and run corpus+vocab+embed DETACHED on the node.
## HF_TOKEN must be exported in the node's environment (gated Gemma weights).
## The ZIM dir is mounted read-only at /zim. Follow with `make gpu-logs`.
gpu-stage1: gpu-sync
	ssh $(SSH_OPTS) $(GPU_HOST) "cd $(GPU_DIR) && \
		docker build -f docker/Dockerfile.generate -t $(IMG_GEN) . && \
		docker rm -f $(CONTAINER) 2>/dev/null || true; \
		docker run -d --name $(CONTAINER) --gpus all \
			-v \$$PWD:/work -w /work \
			-v $(GPU_ZIM_DIR):/zim:ro \
			-v $(GPU_HFCACHE):/root/.cache/huggingface \
			-e HF_TOKEN -e WE_ZIM_DIR=/zim $(GPU_ENV) \
			$(IMG_GEN) make corpus vocab embed DS=$(DS) CORPUS=$(CORPUS)"
	@echo "gpu-stage1: detached ($(DS)). logs -> make gpu-logs ; artifacts -> make gpu-fetch"

## gpu-stage2: reduce+dist on the node (CPU build image), from the work/ vectors there
gpu-stage2:
	@ssh $(SSH_OPTS) $(GPU_HOST) true || (echo "gpu-stage2: $(GPU_HOST) unreachable" >&2; exit 1)
	ssh $(SSH_OPTS) $(GPU_HOST) "cd $(GPU_DIR) && \
		docker build -f docker/Dockerfile.build -t $(IMG_BUILD) . && \
		docker run --rm -v \$$PWD:/work -w /work $(IMG_BUILD) \
			make reduce dist DS=$(DS)"

## gpu-logs: follow the detached stage-1 container
gpu-logs:
	ssh $(SSH_OPTS) $(GPU_HOST) "docker logs -f $(CONTAINER)"

## gpu-stop: stop + remove the detached stage-1 container
gpu-stop:
	-ssh $(SSH_OPTS) $(GPU_HOST) "docker rm -f $(CONTAINER)"

## gpu-fetch: rsync dist/ back (committed from the Mac) + a vectors backup
gpu-fetch:
	@ssh $(SSH_OPTS) $(GPU_HOST) true || (echo "gpu-fetch: $(GPU_HOST) unreachable" >&2; exit 1)
	rsync -az -e "ssh $(SSH_OPTS)" $(GPU_HOST):$(GPU_DIR)/dist/ dist/
	-rsync -az -e "ssh $(SSH_OPTS)" $(GPU_HOST):$(GPU_DIR)/work/$(DS)/vectors.f32.npy work/$(DS)/vectors.f32.npy
