# CoDAR

Code for **CoDAR: Continuous Diffusion Language Models are More Powerful Than You Think**.

Research code for training an autoregressive decoder and a continuous diffusion
transformer over frozen text-model representations.

**Paper:** [arXiv:2603.02547](https://arxiv.org/abs/2603.02547) · **Venue:** EMNLP 2026

## Main experiment

| Component | Entry point | Training configuration | Model configuration |
| --- | --- | --- | --- |
| Decoder | `train/train_decoder.py` | `config/train_config/config_decoder.yaml` | `config/model_config/config_decoder_qwen_owt_512.json` |
| DiT | `train/train_dit.py` | `config/train_config/config_dit.yaml` | `config/model_config/config_dit.json` |

Both trainers use `train/dit_args.py`, `data/data.py`, and
`utils/noise_scheduler.py`. Model implementations live in `model/`.
Other experimental variants are not the documented main training workflow.

## Installation

Use Python 3.10 or newer and a CUDA-compatible PyTorch installation for training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The public configurations load `Qwen/Qwen3-Embedding-0.6B` through Transformers.
To use a local copy, pass `--embed_model_path /path/to/model` to either launcher.
Experiment tracking is disabled by default. SwanLab is optional: install it,
authenticate outside the source code, and set `report_to: swanlab` to enable it.

## Data preparation

The main configurations consume `train_qwen.bin` and `val_qwen.bin`: flat native
`uint32` token streams. Each split must contain at least 512 tokens. Training
uses random offsets within blocks; validation uses contiguous non-overlapping
blocks. Incomplete trailing blocks are dropped.

Given your own, already separated training and validation text files (one
nonempty document per line), or JSONL files with a `text` field:

```bash
python scripts/prepare_data.py \
  --train_file /path/to/train.jsonl \
  --val_file /path/to/val.jsonl \
  --output_dir data/openwebtext_preprocess
```

This tool concatenates tokens without special tokens by default; `--append_eos`
adds EOS between documents. It records preprocessing options and refuses to
replace existing binaries. **This is a format-compatible preparation tool, not
an assertion of the paper's original preprocessing recipe.** The original
OpenWebText source revision, split, and document separator policy still need
to be recorded for exact reproduction. Raw datasets are not distributed here.

## Three-stage workflow

The complete OpenWebText workflow is: prepare token data, train both models,
generate samples from the two checkpoints, then evaluate the generated samples.
Replace every `/path/to/...` value with a real local path.

### 1. Prepare data

Provide already split OpenWebText text/JSONL files and create the binary streams:

```bash
python scripts/prepare_data.py \
  --train_file /path/to/openwebtext/train.jsonl \
  --val_file /path/to/openwebtext/val.jsonl \
  --output_dir /path/to/openwebtext_preprocess \
  --tokenizer /path/to/Qwen3-Embedding-0.6B
```

The output directory must contain `train_qwen.bin` and `val_qwen.bin`.

### 2. Train decoder and DiT

Train the decoder first and then the diffusion model. Both commands use eight
GPUs by default; set `NUM_GPUS` for a different number. The embedding model is
frozen and is used by both stages.

```bash
NUM_GPUS=8 bash scripts/train_decoder.sh \
  --data_path /path/to/openwebtext_preprocess \
  --embed_model_path /path/to/Qwen3-Embedding-0.6B \
  --output_dir outputs/decoder \
  --run_name codar_decoder

NUM_GPUS=8 bash scripts/train_dit.sh \
  --data_path /path/to/openwebtext_preprocess \
  --embed_model_path /path/to/Qwen3-Embedding-0.6B \
  --output_dir outputs/dit \
  --run_name codar_dit
```

The resulting checkpoints are normally under
`outputs/decoder/codar_decoder/checkpoint-*` and
`outputs/dit/codar_dit/checkpoint-*`. Select the desired completed checkpoint
from each directory.

To resume, add `--resume_from_checkpoint /path/to/checkpoint-N` to the relevant
command. LM1B uses the same two entry points with
`config/train_config/config_decoder_lm1b.yaml` and
`config/train_config/config_dit_lm1b.yaml`; pass `/path/to/lm1b` as `--data_path`.

### 3. Generate and evaluate

Generate 1,000 samples with the trained decoder and DiT. The decoder and DiT
must use matching latent dimension and sequence length (the main experiment is
64 dimensions and length 512).

```bash
python utils/generate.py \
  --train_config config/train_config/config_decoder.yaml \
  --decoder_checkpoint /path/to/decoder/checkpoint-33258 \
  --diffusion_checkpoint /path/to/dit/checkpoint-250000 \
  --embed_model_path /path/to/Qwen3-Embedding-0.6B \
  --solver ancestral \
  --sampling_steps 250 \
  --temperature 1.0 \
  --save_pickle_root outputs/samples
```

The command writes a `.pkl` sample file under `outputs/samples`. Evaluate it
with the portable evaluation launcher. Diversity works without a language
model; add GPT-2 Large paths to compute generative perplexity.

```bash
bash scripts/eval.sh \
  --filename outputs/samples/your_samples.pkl \
  --ppl_model /path/to/gpt2-large \
  --tokenizer /path/to/gpt2-large \
  --device cuda \
  --output_json outputs/eval/metrics.json
```

To run evaluation without perplexity, omit `--ppl_model` and `--tokenizer`.
The evaluation output contains `num_samples`, diversity, and (when requested)
`generative_ppl`.

### Paper-aligned settings

The main decoder and DiT both use **64-dimensional** embeddings and sequence
length **512**. The decoder trains on clean embeddings (noise scale **0.0**)
for one OpenWebText epoch; DiT trains for **250,000** steps with a 10,000-step warmup followed by a
constant learning rate. With eight GPUs,
effective batch sizes are `16 × 4 × 8 = 512` and `64 × 1 × 8 = 512`.
The 256/768-dimensional settings belong to the hidden-dimension ablation.
The implementation uses AdamW.
LM1B uses length 128 and reuses the OpenWebText decoder.

See [paper/configuration cross-check](docs/paper_configuration.md) for the
complete parameter mapping, sampling settings, and unresolved discrepancies.
Embedding normalization constants and seed 42 are preserved from the code.

## Validation

```bash
python train/train_decoder.py --help
python train/train_dit.py --help
python utils/generate.py --help
bash scripts/eval.sh --help
bash -n scripts/train_decoder.sh scripts/train_dit.sh scripts/eval.sh
```

These are lightweight checks, not a full training run or a verification of paper
metrics. Full reproduction requires the data, embedding model, GPT-2 Large
perplexity model, and released checkpoints.

## Publication status

This repository contains the research code, citation metadata, and MIT project
license. Third-party components retain their original notices and terms; see
[release notes](docs/release.md) for provenance and attribution details. Data,
embedding models, evaluation models, and trained checkpoints are not distributed
here.
