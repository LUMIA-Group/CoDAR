# Release preparation

## Paper

- [CoDAR: Continuous Diffusion Language Models are More Powerful Than You Think](https://arxiv.org/abs/2603.02547)
- Venue tag: **EMNLP 2026**

## Scope

The documented main experiments are the decoder and DiT entry points and the
two main YAML configurations. Existing experimental scripts and results are
retained locally; this cleanup does not remove research work or rewrite history.
Local datasets, weights, logs, and samples should not be committed.

## Before publication

- Main latent dimension, decoder noise, AdamW choice, and warmup schedule are
  aligned with the paper; see `paper_configuration.md`. Replace checkpoint
  placeholders with released download addresses.
- Record dataset revision, train/validation split, tokenization and separators;
  recover the original software versions and GPU/memory/time requirements.
- Publish matching checkpoints and a portable generation/evaluation interface.
- Supply paper title, authors, citation, and the chosen project license.
- Revoke embedded SwanLab credentials. Removing login literals from the current
  tree does not remove them from previous Git commits. Sanitize history or
  publish a reviewed clean export before making the repository public.
- Verify a fresh clone can prepare data and run short training jobs before
  claiming full reproducibility.

## Third-party attribution to resolve

Existing copyright notices are preserved. `model/modeling_gpt2.py` and GPT-2
configuration code contain Hugging Face/OpenAI/NVIDIA Apache-2.0 notices.
`model/modeling_dit.py` names the MDLM implementation as its source.
`data/data.py` contains a Meta copyright and MIT-license notice that refers to
an upstream LICENSE file. Locate the upstream license text and provenance before
release. Review the provenance of `utils/dpm_solver.py` as well.
No new blanket license is assigned to third-party code by this cleanup.
