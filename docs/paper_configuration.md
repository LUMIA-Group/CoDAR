# Paper-to-configuration cross-check

Source: the author-provided `2816_CoDAR_Continuous_Diffusio.pdf`,
“CoDAR: Continuous Diffusion Language Models are More Powerful Than You Think”,
16 pages. Page references below use PDF page numbers. The PDF is not committed.

## Main OpenWebText experiment

| Setting | Paper evidence | Repository setting |
| --- | --- | --- |
| Embedding dimension | Section 5.1, p. 7: 64 unless specified | Decoder `encoder_latent_dim: 64`; DiT `latent_dim: 64` |
| Encoder | Section 5.1, p. 7: frozen Qwen3-Embedding, normalized representations | `Qwen/Qwen3-Embedding-0.6B`; first 64 hidden channels divided by 2.690504312515259 (checkpoint/code details, not specified numerically by paper) |
| Context length | Section 5.1, p. 6: 512, sequence packing | Decoder `n_ctx: 512`, `n_positions: 513` includes BOS; DiT `max_seq_len: 512` |
| Decoder | Section 5.1, p. 7: GPT-2 small plus cross-attention | 12 layers, width 768, 12 heads, cross-attention enabled |
| DiT architecture | Section 5.1, p. 7: MDLM architecture | 12 layers, width 768, 12 heads in checkpoint/model configuration |
| Decoder training inputs | Section 4.2, p. 6: reconstruction conditioned on clean x0 | `embedding_noise_scale: 0.0` in Python and launcher defaults |
| Diffusion | Section 4.1, p. 5: cosine VP, Gaussian noise, v-prediction | `NoiseScheduleVP(schedule="cosine")`, `parameterization='v'`; unweighted MSE |
| Decoder duration | Appendix A, p. 13: 1 epoch | `num_train_epochs: 1` |
| DiT duration | Section 5.1, p. 6: 250k steps | `max_steps: 250000` |
| Effective batch size | Appendix A, p. 13: 512 for both | Decoder 16/GPU × 4 accumulation × 8 GPUs; DiT 64/GPU × 1 × 8 |
| Decoder learning rate | Appendix A, p. 13 | 1e-3, cosine, warmup ratio 0.05 |
| DiT learning rate | Appendix A, p. 13 | 4e-4, `constant_with_warmup`, warmup_steps 10000 |
| Weight decay | Appendix A, p. 13 | Decoder 0.1; DiT 0.02 |
| Betas and clipping | Appendix A, p. 13 | Both (0.9, 0.95), max_grad_norm 1.0 |

The paper's clean-input decoder objective is corroborated by local `decoder.sh`
passing `--embedding_noise_scale 0.0`, and `utils/generate.py` selecting the
`dec_qwen_owt_64_fixbos_512_0.0/checkpoint-33258` checkpoint.
The checkpoint's trainer state confirms one completed epoch. This resolves the
previous launcher default of 0.1; no inference from the run name alone is needed.

The main DiT checkpoint selected by `utils/generate.py` is
`dit_qwen3_owt_128/checkpoint-250000`. Despite **128 in the directory name**, its
saved config has `latent_dim=64`, `max_seq_len=512`, and its trainer state has
`global_step=max_steps=250000`. The standalone `dit/config.json` agrees.
The new main run name `dit_qwen3_owt_64_512` describes these actual dimensions;
it does not rename or move existing checkpoint directories.

## Sampling and evaluation

- Table 1 (p. 7): 1,000 samples, 250 diffusion steps; decoder temperatures
  0, 0.25, 0.5, 0.75, 1. Table 3's matching T=1 scores identify ancestral
  sampling as the Table 1 CoDAR sampler.
- Tables 3–4 (p. 8): ancestral / DPM-Solver at 25, 50, 100, 250 steps;
  decoder temperature fixed to 1.
- Section 5.1 (p. 6): GPT-2 Large generative perplexity; diversity is the product
  of distinct n-gram ratios for n=2,3,4. Token entropy is not this diversity metric.
- Appendix D, Table 9 (p. 14): latent dimensions 64/256/768, temperature 1;
  256 is an ablation, not the main default.
- Section 5.4 (p. 9): throughput measurements use batch size 100 and 250 steps.
  This is an inference batch size, not the training batch size of 512.

## LM1B main result

Section 5.1 (p. 6) specifies sentence packing, length 128, and 250k training
steps. Section 5.2 (p. 7) explicitly reuses the **OpenWebText-trained decoder**
and sets decoder temperature to 1. The local `config_dit_lm1b` YAML/JSON match
length 128 and latent dimension 64. `config_decoder_lm1b.yaml` trains a different
decoder and should not be presented as necessary for reproducing Table 2.

## Remaining evidence gaps / discrepancies

1. **LM1B checkpoint:** the generation script may use a 230k checkpoint. This is
   an intentional checkpoint choice and can be replaced with the released model
   address when publishing. Use a placeholder such as
   `/path/to/codar/checkpoints/dit_lm1b/checkpoint-230000` in public examples.
2. **Optimizer:** the paper says Adam, but the implementation uses **AdamW**
   (`adamw_torch`). AdamW is the intended optimizer for the release.
3. **Warmup semantics:** `constant_with_warmup` now implements the paper
   description: the learning rate rises from zero to 4e-4 over the first 10,000
   optimizer updates, then stays constant.
4. **Environment:** these saved model configs record Transformers 4.54.0.
   The release preparation pinned 4.53.3 and verified small-model checks; this
   is not a recovered full historical environment.
5. **Data/normalization:** paper text does not supply the full OpenWebText split,
   separators, normalization-estimation recipe, or numeric scale. The data
   conversion tool is format-compatible, not yet exact historical preprocessing.
6. **Other implementation choices:** seed 42, fp16, dropout, token IDs, and
   timestep epsilon remain code settings; the paper does not independently
   establish every value. The denoiser consumes alpha(t) as its time feature.

Main config correction does not by itself validate the reported generation
metrics. No weights were modified and no training was launched for this audit.
