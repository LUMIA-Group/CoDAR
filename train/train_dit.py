from dataclasses import dataclass
import datasets
import pathlib
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

import sys
import argparse

sys.path.append(str(pathlib.Path(__file__).parents[1]))
from train.dit_args import get_args, add_training_overrides
from model.configuration_dit import DITConfig
from model.modeling_dit import DIT
from data.data import get_dataset
from utils.noise_scheduler import NoiseScheduleVP, expand_dims

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class DLMTrainer(Trainer):
    """
    Continuous Diffusion Language Model Trainer.
    """

    def __init__(
        self,
        *args,
        embed_scheme = None,
        embed_model_path = None,
        encoder_latent_dim = None,
        loss_weight_type: str = "scheduler",  # "ones"
        parameterization: str = "data",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = NoiseScheduleVP(schedule="cosine")
        self.time_epsilon = 1e-5
        self.loss_weight_type = loss_weight_type
        self.embed_scheme = embed_scheme
        self.parameterization = parameterization
        self.encoder_latent_dim = encoder_latent_dim
        self._init_embed_model(embed_scheme, embed_model_path)

    def _init_embed_model(self, embed_scheme, embed_model_path):
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        if embed_scheme == "gpt2-last":
            self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_path, use_fast=True, legacy=False)
            self.embed_model = AutoModelForCausalLM.from_pretrained(embed_model_path).to(self.device)
            self.embed_model.eval()
        elif embed_scheme == "qwen3-emb":
            self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_path, use_fast=True, legacy=False)
            self.embed_model = AutoModel.from_pretrained(embed_model_path).to(self.device)
            if self.encoder_latent_dim == 32:
                self.emb_sigma = 2.173140287399292
            elif self.encoder_latent_dim == 64:
                self.emb_sigma = 2.690504312515259
            elif self.encoder_latent_dim == 128:
                self.emb_sigma = 2.793964147567749
            elif self.encoder_latent_dim == 256:
                self.emb_sigma = 2.873631238937378
            elif self.encoder_latent_dim == 768:
                self.emb_sigma = 2.690504312515259
            self.embed_model.eval()


    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss = self.compute_loss(model, inputs, return_outputs=True)

        return (loss.detach(), None, None)
    
    @torch.no_grad()
    def embed(self, inputs):
        if self.embed_scheme == 'gpt2-last':
            embed = self.embed_model(inputs['input_ids'], output_hidden_states=True).hidden_states[0]
            return F.normalize(embed, dim=-1)
        elif self.embed_scheme == 'qwen3-emb':
            embed = self.embed_model(inputs['input_ids'], output_hidden_states=True).last_hidden_state
            embed = embed[:, :, :self.encoder_latent_dim] / self.emb_sigma
            return embed

    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        embedding = self.embed(inputs)
        # forward
        t = torch.rand(embedding.shape[0], device=embedding.device) * (1. - self.time_epsilon) + self.time_epsilon
        alpha_t, sigma_t = self.scheduler.marginal_prob(t)
        alpha_t, sigma_t = expand_dims(alpha_t, embedding.dim()), expand_dims(sigma_t, embedding.dim())
        noise = torch.randn_like(embedding, device=embedding.device)
        z_t = alpha_t * embedding + sigma_t * noise

        # mask = inputs.get('attn_masks', None)
        mask = None
        if mask is not None:
            mask = _prepare_4d_attention_mask_for_sdpa(
                mask=mask, dtype=embedding.dtype, tgt_len=inputs['input_ids'].shape[-1]
            )
        pred = model(z_t, self.scheduler.get_noiseLevel(t, 'alpha_t'), mask)

        if mask is None:
            if self.parameterization == "v":
                v = alpha_t * noise - sigma_t * embedding
                loss = torch.square(v - pred)
                return loss.mean()
            elif self.parameterization == "data":
                loss = torch.square(embedding - pred)
                return loss.mean()
        else:
            per_tok_mse = torch.square(v - pred).mean(dim=-1)
            valid = inputs['attn_masks'].float()
            loss = (per_tok_mse * valid).sum() / valid.sum().clamp_min(1.0)
            return loss


class CustomCollator:
    @torch.no_grad()
    def __call__(self, examples):
        input_ids = [torch.tensor(x["input_ids"]) for x in examples]
        attn_masks = [torch.tensor(x["attention_mask"]) for x in examples]

        input_ids = torch.vstack(input_ids)
        attn_masks = torch.vstack(attn_masks)
        # -100

        return dict(input_ids=input_ids, labels=input_ids, attn_masks=attn_masks)


def train(args):
    global local_rank

    transformers.set_seed(42)

    # model_args, data_args, training_args = get_args(args.train_config)
    training_args = get_args(args)
    local_rank = training_args.local_rank

    config = DITConfig.from_json_file(training_args.model_config)
    rank0_print(config)

    model = DIT(config)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    rank0_print(f"{params_m:.2f} M parameters")

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"

    data_collator = CustomCollator()

    rank0_print("Loading data...")
    max_seq_len = config.max_seq_len
    tmp_tokenizer = AutoTokenizer.from_pretrained(training_args.embed_model_path, use_fast=True, legacy=False)
    train_dataset, eval_dataset = get_dataset(training_args, max_seq_len, tmp_tokenizer)
    del tmp_tokenizer
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    rank0_print("Starting train...")
    trainer = DLMTrainer(model=model,
                         args=training_args,
                         embed_scheme=training_args.embed_scheme,
                         embed_model_path=training_args.embed_model_path,
                         encoder_latent_dim=config.latent_dim,
                         parameterization='v',
                         **data_module)
    trainer.train(resume_from_checkpoint=training_args.resume)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="", help="Path to training config yaml file")
    parser.add_argument("--num_gpus", type=int, default=1, help="")
    add_training_overrides(parser)
    args = parser.parse_args()
    train(args)