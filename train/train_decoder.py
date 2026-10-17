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
from model.configuration_gpt2 import GPT2Config
from model.modeling_gpt2 import GPT2LMHeadModel
from data.data import get_dataset
from utils.noise_scheduler import NoiseScheduleVP, expand_dims

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class DecoderTrainer(Trainer):
    """
    AR Decoder Trainer.
    """

    def __init__(
        self,
        *args,
        embed_scheme = None,
        embed_model_path = None,
        encoder_latent_dim = None,
        embedding_noise_scale: float = 0.0,
        loss_weight_type: str = "scheduler",  # "ones"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = NoiseScheduleVP(schedule="linear")
        self.time_epsilon = 1e-5
        self.loss_weight_type = loss_weight_type
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.embed_scheme = embed_scheme
        self.encoder_latent_dim = encoder_latent_dim
        self.embedding_noise_scale = embedding_noise_scale
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

    def _preprocess_inputs(self, inputs):
        pass

    def _postprocess_outputs(self, outputs):
        pass

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = -self.scheduler.weight(t).unsqueeze(1).repeat(1, l)  # b, 1
        elif self.loss_weight_type == "ones":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss = self.compute_loss(model, inputs, is_eval=True)

        return (loss.detach(), None, None)
    
    @torch.no_grad()
    def embed(self, inputs):
        if self.embed_scheme == 'gpt2-last':
            embed = self.embed_model(inputs['input_ids'], output_hidden_states=True).hidden_states[0]
            return F.normalize(embed, dim=-1)
        elif self.embed_scheme == 'qwen3-emb':
            embed = self.embed_model(inputs['input_ids'][:, 1:], output_hidden_states=True).last_hidden_state
            embed = embed[:, :, :self.encoder_latent_dim] / self.emb_sigma
            # return F.normalize(embed, dim=-1)
            return embed

    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor = None,
        **kwargs,
    ):
        mask = inputs['attn_masks']

        embedding = self.embed(inputs)
        if self.embedding_noise_scale > 0:
            embedding = embedding + self.embedding_noise_scale * torch.rand_like(embedding)
        embedding += 0.1 * torch.rand_like(embedding)
        is_eval = kwargs.get("is_eval", False)

        loss = model(
            input_ids=inputs['input_ids'],
            attention_mask=mask,
            encoder_hidden_states=embedding,
            labels=inputs['labels'],
            # encoder_attention_mask=mask,
            num_items_in_batch=num_items_in_batch).loss

        """ logits = model(
            inputs_embeds=embedding,
            attention_mask=mask,
            num_items_in_batch=num_items_in_batch).logits
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = inputs['labels'][:, 1:].contiguous()       # [B, T-1]

        # token-level losses (no reduction yet)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),   # [B*(T-1), V]
            shift_labels.view(-1),).mean() """

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        return loss

pad_id = 151643 # qwen2 tokenizer pad id
# pad_id = 50256
mask_id = 151644 # <|im_start|>
class CustomCollatorMask:
    @torch.no_grad()
    def __call__(self, examples):
        input_ids = [torch.tensor(x["input_ids"]) for x in examples]

        input_ids = torch.vstack(input_ids)
        bos_tensor = torch.full((input_ids.size(0), 1), pad_id, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)

        attn_masks = [torch.tensor(x["attention_mask"]) for x in examples]
        attn_masks = torch.vstack(attn_masks)
        attend_bos_tensor = torch.full((attn_masks.size(0), 1), 1, dtype=attn_masks.dtype, device=attn_masks.device)
        attn_masks = torch.cat([attend_bos_tensor, attn_masks], dim=-1)
        labels = input_ids.clone()
        labels[attn_masks == 0] = -100

        # random mask
        sampled = torch.rand_like(input_ids, dtype=torch.float) < 0.3
        protect = (attn_masks == 0)
        bos_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        bos_positions[:, 0] = True
        protect |= bos_positions
        to_mask = sampled & (~protect)  # boolean mask for real corruption; broadcast-safe with bool masks. :contentReference[oaicite:3]{index=3}

        # ----- apply masking -----
        masked_input_ids = input_ids.clone()
        masked_input_ids[to_mask] = mask_id

        return dict(input_ids=masked_input_ids, labels=input_ids, attn_masks=attn_masks)

class CustomCollator:
    @torch.no_grad()
    def __call__(self, examples):
        input_ids = [torch.tensor(x["input_ids"]) for x in examples]

        input_ids = torch.vstack(input_ids)
        bos_tensor = torch.full((input_ids.size(0), 1), pad_id, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)

        attn_masks = [torch.tensor(x["attention_mask"]) for x in examples]
        attn_masks = torch.vstack(attn_masks)
        attend_bos_tensor = torch.full((attn_masks.size(0), 1), 1, dtype=attn_masks.dtype, device=attn_masks.device)
        attn_masks = torch.cat([attend_bos_tensor, attn_masks], dim=-1)
        # labels = input_ids.clone()
        # labels[attn_masks == 0] = -100

        return dict(input_ids=input_ids, labels=input_ids, attn_masks=attn_masks)


def train(args):
    global local_rank

    transformers.set_seed(42)

    # model_args, data_args, training_args = get_args(args.train_config)
    training_args = get_args(args)
    local_rank = training_args.local_rank

    config = GPT2Config.from_json_file(training_args.model_config)
    # config = LlamaConfig.from_json_file(training_args.model_config)
    rank0_print(config)

    model = GPT2LMHeadModel(config)
    # model = torch.nn.Linear(config.encoder_latent_dim, config.vocab_size, bias=False)
    # model = LlamaForCausalLM(config)
    params_m = sum(p.numel() for name, p in model.named_parameters() if (p.requires_grad and "wte" not in name)) / 1e6
    rank0_print(f"{params_m:.2f} M parameters")

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"

    data_collator = CustomCollator()

    rank0_print("Loading data...")
    max_seq_len = config.n_ctx
    # max_seq_len = config.max_position_embeddings
    tmp_tokenizer = AutoTokenizer.from_pretrained(training_args.embed_model_path, use_fast=True, legacy=False)
    train_dataset, eval_dataset = get_dataset(training_args, max_seq_len, tmp_tokenizer)
    del tmp_tokenizer
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    rank0_print("Starting train...")
    trainer = DecoderTrainer(
        model=model,
        args=training_args,
        embed_scheme=training_args.embed_scheme,
        embed_model_path=training_args.embed_model_path,
        encoder_latent_dim=config.encoder_latent_dim,
        embedding_noise_scale=args.embedding_noise_scale,
        **data_module
        )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     **data_module
    #     )
    trainer.train(resume_from_checkpoint=training_args.resume)
    # print(trainer.evaluate())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="", help="Path to training config yaml file")
    parser.add_argument("--num_gpus", type=int, default=1, help="")
    parser.add_argument("--embedding_noise_scale", type=float, default=0.0)
    add_training_overrides(parser)
    args = parser.parse_args()
    train(args)