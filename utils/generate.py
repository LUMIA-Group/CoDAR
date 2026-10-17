from dataclasses import dataclass
import datasets
import pathlib
from typing import Dict, Sequence
from tqdm import tqdm

import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

import sys
import argparse

sys.path.append(str(pathlib.Path(__file__).parents[1]))
from train.dit_args import get_args
from model.modeling_dit import DIT
from model.configuration_gpt2 import GPT2Config
from model.modeling_gpt2 import GPT2LMHeadModel
# from model.modeling_gpt2_pref import GPT2LMHeadModel
from data.data import get_dataset
from utils.noise_scheduler import NoiseScheduleVP, expand_dims
from utils.dpm_solver import DPM_Solver

local_rank = None

# import swanlab

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class AncestralSampler:
    """Ancestral Sampler"""
    def __init__(self, noise_scheduler: NoiseScheduleVP, time_steps: torch.Tensor, model_pred_data=False, self_cond=False):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.self_cond = self_cond

    def sampling(self, model, z_T, ):
        x = z_T
        bs = z_T.shape[0]
        for i in tqdm(range(len(self.t_array))):
            t = self.t_array[i]
            s = self.s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            vec_t = torch.ones(bs, device=x.device) * t

            # noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            # noise_level = torch.ones(bs, device=x.device) * t

            pred_v = model(x, self.noise_scheduler.get_noiseLevel(vec_t, 'alpha_t'))

            # Compute x_0 from predicted velocity
            x_0 = alpha_t * x - sigma_t * pred_v

            x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), x_0.dim()) * x_0

            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * torch.randn_like(x)

        return x_mean

class Generator:
    """
    Generator
    """

    def __init__(
        self,
        *args,
        embed_scheme = None,
        embed_model_path = None,
        diffusion_ckpt = None,
        decoder = None,
        decoder_tokenizer = None,
        encoder_latent_dim = None,
        solver = "ancestral",
        steps = None,
        temperature = None,
        hidden_size = None,
        context = False,
        dataset=None,
        **kwargs,
    ):
        self.scheduler = NoiseScheduleVP(schedule="cosine")
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.time_epsilon = 1e-5
        self.decoder = decoder
        self.decoder.to(self.device).eval()
        self.decoder_tokenizer = decoder_tokenizer
        self.embed_scheme = embed_scheme
        self.encoder_latent_dim = encoder_latent_dim
        self._init_embed_model(embed_scheme, embed_model_path)
        self._init_diffusion_model(diffusion_ckpt)
        self.solver = solver
        self.steps = steps
        self.temperature = temperature
        self.do_sample = True if self.temperature > 0 else False
        self.ctx_len = 512 if dataset == "owt" else 128
        self.hidden_size = hidden_size
        self.context = context
        if solver == "ancestral":
            self.ancestral_sampler = AncestralSampler(
                noise_scheduler=self.scheduler,
                time_steps=torch.linspace(self.scheduler.T, self.time_epsilon, self.steps, device=self.device))
        elif solver == "dpm":
            self.dpm_solver = DPM_Solver(self.diffusion_model, self.scheduler, algorithm_type="dpmsolver")
        else:
            raise ValueError(solver)
        

    def _init_embed_model(self, embed_scheme, embed_model_path):
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

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
            else:
                raise ValueError()
            self.embed_model.eval()
    
    def _init_diffusion_model(self, ckpt_path):
        diffusion_model = DIT.from_pretrained(ckpt_path)
        diffusion_model.to(self.device).eval()
        self.diffusion_model = diffusion_model

    
    @torch.no_grad()
    def embed(self, inputs):
        if self.embed_scheme == 'gpt2-last':
            embed = self.embed_model(inputs['input_ids'], output_hidden_states=True).hidden_states[0]
            return F.normalize(embed, dim=-1)
        elif self.embed_scheme == 'qwen3-emb':
            embed = self.embed_model(inputs['input_ids'], output_hidden_states=True).last_hidden_state
            embed = embed[:, :, :self.encoder_latent_dim] / self.emb_sigma
            return embed

    @torch.no_grad()
    def uncond_generate(
        self,
        num_samples: int = 1,
        **kwargs,
    ):
        # input_ids = torch.tensor([[151643] * num_samples], device='cuda', dtype=torch.long)
        input_ids = torch.full(size=(num_samples, 1), fill_value=151643, device='cuda', dtype=torch.long)
        # Ancestral Sampler
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            ratio = 1
            if self.solver == "ancestral":
                diff_embed = self.ancestral_sampler.sampling(self.diffusion_model, z_T=torch.randn([num_samples, self.ctx_len // ratio, self.hidden_size], device='cuda'),)
            elif self.solver == "dpm":
                diff_embed = self.dpm_solver.sample(
                    x=torch.randn([num_samples, self.ctx_len // ratio, self.hidden_size], device='cuda'),
                    steps=self.steps,
                )
        # output_ids = self.decoder(diff_embed).argmax(-1)
        # return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        output_ids = self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=diff_embed,
            # diffusion_output=diff_embed,
            # decoder_input_ids=input_ids,
            max_new_tokens=self.ctx_len,
            temperature=self.temperature,
            do_sample=self.do_sample,
            use_cache=True,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
        )
        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    @torch.no_grad()
    def traj(
        self,
        num_samples: int = 1,
        **kwargs,
    ):
        num_steps = 100
        input_ids = torch.full(size=(num_steps+1, 1), fill_value=151643, device='cuda', dtype=torch.long)
        # Ancestral Sampler
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            diff_embed, intermediate = self.dpm_solver.sample(
                x=torch.randn([1, self.ctx_len, 64], device='cuda'),
                steps=num_steps,
                return_intermediate=True
            )
        intermediate = torch.vstack(intermediate).to(diff_embed)
        # print(intermediate.shape)
        output_ids = self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=intermediate,
            # diffusion_output=diff_embed,
            # decoder_input_ids=input_ids,
            max_new_tokens=self.ctx_len,
            temperature=1.,
            do_sample=self.do_sample,
            use_cache=True,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
        )
        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    @torch.no_grad()
    def test_decoder(
        self,
        token_list
    ):
        tokens = torch.tensor(token_list, dtype=torch.long, device="cuda")
        input_ids = torch.full(size=(len(token_list), 1), fill_value=151643, device='cuda', dtype=torch.long)
        embeds = self.embed({"input_ids": tokens})
        embeds += 0.1 * torch.rand_like(embeds)
        # for _ in range(64):
        #     logits = self.decoder(
        #     input_ids=input_ids,
        #     encoder_hidden_states=embeds,)
        #     next_tokens = self.sample_next_token(logits.logits[:, -1], temperature=0.0, )
        #     input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        output_ids = self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=embeds,
            # decoder_input_ids=input_ids,
            max_new_tokens=128,
            temperature=1.,          # since you used temperature=0.0 in sampling
            do_sample=True,          # greedy decoding
            use_cache=True,
            pad_token_id=self.decoder_tokenizer.pad_token_id,         # set appropriately
            eos_token_id=self.decoder_tokenizer.eos_token_id,         # set appropriately
        )

        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    
    @torch.no_grad()
    def test_diffusion(
        self,
        token_list,
        t: float,
    ):
        tokens = torch.tensor(token_list, dtype=torch.long, device="cuda")
        embeds = self.embed({"input_ids": tokens})
        sampler = AncestralSampler(
            noise_scheduler=self.scheduler,
            time_steps=torch.linspace(t, self.time_epsilon, 250, device=self.device))
        t = torch.tensor([t], device="cuda")
        alpha_t, sigma_t = self.scheduler.marginal_prob(t)
        alpha_t, sigma_t = expand_dims(alpha_t, embeds.dim()), expand_dims(sigma_t, embeds.dim())
        noise = torch.randn_like(embeds, device=embeds.device)
        z_t = alpha_t * embeds + sigma_t * noise
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            sampled_embeds = sampler.sampling(self.diffusion_model, z_T=z_t)
        mse = torch.square(sampled_embeds - embeds).mean().cpu().item()
        import pdb; pdb.set_trace()
        mean_of_diff = (sampled_embeds - embeds).mean().abs().cpu().item()



    def _top_k_top_p_filtering(self, logits: torch.Tensor,
                           top_k: int | None = None,
                           top_p: float | None = None,
                           min_tokens_to_keep: int = 1) -> torch.Tensor:
        """
        Filter a batch of logits (shape [bs, vocab]) using top-k and/or top-p.
        Ensures at least `min_tokens_to_keep` tokens remain per row.
        """
        # Top-k
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth_vals, float("-inf"))

        # Top-p (nucleus)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens where cumulative prob exceeds top_p
            to_remove = cumprobs > top_p
            # Always keep at least a few highest-prob tokens
            if min_tokens_to_keep > 0:
                to_remove[..., :min_tokens_to_keep] = False
            # Scatter back to original index order
            to_remove_orig = torch.zeros_like(to_remove).scatter(1, sorted_idx, to_remove)
            logits = logits.masked_fill(to_remove_orig, float("-inf"))

        return logits

    def sample_next_token(
        self,
        logits_last: torch.Tensor,          # shape [bs, vocab]
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        disallow_mask: torch.Tensor | None = None,  # optional bool mask [bs, vocab]
        generator: torch.Generator | None = None,   # for reproducibility
    ) -> torch.Tensor:
        """
        Returns next-token ids with shape [bs], dtype torch.long.
        - If temperature==0 -> greedy argmax.
        - Otherwise samples from softmax after optional top-k/top-p filtering.
        """
        logits = logits_last.clone()

        # Optionally mask out tokens you never want to sample (e.g., pad)
        if disallow_mask is not None:
            logits = logits.masked_fill(disallow_mask, float("-inf"))

        # Greedy
        if temperature == 0 or temperature is None:
            return torch.argmax(logits, dim=-1)

        # Scale by temperature, then filter
        logits = logits / max(temperature, 1e-8)
        logits = self._top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, min_tokens_to_keep=1)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        return next_ids

class CustomCollator:
    @torch.no_grad()
    def __call__(self, examples):
        input_ids = [torch.tensor(x["input_ids"]) for x in examples]

        input_ids = torch.vstack(input_ids)

        return dict(input_ids=input_ids, labels=input_ids)


def train(args):
    # from safetensors.torch import load_file

    # transformers.set_seed(42)

    # model_args, data_args, training_args = get_args(args.train_config)
    training_args = get_args(args)
    local_rank = training_args.local_rank

    config = GPT2Config.from_json_file(training_args.model_config)
    # config = LlamaConfig.from_json_file(training_args.model_config)
    rank0_print(config)

    context = args.context
    decoder_checkpoint = pathlib.Path(args.decoder_checkpoint)
    if not decoder_checkpoint.is_dir():
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_checkpoint}")
    model = GPT2LMHeadModel.from_pretrained(decoder_checkpoint)
    # model = torch.nn.Linear(args.hidden_size, config.vocab_size, bias=False)
    # from safetensors.torch import load_file

    # model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(training_args.embed_model_path)

    # params_m = sum(p.numel() for name, p in model.named_parameters() if (p.requires_grad and "wte" not in name)) / 1e6
    # rank0_print(f"{params_m:.2f} M parameters")

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"

    data_collator = CustomCollator()

    max_seq_len = config.n_ctx

    diff_ckpt = pathlib.Path(args.diffusion_checkpoint)
    if not diff_ckpt.is_dir():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diff_ckpt}")

    generator = Generator(
        model=model,
        args=training_args,
        embed_scheme=training_args.embed_scheme,
        embed_model_path=training_args.embed_model_path,
        diffusion_ckpt=diff_ckpt,
        decoder=model,
        decoder_tokenizer=tokenizer,
        encoder_latent_dim=config.encoder_latent_dim,
        solver=args.solver,
        steps=args.sampling_steps,
        temperature=args.temperature,
        hidden_size=args.hidden_size,
        context=context,
        dataset=args.dataset,
    )


    if not args.test_dec:
        samples = []
        for _ in tqdm(range(10)):
            samples.extend(generator.uncond_generate(num_samples=100))
        
        import pickle
        with open(f"outputs/samples/samples_512_cos250K{"_dpm" if args.solver=="dpm" else ""}_{args.sampling_steps}_fixbos_t{args.temperature:.2f}{"_ctx" if context else ""}_{args.hidden_size}{"" if args.dataset=="owt" else "_lm1b"}.pkl", mode="wb") as f:
            pickle.dump(samples, f)
    else:
        train_dataset, eval_dataset = get_dataset(training_args, max_seq_len, generator.decoder_tokenizer)
        subset = []
        for i in range(1000):
            subset.append(train_dataset[i]['input_ids'].tolist())

        samples = []
        for i in tqdm(range(20)):
            samples.extend(generator.test_decoder(subset[i*50:(i+1)*50]))
        import pickle
        with open(f"test_decoder_t{args.temperature:.2f}.pkl", mode="wb") as f:
            pickle.dump(samples, f)
    # subset = tokenizer.batch_decode(subset, skip_special_tokens=True)
    # with open(f"samples_subset_owt_{config.encoder_latent_dim}_fixbos.pkl", mode="wb") as f:
    #     pickle.dump(subset, f)

    # PPL
    # train_dataset, eval_dataset = get_dataset(training_args, max_seq_len, generator.decoder_tokenizer)
    # subset = []
    # for i in range(1000):
    #     subset.append(train_dataset[i]['input_ids'].tolist())

    # ppl = []
    # for i in tqdm(range(20)):
    #     ppl.append(generator.ppl(subset[i*50:(i+1)*50]))
    # print(sum(ppl) / len(ppl))
    # print(ppl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="", help="Path to training config yaml file")
    parser.add_argument("--num_gpus", type=int, default=1, help="")
    parser.add_argument("--solver", type=str, default="ancestral", help="")
    parser.add_argument("--sampling_steps", type=int, default=250, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--hidden_size", type=int, default=64, help="")
    parser.add_argument("--dataset", type=str, default="owt", help="")
    parser.add_argument("--context", action='store_true')
    parser.add_argument("--test_dec", action='store_true')
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--diffusion_checkpoint", required=True)
    parser.add_argument("--embed_model_path", required=True)
    args = parser.parse_args()
    train(args)