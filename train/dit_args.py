from dataclasses import dataclass, field
from typing import Optional, Tuple

import transformers

# @dataclass
# class ModelArguments:
#     model_config: Optional[str] = field(default=None)


# @dataclass
# class DataArguments:
#     data_path: str = field(
#         default="sharegpt_clean.json",
#         metadata={"help": "Path to the training data."},
#     )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    num_gpus: int = field(default=1)
    # data
    dataset: Optional[str] = field(default=None)
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    # model
    model_config: Optional[str] = field(default=None)
    embed_scheme: Optional[str] = field(default=None)
    embed_model_path: Optional[str] = field(default=None)
    resume: Optional[str] = field(default=None)


def get_args(args):

    # parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_yaml_file(args.yaml_path)
    parser = transformers.HfArgumentParser(TrainingArguments)
    training_args = parser.parse_yaml_file(args.train_config)[0]

    training_args.num_gpus = args.num_gpus

    for name in ("embed_model_path", "data_path", "output_dir", "run_name"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(training_args, name, value)
    if getattr(args, "resume_from_checkpoint", None) is not None:
        training_args.resume = args.resume_from_checkpoint
    return training_args


def add_training_overrides(parser):
    """Portable path overrides shared by the two main training entry points."""
    for name in ("embed_model_path", "data_path", "output_dir", "run_name"):
        parser.add_argument(f"--{name}", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Explicit checkpoint directory; fresh training is the default.")
