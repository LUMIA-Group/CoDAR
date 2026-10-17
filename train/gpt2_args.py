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
    resume: Optional[str] = field(default=None)


def get_args(args):

    # parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_yaml_file(args.yaml_path)
    parser = transformers.HfArgumentParser(TrainingArguments)
    training_args = parser.parse_yaml_file(args.train_config)[0]

    training_args.num_gpus = args.num_gpus

    return training_args