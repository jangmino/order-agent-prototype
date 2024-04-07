# Fine-Tune mistral7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional
import re

import torch
import sys
import tyro
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging as hf_logging,
)
import logging
from trl import SFTTrainer

# from trl.import_utils import is_xpu_available
from trl.trainer import ConstantLengthDataset


@dataclass
class ScriptArguments:
    cache_dir: Optional[str] = field(
        default="/Jupyter/huggingface/.cache", metadata={"help": "the cache dir"}
    )
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={"help": "the model name"},
    )

    dataset_name: Optional[str] = field(
        default="/Jupyter/dev_src/ASR-for-noisy-edge-devices/data/food-order-understanding-gpt4-30k.json",
        metadata={"help": "the dataset name"},
    )
    subset: Optional[str] = field(
        default="data/finetune", metadata={"help": "the subset to use"}
    )
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(
        default=4000, metadata={"help": "the size of the validation set"}
    )
    streaming: Optional[bool] = field(
        default=False, metadata={"help": "whether to stream the dataset"}
    )
    shuffle_buffer: Optional[int] = field(
        default=5000, metadata={"help": "the shuffle buffer size"}
    )
    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "the sequence length"}
    )
    num_workers: Optional[int] = field(
        default=8, metadata={"help": "the number of workers"}
    )
    model_type: Optional[str] = field(
        default="mistral",
        metadata={"help": "You should choose one of mistral, llama2, phi2, midm"},
    )

    # training_args: TrainingArguments = field(
    #     default_factory=lambda: TrainingArguments(
    #         output_dir="./results",
    #         # max_steps=500,
    #         logging_steps=20,
    #         # save_steps=10,
    #         per_device_train_batch_size=1,
    #         per_device_eval_batch_size=1,
    #         gradient_accumulation_steps=2,
    #         gradient_checkpointing=False,
    #         group_by_length=False,
    #         learning_rate=1e-4,
    #         lr_scheduler_type="cosine",
    #         # warmup_steps=100,
    #         warmup_ratio=0.03,
    #         max_grad_norm=0.3,
    #         weight_decay=0.05,
    #         save_total_limit=3,
    #         save_strategy="epoch",
    #         num_train_epochs=1,
    #         optim="paged_adamw_32bit",
    #         bf16=True,
    #         remove_unused_columns=False,
    #         run_name="sft_mistral",
    #         report_to="wandb",
    #     )
    # )

    packing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use packing for SFTTrainer"}
    )

    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "down_proj",
                "up_proj",
                "gate_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
    )

    merge_with_final_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": "Do only merge with final checkpoint"}
    )


def build_peft_config(peft_config: LoraConfig, model_type):
    """peft_config에 따라 target_modules를 설정한다."""

    if model_type == "mistral" or model_type == "llama2" or model_type == "gemma":
        peft_config.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
        ]
    elif model_type == "phi2":
        peft_config.target_modules = (
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "dense",
                "fc1",
                "fc2",
            ],
        )
    elif model_type == "midm":
        peft_config.target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return peft_config


def chars_token_ratio(dataset, tokenizer, prepare_sample_text, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def input_converter(example):
    """
    주문 문장 데이터셋에서 일부 포맷 변경
    오리지널 데이터셋(food-order-understanding-gpt4-30k.json은 다음처럼 명령, 응답이라는 글자가 포함되어 있다.)
    """

    m = re.match(r"(.*)### 명령: (.*) ### 응답:\n", example["input"], re.DOTALL)

    return {"input": m.group(2) if m else example["input"]}


def function_prepare_sample_text(model_type: str):
    def _prepare_sample_text_gemma(example):
        """Prepare the text from a sample of the dataset."""

        prompt_template = """<start_of_turn>user\n{System} {User}<end_of_turn>\n<start_of_turn>model\n{Agent}"""

        default_system_msg = "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:"

        text = prompt_template.format(
            System=default_system_msg, User=example["input"], Agent=example["output"]
        )

        return text

    def _prepare_sample_text_mistral(example):
        """Prepare the text from a sample of the dataset."""

        prompt_template = """[INST] {System} {User} [/INST] {Agent}"""

        default_system_msg = "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:"

        text = prompt_template.format(
            System=default_system_msg, User=example["input"], Agent=example["output"]
        )

        return text

    def _prepare_sample_text_llama2(example):
        """Prepare the text from a sample of the dataset."""
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

        prompt_template = (
            f"{B_INST} {B_SYS}{{System}}{E_SYS}{{User}} {E_INST} {{Agent}}"
        )

        default_system_msg = "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다."

        text = prompt_template.format(
            System=default_system_msg, User=example["input"], Agent=example["output"]
        )

        return text

    def prepare_sample_text_phi2(example):
        """Prepare the text from a sample of the dataset."""
        prompt_template = """Instruct:{System}{User}\nOutput:{Agent}"""

        default_system_msg = "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:"

        text = prompt_template.format(
            System=default_system_msg, User=example["input"], Agent=example["output"]
        )

        return text

    def prepare_sample_text_midm(example):
        """Prepare the text from a sample of the dataset."""
        prompt_template = """###System;{System}\n###User;{User}\n###Midm;{Agent}"""

        default_system_msg = "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다."

        text = prompt_template.format(
            System=default_system_msg, User=example["input"], Agent=example["output"]
        )

        return text

    if model_type == "mistral":
        return _prepare_sample_text_mistral
    elif model_type == "llama2":
        return _prepare_sample_text_llama2
    elif model_type == "phi2":
        return prepare_sample_text_phi2
    elif model_type == "midm":
        return prepare_sample_text_midm
    elif model_type == "gemma":
        return _prepare_sample_text_gemma
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_datasets(tokenizer, args):
    dataset = local_dataset(args.dataset_name).map(input_converter)
    train_data = dataset["train"]
    valid_data = dataset["test"]

    prepare_sample_text = function_prepare_sample_text(args.model_type)

    chars_per_token = chars_token_ratio(train_data, tokenizer, prepare_sample_text)
    logging.info(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}"
    )

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


############
# Utilities
def local_dataset(dataset_name):
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    return split_dataset


############
def train():
    # model_args = tyro.cli(ScriptArguments)
    # logging.info(model_args)

    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if model_args.merge_with_final_checkpoint:
        # 머지만 하고 종료
        # output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
        output_dir = training_args.output_dir
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            # repo_type="model",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
            # use_auth_token=True,
        )
        model = model.merge_and_unload()

        for param in model.parameters():
            param.data = param.data.contiguous()

        output_merged_dir = os.path.join(
            training_args.output_dir, "final_merged_checkpoint"
        )
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        logging.info("Merge and Safe Seirialization are Done.")
        return

    if training_args.group_by_length and model_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    if training_args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        device_map="auto",  # {"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
        cache_dir=model_args.cache_dir,
    )
    base_model.config.use_cache = False

    peft_config = model_args.peft_config
    peft_config = build_peft_config(peft_config, model_args.model_type)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # training_args = model_args.training_args

    train_dataset, eval_dataset = create_datasets(tokenizer, model_args)

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=model_args.packing,
        max_seq_length=model_args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    logging.info(f"Start Fine Tuning... with {model_args.model_type}")
    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    logging.info(f"final checkpoint saved to {output_dir}")
    trainer.model.save_pretrained(output_dir)

    logging.info("Fine Tuning is done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train()
