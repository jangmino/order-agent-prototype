# A General Evaluation Script using PEFT trained model for food order understanding
import os
from dataclasses import dataclass, field
from typing import Optional
import re
import numpy as np
import pandas as pd

import sys
import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset, Dataset, load_metric
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TextStreamer,
    logging as hf_logging,
)
import logging
from trl import SFTTrainer

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
    model_type: Optional[str] = field(
        default="mistral",
        metadata={"help": "You should choose one of mistral, llama2, phi2, midm"},
    )
    eval_result_file: Optional[str] = field(
        default="eval_df.pkl",
        metadata={"help": "evaluation result file name (pickle from pandas)"},
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

        prompt_template = """<start_of_turn>user\n{System} {User}<end_of_turn>\n<start_of_turn>model\n"""

        default_system_msg = "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:"

        text = prompt_template.format(
            System=default_system_msg, User=example["input"], Agent=example["output"]
        )

        return text
    def _prepare_sample_text_mistral(example):
        """Prepare the text from a sample of the dataset."""

        prompt_template = """[INST] {System} {User} [/INST] """

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

        prompt_template = f"{B_INST} {B_SYS}{{System}}{E_SYS}{{User}} {E_INST} "

        default_system_msg = (
            "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다."
        )

        text = prompt_template.format(
            System=default_system_msg, User=example["input"], Agent=example["output"]
        )

        return text

    def prepare_sample_text_phi2(example):
        """Prepare the text from a sample of the dataset."""
        prompt_template = """Instruct:{System}{User}\nOutput:"""

        default_system_msg = "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:"

        text = prompt_template.format(
            System=default_system_msg, User=example["input"], Agent=example["output"]
        )

        return text

    def prepare_sample_text_midm(example):
        """Prepare the text from a sample of the dataset."""
        prompt_template = """###System;{System}\n###User;{User}\n###Midm;"""

        default_system_msg = (
            "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다."
        )

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


def wrapper_generate(tokenizer, model, input_prompt, do_stream=False):
    def remove_turn_markers(text):
        # '<start_of_turn>'과 '<end_of_turn>' 문자열을 빈 문자열로 대체하여 제거
        cleaned_text = re.sub(r'<start_of_turn>|<end_of_turn>', '', text)
        return cleaned_text

    data = tokenizer(input_prompt, return_tensors="pt")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_ids = data.input_ids[..., :-1]
    with torch.no_grad():
        pred = model.generate(
            input_ids=input_ids.cuda(),
            streamer=streamer if do_stream else None,
            use_cache=True,
            max_new_tokens=float("inf"),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded_text = tokenizer.batch_decode(pred, skip_special_tokens=True)
    # midm 결과에 대해 특별 처리
    decoded_text = decoded_text[0].replace("<[!newline]>", "\n")
    
    # gemma 결과에 대해 특별 처리
    input_prompt = remove_turn_markers(input_prompt)
    return decoded_text[len(input_prompt) :]


############
def evaluate():
    # script_args = tyro.cli(ScriptArguments)
    # logging.info(script_args)
    
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    trained_model = AutoPeftModelForCausalLM.from_pretrained(
        training_args.output_dir,
        quantization_config=bnb_config,
        device_map="auto",  # {"": Accelerator().local_process_index},
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    trained_model.config.pad_token_id = tokenizer.pad_token_id

    dataset = local_dataset(model_args.dataset_name)
    test_dataset = dataset["test"]
    test_dataset = test_dataset.map(input_converter)

    prepare_sample_text = function_prepare_sample_text(model_args.model_type)

    logging.info(f"Start Evaluating... with {model_args.model_type}")
    trained_model.eval()
    predict_result = []
    for i in tqdm(range(len(test_dataset))):
        # for i in tqdm(range(10)):
        ret = wrapper_generate(
            tokenizer=tokenizer,
            model=trained_model,
            input_prompt=prepare_sample_text(test_dataset[i]),
        )
        predict_result.append(
            {
                "idx": i,
                "body": test_dataset[i]["input"],
                "label": test_dataset[i]["output"],
                "predict": ret,
            }
        )

    bleu_metric = load_metric("sacrebleu")

    for item in predict_result:
        bleu_metric.add(prediction=item["predict"], reference=[item["label"]])

    info = bleu_metric.compute(smooth_method="floor", smooth_value=0)
    info["precisions"] = [np.round(p, 2) for p in info["precisions"]]
    eval_df = pd.DataFrame.from_dict(info, orient="index", columns=["Value"])
    logging.info(f"{eval_df=}")
    eval_df.to_pickle(model_args.eval_result_file)

    logging.info("End Evaluating...")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    evaluate()
