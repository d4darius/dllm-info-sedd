"""
LoRA Fine-Tuning Script for dLLM on JSONL datasets

How to run:
===========
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm

# Local (1 GPU) 
python scripts/train_lora.py \
    --model_name_or_path dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
    --dataset_file data.jsonl \
    --output_dir ./lora_outputs \
    --eval_strategy no \
    --lora True

# Slurm Cluster (FSDP):
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "scripts/train_lora.py" \
    --model_name_or_path dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
    --dataset_file data.jsonl \
    --output_dir ./lora_outputs \
    --eval_strategy no \
    --lora True
"""

import functools
import os
import sys
from dataclasses import dataclass, field

# Add repo root to Python Path so 'import dllm' works out of the box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import accelerate
import torch
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class TrainArguments(dllm.utils.DataArguments):
    dataset_file: str = field(
        default="data.jsonl",
        metadata={"help": "Path to a JSONL dataset file with 'article' and 'summary' keys."}
    )
    random_length_ratio: float = field(
        default=0.01,
        metadata={
            "help": (
                "The probability of randomly cut sequences during training. "
            )
        },
    )


def load_fine_tuning_dataset(data_path, tokenizer):
    import datasets
    # Load from local JSONL
    ds = datasets.load_dataset("json", data_files={"train": data_path})["train"]

    # Map articles to prompts and summaries to responses
    def map_to_messages(row):
        messages = [{"role": "user", "content": row.get("article", row.get("x", ""))}]
        summary = row.get("summary", row.get("y", ""))
        if summary:
            messages.append({"role": "assistant", "content": summary})
        return {"messages": messages}
    
    # Use multiple processes for faster mapping
    num_proc = os.cpu_count()
    ds = ds.map(map_to_messages, desc="Formatting to messages", num_proc=num_proc)

    # In fine-tuning, `mask_prompt_loss=True` configures standard autoregressive/seq2seq
    # masking where the prompt (article) is not penalized. Alternatively, pass False
    # to jointly train over the entire sequence. We use default (True) for summarization
    # unless you want MDLM to learn the unconditional distribution of the articles too.
    map_fn = functools.partial(
        dllm.utils.data.default_sft_map_fn, 
        tokenizer=tokenizer, 
        mask_prompt_loss=True 
    )
    
    ds = ds.map(
        map_fn,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing for fine-tuning"
    )
    return {"train": ds}


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (dllm.utils.ModelArguments, TrainArguments, dllm.core.trainers.MDLMConfig)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # FIX: If eval_strategy is set to perform evaluation (e.g., 'steps' or 'epoch') 
    # but no eval dataset is provided in this script, the Trainer will raise a ValueError.
    # We default it to 'no' for this specific fine-tuning script.
    if training_args.eval_strategy != "no":
        logger.info(f"Changing eval_strategy from {training_args.eval_strategy} to 'no' because no evaluation dataset is provided.")
        training_args.eval_strategy = "no"

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    # Initialize model weights from scratch or pre-trained based on model_name_or_path
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, dtype=torch.bfloat16
        )

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    
    # ----- PEFT: LoRA -------------------------------------------------------------
    # `load_peft` automatically attaches adapters correctly if `model_args.lora` is True
    model = dllm.utils.load_peft(model=model, model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = load_fine_tuning_dataset(data_args.dataset_file, tokenizer)
        
        # NOTE: If memory is an issue, uncomment grouping logic:
        # dataset = dataset.map(
        #     functools.partial(
        #         dllm.utils.tokenize_and_group,
        #         tokenizer=tokenizer,
        #         text_field=data_args.text_field,
        #         seq_length=data_args.max_length,
        #     ),
        #     batched=True, ...
        # )

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start fine-tuning...")
    
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        processing_class=tokenizer,  # Replaces 'tokenizer' to fix FutureWarning
        train_dataset=dataset["train"],
        args=training_args,
        data_collator=(
            dllm.utils.collators.RandomTruncateWrapper(
                dllm.utils.collators.PrependBOSWrapper(
                    transformers.DataCollatorForSeq2Seq(
                        tokenizer,
                        return_tensors="pt",
                        padding=True,
                    ),
                    bos_token_id=tokenizer.bos_token_id,
                    label_pad_token_id=-100
                ),
                random_length_ratio=data_args.random_length_ratio,
            )
        ),
    )
    trainer.train()
    
    # Save the final LoRA weights and tokenizer configuration
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
