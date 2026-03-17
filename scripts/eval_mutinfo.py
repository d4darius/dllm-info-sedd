"""
Info-SEDD Mutual Information Evaluation Script
Evaluates the mutual information of summarizer outputs given articles.

How to run:
===========
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm

# Local:
python scripts/eval_mutinfo.py \
    --model_name_or_path <path-to-dllm-model> \
    --dataset_file data.jsonl \
    --variant "j" \
    --lora True  # (if evaluating a PEFT fine-tuned model)

# Slurm Cluster:
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
python scripts/eval_mutinfo.py \
    --model_name_or_path <path-to-dllm-model> \
    --dataset_file data.jsonl \
    --variant "j"
"""

import functools
import os
import sys
import json
from dataclasses import dataclass, field

# Add repo root to Python Path so 'import dllm' works out of the box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import accelerate
import torch
import transformers

import dllm
from dllm.core.trainers.info_sedd import InfoSEDDConfig, InfoSEDDTrainer


logger = dllm.utils.get_default_logger(__name__)


@dataclass
class EvalArguments(dllm.utils.DataArguments):
    dataset_file: str = field(
        default="data.jsonl",
        metadata={"help": "Path to a JSONL dataset file with 'article' and 'summary' keys."}
    )
    mc_estimates: int = field(
        default=100,
        metadata={"help": "Number of Monte-Carlo estimates (evaluation batches) to execute."}
    )
    max_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length allowed to prevent CUDA OOM."}
    )
    eval_name: str = field(
        default="default_model",
        metadata={"help": "Name of the model being evaluated (used for the JSON output)."}
    )


def load_mutinfo_dataset_for_eval(data_path, tokenizer, max_tokens):
    import datasets
    # Load from local JSONL
    ds = datasets.load_dataset("json", data_files={"test": data_path})["test"]

    # Map articles to prompts and summaries to responses
    def map_to_messages(row):
        # Format as chat messages so default_sft_map_fn parses prompt_len correctly
        messages = [
            {"role": "user", "content": row.get("article", row.get("x", ""))}
        ]
        
        # Determine the content of the assistant (the summary Y)
        summary = row.get("summary", row.get("y", ""))
        if summary:
            messages.append({"role": "assistant", "content": summary})
            
        return {"messages": messages}
    
    ds = ds.map(map_to_messages, desc="Formatting to messages")

    # Customize mapping to ensure prompt_len is always calculated and returned for InfoSEDD
    def map_fn(row):
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        # We don't mask prompt loss here, InfoSEDD handles it dynamically!
        return {
            "input_ids": prompt_response_tokens, 
            "labels": prompt_response_tokens.copy(),
            "prompt_len": len(prompt_tokens)
        }
    
    ds = ds.map(
        map_fn,
        remove_columns=ds.column_names,
        desc="Tokenizing and extracting varying prompt lengths"
    )

    # ---------------------------------------------------------
    # FIX: Filter out sequences that will blow up the GPU memory
    # ---------------------------------------------------------
    original_size = len(ds)
    ds = ds.filter(lambda row: len(row["input_ids"]) <= max_tokens, desc="Filtering long sequences")
    filtered_size = len(ds)
    
    print(f"\n[INFO] Filtered out {original_size - filtered_size} sequences longer than {max_tokens} tokens.\n")
    # ---------------------------------------------------------

    var_indices = ds[0]["prompt_len"]

    # The original implementation statically extracts the prompt boundary
    ds = ds.select_columns([col for col in ds.column_names if col != "prompt_len"])
    
    return {"test": ds, "var_indices": var_indices}


def evaluate():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser((dllm.utils.ModelArguments, EvalArguments, InfoSEDDConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)


    if "eval_batch_size" not in training_args.__dict__:
        training_args.per_device_eval_batch_size = 8

    # Prevent Trainer from discarding 'prompt_len' since forward() doesn't explicitly expect it
    training_args.remove_unused_columns = False

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.models.get_model(model_args=model_args)

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.models.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset_output = load_mutinfo_dataset_for_eval(data_args.dataset_file, tokenizer, data_args.max_tokens)
        dataset = dataset_output["test"]
        
        # Original diffusion implementation relies on extracting var_indices statically
        # and passing it directly down into the training/evaluation config.
        training_args.var_indices = dataset_output["var_indices"]
        
    # ----- Evaluation -------------------------------------------------------------
    logger.info(f"Start Information Metrics Eval over variant: {training_args.variant}...")
    
    if hasattr(training_args, "eval_steps"):
        training_args.eval_steps = data_args.mc_estimates
        
    trainer = InfoSEDDTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        eval_dataset=dataset,
        data_collator=dllm.utils.collators.PrependBOSWrapper(
            transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
            bos_token_id=tokenizer.bos_token_id,
            label_pad_token_id=-100
        )
    )

    # Limit batches to mc_estimates if standard Trainer evaluates the entire dataset
    # We can handle this by slicing the evaluation dataset.
    total_samples_needed = data_args.mc_estimates * training_args.per_device_eval_batch_size
    if len(dataset) > total_samples_needed:
        trainer.eval_dataset = dataset.select(range(total_samples_needed))

    eval_metrics = trainer.evaluate()
    
    # In InfoSEDD formulation, the loss directly approaches Mutual Information (in nats)
    mutinfo_estimate = eval_metrics.get("eval_loss", float("nan"))
    logger.info(f"Mutual information estimate (nats): {mutinfo_estimate}")
    
    # Save the estimate output to a JSON file
    result_dir = training_args.output_dir if training_args.output_dir else "mi_results"
    full_res_path = os.path.join(result_dir, "infosedd", training_args.variant)
    os.makedirs(full_res_path, exist_ok=True)
    
    # Create a nice dictionary with all your metadata
    results_dict = {
        "model_name": data_args.eval_name,
        "variant": training_args.variant,
        "mc_estimates_used": data_args.mc_estimates,
        "max_tokens_limit": data_args.max_tokens,
        "mutinfo_estimate_nats": mutinfo_estimate
    }
    
    # Name the file dynamically based on the model name
    json_filename = f"{data_args.eval_name}_mutinfo.json"
    
    with open(os.path.join(full_res_path, json_filename), "w") as f:
        json.dump(results_dict, f, indent=4)
        
    logger.info(f"Final estimate securely saved to {os.path.join(full_res_path, json_filename)}")

if __name__ == "__main__":
    evaluate()
