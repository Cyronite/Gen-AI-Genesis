#!/usr/bin/env python3
"""
Command-line training script for fine-tuning DeepSeek Coder V2 Lite Instruct model
with QLoRA on tokenized code files.
"""

import os
import argparse
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
from datasets import Dataset
import logging
import yaml
from datetime import datetime
from pathlib import Path
import wandb
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek Coder V2 model with QLoRA")

    # Model args
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-v2-lite-instruct",
                      help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="./trained_model",
                      help="Directory to save the model")

    # Data args
    parser.add_argument("--dataset_dir", type=str, required=True,
                      help="Directory containing tokenized JSON files")
    parser.add_argument("--train_size", type=float, default=0.95,
                      help="Proportion of data to use for training vs. evaluation")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                      help="Maximum sequence length for training")

    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_train_epochs", type=int, default=3)

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Scheduler and optimization
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                      help="Learning rate scheduler: linear, cosine, constant, constant_with_warmup")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")

    # Quantization and hardware optimization
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type for 4-bit: fp4 or nf4")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use flash attention implementation")

    # Evaluation and logging
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true", help="Log training with Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="deepseek-coder-finetuning")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge_weights", action="store_true", help="Merge adapter weights into base model after training")
    parser.add_argument("--config_file", type=str, help="Path to YAML config file (overrides command line args)")

    return parser.parse_args()

def load_config_from_file(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def update_args_from_config(args, config):
    """Update arguments with values from config file"""
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    return args

def load_dataset_from_tokenized(dataset_dir, max_seq_length, train_size, seed):
    """Load dataset from tokenized files"""
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")

    tokenized_files = [f for f in os.listdir(dataset_dir) if f.endswith("_tokenized.json")]
    if not tokenized_files:
        raise ValueError(f"No tokenized files found in {dataset_dir}")

    logger.info(f"Found {len(tokenized_files)} tokenized project files")

    # Process and combine tokenized data
    all_examples = []

    for file_name in tokenized_files:
        file_path = os.path.join(dataset_dir, file_name)
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)

            project_name = project_data.get("project_name", os.path.splitext(file_name)[0].replace("_tokenized", ""))

            # Extract file tokens as training examples
            for file_path, file_data in project_data.get("files", {}).items():
                if not file_data.get("success", False) or "tokens" not in file_data:
                    continue

                tokens = file_data["tokens"]
                input_ids = tokens.get("input_ids", [[]])[0]
                attention_mask = tokens.get("attention_mask", [[]])[0]

                # Skip very short files
                if len(input_ids) < 10:
                    continue

                # Create fixed-length chunks for training
                for i in range(0, len(input_ids), max_seq_length // 2):
                    chunk_ids = input_ids[i:i + max_seq_length]
                    chunk_mask = attention_mask[i:i + max_seq_length]

                    # Skip very small chunks at the end
                    if len(chunk_ids) < max_seq_length // 4:
                        continue

                    # Pad or truncate to max_seq_length
                    if len(chunk_ids) < max_seq_length:
                        # Pad with zeros
                        padding_length = max_seq_length - len(chunk_ids)
                        chunk_ids = chunk_ids + [0] * padding_length
                        chunk_mask = chunk_mask + [0] * padding_length
                    elif len(chunk_ids) > max_seq_length:
                        # Truncate
                        chunk_ids = chunk_ids[:max_seq_length]
                        chunk_mask = chunk_mask[:max_seq_length]

                    all_examples.append({
                        "input_ids": chunk_ids,
                        "attention_mask": chunk_mask,
                        "file_path": file_path,
                        "project": project_name,
                    })

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Created {len(all_examples)} training examples")

    # Create dataset
    dataset = Dataset.from_list(all_examples)

    # Split dataset into train and eval
    dataset_dict = dataset.train_test_split(
        test_size=(1.0 - train_size),
        seed=seed
    )

    logger.info(f"Train dataset size: {len(dataset_dict['train'])}")
    logger.info(f"Eval dataset size: {len(dataset_dict['test'])}")

    return dataset_dict["train"], dataset_dict["test"]

def load_tokenizer_and_model(args):
    """Load tokenizer and model based on arguments"""
    # Quantization config
    if args.load_in_4bit:
        compute_dtype = getattr(torch, "bfloat16" if args.bf16 else "float16" if args.fp16 else "float32")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        use_flash_attention_2=args.use_flash_attention,
    )

    # Prepare model for k-bit training
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )

    # Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Log trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"Trainable params: {trainable_params} ({trainable_params / all_param * 100:.2f}% of all params)")

    return tokenizer, model

def train(args, tokenizer, model, train_dataset, eval_dataset):
    """Train the model with the given datasets"""
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"qlora-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="wandb" if args.use_wandb else "none",
        push_to_hub=False,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save the final model (ensuring it's in persistent storage)
    logger.info(f"Saving model to {args.output_dir}")
    try:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model and tokenizer successfully saved to {args.output_dir}")

        # Save a marker file to indicate successful save
        with open(os.path.join(args.output_dir, "save_complete.txt"), "w") as f:
            f.write(f"Save completed at {datetime.now().isoformat()}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

    # Optionally merge adapter weights
    if args.merge_weights:
        logger.info("Merging adapter weights into base model...")
        from peft import PeftModel

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
            trust_remote_code=True,
        )

        # Load adapter model
        adapter_model = PeftModel.from_pretrained(base_model, args.output_dir)

        # Merge weights
        merged_model = adapter_model.merge_and_unload()

        # Save merged model
        merged_output_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_output_dir, exist_ok=True)
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

        logger.info(f"Merged model saved to {merged_output_dir}")

    # Save config used for training
    try:
        config_path = os.path.join(args.output_dir, 'training_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f)
        logger.info(f"Training configuration saved to {config_path}")

        # Also save a timestamped copy for reference
        timestamp_config = os.path.join(args.output_dir, f'training_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
        shutil.copy(config_path, timestamp_config)
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")

    return trainer

def main():
    # Parse arguments
    args = parse_args()

    # If config file is provided, update args
    if args.config_file:
        config = load_config_from_file(args.config_file)
        args = update_args_from_config(args, config)

    # Set random seed
    torch.manual_seed(args.seed)

    # Check if running in Hugging Face Space and adapt paths accordingly
    if os.environ.get("SPACE_ID"):
        HF_HOME = os.environ.get("HF_HOME", "/home/user")
        SPACE_PERSISTENT_DIR = os.path.join(HF_HOME, "spaces", os.environ.get("SPACE_ID", "default"))

        # Use persistent storage for output directory
        if not args.output_dir.startswith(SPACE_PERSISTENT_DIR):
            persistent_output_dir = os.path.join(SPACE_PERSISTENT_DIR, "trained_model")
            logger.info(f"Running in Hugging Face Space, redirecting output to persistent storage: {persistent_output_dir}")
            args.output_dir = persistent_output_dir

        # Use persistent storage for dataset directory if it's the default
        if args.dataset_dir == "tokenized_data":
            persistent_data_dir = os.path.join(SPACE_PERSISTENT_DIR, "tokenized_data")
            logger.info(f"Redirecting dataset path to persistent storage: {persistent_data_dir}")
            args.dataset_dir = persistent_data_dir

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log the config
    logger.info(f"Training configuration: {vars(args)}")

    # Load dataset
    train_dataset, eval_dataset = load_dataset_from_tokenized(
        args.dataset_dir,
        args.max_seq_length,
        args.train_size,
        args.seed
    )

    # Load model and tokenizer
    tokenizer, model = load_tokenizer_and_model(args)

    # Train
    trainer = train(args, tokenizer, model, train_dataset, eval_dataset)

    logger.info("Training completed!")

    return 0

if __name__ == "__main__":
    main()
