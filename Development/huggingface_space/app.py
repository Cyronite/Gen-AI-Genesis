import os
import gradio as gr
import json
import torch
import shutil
import subprocess
import sys
from huggingface_hub import notebook_login, HfFolder
from pathlib import Path
from transformers import AutoTokenizer
import yaml
from datetime import datetime

# Configure default paths for Hugging Face Spaces - using absolute paths for persistence
import os
HF_HOME = os.environ.get("HF_HOME", "/home/user")
SPACE_PERSISTENT_DIR = os.path.join(HF_HOME, "spaces", os.environ.get("SPACE_ID", "default"))

# Ensure all data is stored in the persistent storage area
TOKENIZED_DATA_DIR = os.path.join(SPACE_PERSISTENT_DIR, "tokenized_data")
OUTPUT_DIR = os.path.join(SPACE_PERSISTENT_DIR, "trained_model")
LOGS_DIR = os.path.join(SPACE_PERSISTENT_DIR, "logs")
CONFIG_DIR = os.path.join(SPACE_PERSISTENT_DIR, "config")

# Ensure directories exist in persistent storage
os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Log the persistent storage locations
print(f"Using persistent storage at: {SPACE_PERSISTENT_DIR}")
print(f"Model will be saved to: {OUTPUT_DIR}")
print(f"Tokenized data directory: {TOKENIZED_DATA_DIR}")

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check available GPU
def get_gpu_info():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        gpu_memory = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count)]
        return {
            "available": True,
            "count": gpu_count,
            "names": gpu_names,
            "memory_gb": gpu_memory
        }
    else:
        return {"available": False}

def upload_file(files):
    """Process uploaded tokenized files"""
    if not files:
        return "No files uploaded."
    
    # Ensure the directory exists
    os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)
    
    # Copy uploaded files to persistent storage
    uploaded_count = 0
    for file_path in files:
        try:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(TOKENIZED_DATA_DIR, file_name)
            shutil.copy(file_path, dest_path)
            uploaded_count += 1
            print(f"Uploaded {file_name} to {dest_path}")
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")
    
    return f"Uploaded {uploaded_count} files to persistent storage at {TOKENIZED_DATA_DIR}."

def save_config(config_dict):
    """Save configuration to YAML file"""
    config_path = os.path.join(CONFIG_DIR, f"train_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
    
    return f"Configuration saved to {config_path}"

def start_training_process(config_dict):
    """Start training in a subprocess"""
    # Ensure output directory is set to persistent storage
    if config_dict.get("output_dir") != OUTPUT_DIR:
        config_dict["output_dir"] = OUTPUT_DIR
        
    # Save config to a file in persistent storage
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = os.path.join(CONFIG_DIR, f"current_training_{timestamp}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
    
    # Also create a symlink to the latest config for easier access
    latest_config = os.path.join(CONFIG_DIR, "latest_config.yaml")
    if os.path.exists(latest_config):
        os.remove(latest_config)
    shutil.copy(config_path, latest_config)
    
    # Prepare command - ensure absolute paths
    command = [
        sys.executable, 
        os.path.abspath("../train.py") if os.path.exists("../train.py") else os.path.abspath("train.py"), 
        "--config_file", 
        config_path
    ]
    
    # Launch process
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output in a non-blocking way
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                output_lines.append(line.strip())
                if len(output_lines) > 100:
                    output_lines = output_lines[-100:]
                
        return_code = process.poll()
        
        if return_code == 0:
            return "\n".join(output_lines) + "\n\nTraining completed successfully!"
        else:
            return "\n".join(output_lines) + f"\n\nTraining failed with return code {return_code}"
            
    except Exception as e:
        return f"Error starting training process: {str(e)}"

def validate_tokenized_files():
    """Validate tokenized files and return stats"""
    if not os.path.exists(TOKENIZED_DATA_DIR):
        os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)
        return f"Created tokenized data directory at {TOKENIZED_DATA_DIR}. Please upload files."
        
    tokenized_files = [f for f in os.listdir(TOKENIZED_DATA_DIR) if f.endswith("_tokenized.json")]
    if not tokenized_files:
        return f"No tokenized files found in {TOKENIZED_DATA_DIR}. Please upload files."
    
    total_files = 0
    total_projects = len(tokenized_files)
    total_tokens = 0
    
    for file_name in tokenized_files:
        try:
            with open(os.path.join(TOKENIZED_DATA_DIR, file_name), 'r') as f:
                data = json.load(f)
                
            project_files = data.get("files", {})
            total_files += len(project_files)
            
            # Count tokens
            for file_data in project_files.values():
                if file_data.get("success", False):
                    total_tokens += file_data.get("token_count", 0)
                    
        except Exception as e:
            return f"Error processing {file_name}: {str(e)}"
    
    return f"Found {total_projects} projects with {total_files} files totaling approximately {total_tokens:,} tokens."

def test_tokenizer():
    """Test if tokenizer loads correctly"""
    try:
        model_name = "deepseek-ai/deepseek-coder-v2-lite-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Test tokenization
        test_text = "def hello_world():\n    print('Hello, world!')"
        tokens = tokenizer(test_text, return_tensors="pt")
        
        return f"Tokenizer loaded successfully. Sample tokenization: {tokens.input_ids.shape}"
        
    except Exception as e:
        return f"Error loading tokenizer: {str(e)}"

def login_huggingface(token):
    """Login to Hugging Face hub"""
    try:
        # Save token
        with open(os.path.expanduser("~/.huggingface/token"), "w") as f:
            f.write(token)
            
        # Verify login
        if HfFolder().get_token() == token:
            return "Successfully logged in to Hugging Face!"
        else:
            return "Failed to save token. Please try again."
            
    except Exception as e:
        return f"Error during login: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="DeepSeek Coder Fine-tuning") as interface:
        gr.Markdown("# DeepSeek Coder V2 Fine-tuning with QLoRA")
        gr.Markdown(f"""
        This Hugging Face Space allows you to fine-tune the DeepSeek Coder V2 Lite Instruct model on your tokenized code files.
        
        ## Storage Information:
        - Models will be saved to: `{OUTPUT_DIR}` (persistent)
        - Tokenized data stored in: `{TOKENIZED_DATA_DIR}` (persistent)
        - Training logs saved to: `{LOGS_DIR}` (persistent)
        
        ## Instructions:
        1. Login to Hugging Face (optional, for pushing models)
        2. Upload your tokenized project files (JSON files)
        3. Configure the training parameters
        4. Start the training process
        """)
        
        # System status
        with gr.Accordion("System Status", open=True):
            gpu_info = get_gpu_info()
            if gpu_info["available"]:
                gpu_text = f"🟢 GPU Available: {gpu_info['count']} x {gpu_info['names'][0]} ({gpu_info['memory_gb'][0]:.1f} GB)"
            else:
                gpu_text = "🔴 No GPU detected. Training will be extremely slow!"
                
            gr.Markdown(f"**{gpu_text}**")
            
            with gr.Row():
                test_tokenizer_btn = gr.Button("Test Tokenizer")
                validate_files_btn = gr.Button("Validate Tokenized Files")
                
            system_output = gr.Textbox(label="System Check Result", lines=3)
            
            test_tokenizer_btn.click(
                fn=test_tokenizer,
                inputs=[],
                outputs=system_output
            )
            
            validate_files_btn.click(
                fn=validate_tokenized_files,
                inputs=[],
                outputs=system_output
            )
        
        # HF Login
        with gr.Accordion("Hugging Face Login (Optional)", open=False):
            gr.Markdown("Login to Hugging Face to push models to the Hub")
            hf_token = gr.Textbox(label="Hugging Face Token", type="password")
            login_button = gr.Button("Login")
            login_output = gr.Textbox(label="Login Result")
            
            login_button.click(
                fn=login_huggingface,
                inputs=hf_token,
                outputs=login_output
            )
        
        with gr.Tabs():
            with gr.TabItem("Upload Data"):
                with gr.Row():
                    upload_files = gr.Files(label="Upload Tokenized Files (JSON)")
                    upload_button = gr.Button("Upload Files")
                
                upload_output = gr.Textbox(label="Upload Result")
                
                upload_button.click(
                    fn=upload_file,
                    inputs=upload_files,
                    outputs=upload_output
                )
            
            with gr.TabItem("Training Configuration"):
                with gr.Row():
                    with gr.Column():
                        model_name = gr.Textbox(
                            label="Model Name", 
                            value="deepseek-ai/deepseek-coder-v2-lite-instruct"
                        )
                        output_dir = gr.Textbox(
                            label="Output Directory (Persistent Storage)", 
                            value=OUTPUT_DIR,
                            info="This is set to a persistent storage location that will survive Space restarts"
                        )
                        dataset_dir = gr.Textbox(
                            label="Dataset Directory (Persistent Storage)",
                            value=TOKENIZED_DATA_DIR,
                            info="This is set to a persistent storage location that will survive Space restarts"
                        )
                        
                        with gr.Row():
                            train_size = gr.Slider(
                                label="Train/Eval Split",
                                minimum=0.5,
                                maximum=1.0,
                                value=0.95,
                                step=0.01
                            )
                            seed = gr.Number(
                                label="Random Seed",
                                value=42,
                                precision=0
                            )
                        
                        max_seq_length = gr.Slider(
                            label="Maximum Sequence Length",
                            minimum=1024,
                            maximum=16384,
                            value=8192,
                            step=1024
                        )
                
                    with gr.Column():
                        with gr.Row():
                            train_batch_size = gr.Number(
                                label="Train Batch Size",
                                value=2,
                                precision=0
                            )
                            eval_batch_size = gr.Number(
                                label="Eval Batch Size",
                                value=2,
                                precision=0
                            )
                            grad_accum_steps = gr.Number(
                                label="Gradient Accumulation Steps",
                                value=4,
                                precision=0
                            )
                        
                        with gr.Row():
                            num_epochs = gr.Number(
                                label="Number of Epochs",
                                value=3,
                                precision=0
                            )
                            learning_rate = gr.Number(
                                label="Learning Rate",
                                value=2e-4,
                                precision=6
                            )
                            weight_decay = gr.Number(
                                label="Weight Decay",
                                value=0.001,
                                precision=5
                            )
                        
                        with gr.Row():
                            eval_steps = gr.Number(
                                label="Evaluation Steps",
                                value=200,
                                precision=0
                            )
                            save_steps = gr.Number(
                                label="Save Steps",
                                value=200,
                                precision=0
                            )
                            logging_steps = gr.Number(
                                label="Logging Steps",
                                value=50,
                                precision=0
                            )
                
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                lora_r = gr.Number(
                                    label="LoRA Rank (r)",
                                    value=64,
                                    precision=0
                                )
                                lora_alpha = gr.Number(
                                    label="LoRA Alpha",
                                    value=16,
                                    precision=0
                                )
                                lora_dropout = gr.Number(
                                    label="LoRA Dropout",
                                    value=0.05,
                                    precision=4
                                )
                            
                            with gr.Row():
                                use_flash_attn = gr.Checkbox(
                                    label="Use Flash Attention",
                                    value=True
                                )
                                load_in_4bit = gr.Checkbox(
                                    label="Use 4-bit Quantization",
                                    value=True
                                )
                                merge_modules = gr.Checkbox(
                                    label="Merge Adapter Weights After Training",
                                    value=False
                                )
                    
                        with gr.Column():
                            with gr.Row():
                                warmup_ratio = gr.Number(
                                    label="Warmup Ratio",
                                    value=0.03,
                                    precision=4
                                )
                                max_grad_norm = gr.Number(
                                    label="Max Gradient Norm",
                                    value=0.3,
                                    precision=2
                                )
                            
                            with gr.Row():
                                lr_scheduler = gr.Dropdown(
                                    label="Learning Rate Scheduler",
                                    choices=["linear", "cosine", "constant", "constant_with_warmup"],
                                    value="cosine"
                                )
                                precision = gr.Radio(
                                    label="Training Precision",
                                    choices=["bf16", "fp16", "fp32"],
                                    value="bf16"
                                )
                                use_wandb = gr.Checkbox(
                                    label="Use Weights & Biases",
                                    value=False
                                )
                
                save_button = gr.Button("Save Configuration")
                save_output = gr.Textbox(label="Save Result")
                
                save_button.click(
                    fn=lambda *args: save_config({
                        "model_name": args[0],
                        "output_dir": args[1],
                        "dataset_dir": args[2],
                        "train_size": args[3],
                        "seed": int(args[4]),
                        "max_seq_length": int(args[5]),
                        "per_device_train_batch_size": int(args[6]),
                        "per_device_eval_batch_size": int(args[7]),
                        "gradient_accumulation_steps": int(args[8]),
                        "num_train_epochs": int(args[9]),
                        "learning_rate": float(args[10]),
                        "weight_decay": float(args[11]),
                        "eval_steps": int(args[12]),
                        "save_steps": int(args[13]),
                        "logging_steps": int(args[14]),
                        "lora_r": int(args[15]),
                        "lora_alpha": int(args[16]),
                        "lora_dropout": float(args[17]),
                        "use_flash_attention": args[18],
                        "load_in_4bit": args[19],
                        "merge_weights": args[20],
                        "warmup_ratio": float(args[21]),
                        "max_grad_norm": float(args[22]),
                        "lr_scheduler_type": args[23],
                        "bf16": args[24] == "bf16",
                        "fp16": args[24] == "fp16",
                        "use_wandb": args[25]
                    }),
                    inputs=[
                        model_name, output_dir, dataset_dir, train_size, seed, max_seq_length,
                        train_batch_size, eval_batch_size, grad_accum_steps,
                        num_epochs, learning_rate, weight_decay,
                        eval_steps, save_steps, logging_steps,
                        lora_r, lora_alpha, lora_dropout,
                        use_flash_attn, load_in_4bit, merge_modules,
                        warmup_ratio, max_grad_norm, lr_scheduler,
                        precision, use_wandb
                    ],
                    outputs=save_output
                )
            
            with gr.TabItem("Start Training"):
                gr.Markdown("""
                Make sure you've uploaded your tokenized files and configured the training parameters before starting the training process.
                
                **Note**: Training will use the configuration you've set in the 'Training Configuration' tab.
                """)
                
                start_button = gr.Button("Start Training", variant="primary")
                training_output = gr.Textbox(label="Training Status", max_lines=25, autoscroll=True)
                
                start_button.click(
                    fn=lambda *args: start_training_process({
                        "model_name": args[0],
                        "output_dir": args[1],
                        "dataset_dir": args[2],
                        "train_size": args[3],
                        "seed": int(args[4]),
                        "max_seq_length": int(args[5]),
                        "per_device_train_batch_size": int(args[6]),
                        "per_device_eval_batch_size": int(args[7]),
                        "gradient_accumulation_steps": int(args[8]),
                        "num_train_epochs": int(args[9]),
                        "learning_rate": float(args[10]),
                        "weight_decay": float(args[11]),
                        "eval_steps": int(args[12]),
                        "save_steps": int(args[13]),
                        "logging_steps": int(args[14]),
                        "lora_r": int(args[15]),
                        "lora_alpha": int(args[16]),
                        "lora_dropout": float(args[17]),
                        "use_flash_attention": args[18],
                        "load_in_4bit": args[19],
                        "merge_weights": args[20],
                        "warmup_ratio": float(args[21]),
                        "max_grad_norm": float(args[22]),
                        "lr_scheduler_type": args[23],
                        "bf16": args[24] == "bf16",
                        "fp16": args[24] == "fp16",
                        "use_wandb": args[25]
                    }),
                    inputs=[
                        model_name, output_dir, dataset_dir, train_size, seed, max_seq_length,
                        train_batch_size, eval_batch_size, grad_accum_steps,
                        num_epochs, learning_rate, weight_decay,
                        eval_steps, save_steps, logging_steps,
                        lora_r, lora_alpha, lora_dropout,
                        use_flash_attn, load_in_4bit, merge_modules,
                        warmup_ratio, max_grad_norm, lr_scheduler,
                        precision, use_wandb
                    ],
                    outputs=training_output
                )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    # Copy main training script to accessible location
    if not os.path.exists("train.py"):
        parent_dir = str(Path(__file__).parent.parent)
        source_path = os.path.join(parent_dir, "train.py")
        if os.path.exists(source_path):
            shutil.copy(source_path, "train.py")
            logger.info(f"Copied train.py from {source_path} to current directory")
        else:
            logger.warning(f"Could not find train.py at {source_path}")
    
    # Print persistent storage info
    logger.info(f"============ PERSISTENT STORAGE INFORMATION ============")
    logger.info(f"Persistent root: {SPACE_PERSISTENT_DIR}")
    logger.info(f"Model output directory: {OUTPUT_DIR}")
    logger.info(f"Tokenized data directory: {TOKENIZED_DATA_DIR}")
    logger.info(f"Log directory: {LOGS_DIR}")
    logger.info(f"Config directory: {CONFIG_DIR}")
    logger.info(f"======================================================")
    
    # Create a symlink to trained model in the current directory for easier access
    if not os.path.exists("trained_model") and os.path.exists(OUTPUT_DIR):
        try:
            os.symlink(OUTPUT_DIR, "trained_model")
            logger.info(f"Created symlink to trained model at ./trained_model")
        except Exception as e:
            logger.warning(f"Could not create symlink to trained model: {e}")
    
    demo = create_interface()
    demo.queue()
    demo.launch()