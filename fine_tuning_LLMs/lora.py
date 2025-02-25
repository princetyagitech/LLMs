# Library Imports
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Constants (Replace with your dataset and model)
DATASET_NAME = "ceadar-ie/FinTalk-19k"  # Example: "ceadar-ie/FinTalk-19k"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"  # Example: "meta-llama/Llama-2-13b-chat-hf"
OUTPUT_DIR = "./output_directory"
OUTPUT_MODEL_DIR = "saved_model_directory"
BATCH_SIZE = 10
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 15
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 4096

# Function Definitions

def convert_to_custom_format(data):
    """
    Convert dataset entries to a custom text format.

    Args:
        data: Dataset entries to be converted.

    Returns:
        custom_data: List of entries in the custom format.
    """
    custom_data = []
    for entry in data:
        inst_text = f"<s>[INST] {entry['instruction']} [/INST]"
        context_text = f" {entry['context']}"
        response_text = f"{entry['response']} </s>"
        full_text = f"{inst_text} {context_text} {response_text}"
        custom_data.append({'text': full_text})
    return custom_data

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: The model whose parameters are to be printed.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params:.2f}")

# Load and Process Dataset

# Load the dataset
dataset = load_dataset(DATASET_NAME, split="train")

# Apply the conversion function to the dataset
custom_format_data = convert_to_custom_format(dataset)
custom_dataset = Dataset.from_dict({"text": [entry['text'] for entry in custom_format_data]})

# Print the custom dataset
print(custom_dataset)

# Model and Tokenizer Configuration

# Configuration for model quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the pre-trained model with quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# Print the number of trainable parameters
print_trainable_parameters(model)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False  # Freeze the model - train adapters later
    if param.ndim == 1:
        # Cast small parameters (e.g., layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# LoRA Configuration

# LoRA (Low-Rank Adaptation) configuration
peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.05,
    r=128,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Configuration

# Training arguments configuration
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=10,
    learning_rate=LEARNING_RATE,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# Training Initialization and Execution

# Trainer initialization
trainer = SFTTrainer(
    model=model,
    train_dataset=custom_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Start the training process
trainer.train()

# Save the trained model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained(OUTPUT_MODEL_DIR)
