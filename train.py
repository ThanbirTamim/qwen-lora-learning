import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)


# -------------------------
# Load dataset
# -------------------------
with open("data/train.json", "r") as f:
    data = json.load(f)


# -------------------------
# Proper Qwen Chat Format
# -------------------------
def format_example(item):
    return f"""<|im_start|>user
{item['instruction']}<|im_end|>
<|im_start|>assistant
{item['response']}<|im_end|>
"""


formatted_data = [{"text": format_example(item)} for item in data]

dataset = Dataset.from_list(formatted_data)


# -------------------------
# Tokenization
# -------------------------
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset.map(tokenize)


# -------------------------
# LoRA Config
# -------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)


model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# -------------------------
# Training setup
# -------------------------
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=20,
    per_device_train_batch_size=1,
    learning_rate=1e-4,
    logging_steps=1,
    save_steps=5,
    report_to="none"
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


print("Starting training...")
trainer.train()


# -------------------------
# Save adapter
# -------------------------
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")

print("Training complete. Adapter saved.")