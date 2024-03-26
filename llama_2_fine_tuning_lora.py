# https://www.datacamp.com/tutorial/fine-tuning-llama-2

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer


# Model from Hugging Face hub
base_model = "meta-llama/Llama-2-7b-chat-hf"

# New instruction dataset
# https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k
guanaco_dataset = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(guanaco_dataset, split="train")
# Fine-tuned model
new_model = "llama-2-7b-chat-fine-tuned"


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# LoRA Config
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Params
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)
# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Training
trainer.train()

# Save Model
trainer.model.save_pretrained(new_model)

# Generate Text
query = "How do I use the OpenAI API?"
text_gen = pipeline(task="text-generation", model=new_model, tokenizer=tokenizer, max_length=200)
output = text_gen(f"<s>[INST] {query} [/INST]")
print(output[0]['generated_text'])