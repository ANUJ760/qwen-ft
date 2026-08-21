from pathlib import Path
import yaml
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig

from trl import SFTConfig, SFTTrainer



ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = (
    ROOT / "finetune" / "configs" / "qlora_config.yaml"
)


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


MODEL_NAME = config["model"]["name"]
MAX_SEQ_LENGTH = config["model"]["max_seq_length"]

DATASET_PATH = ROOT / "dataset" / "training_pairs.jsonl"

OUTPUT_DIR = (
    ROOT
    / "finetune"
    / config["output"]["directory"]
).resolve()




dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH),
    split="train",
)

print(f"Loaded {len(dataset)} examples")




split = dataset.train_test_split(
    test_size=config["dataset"]["validation_split"],
    seed=config["dataset"]["seed"],
)

train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")



tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token




bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False




lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],

    target_modules=config["lora"]["target_modules"],

    bias="none",
    task_type="CAUSAL_LM",
)



training_args = SFTConfig(

    output_dir=str(OUTPUT_DIR),

    num_train_epochs=config["training"]["epochs"],

    per_device_train_batch_size=(
        config["training"]["train_batch_size"]
    ),

    per_device_eval_batch_size=(
        config["training"]["eval_batch_size"]
    ),

    gradient_accumulation_steps=(
        config["training"]["gradient_accumulation_steps"]
    ),

    learning_rate=config["training"]["learning_rate"],

    warmup_ratio=config["training"]["warmup_ratio"],

    optim="paged_adamw_8bit",

    fp16=config["training"]["fp16"],
    bf16=False,

    gradient_checkpointing=(
        config["training"]["gradient_checkpointing"]
    ),

    max_length=MAX_SEQ_LENGTH,

    logging_steps=config["training"]["logging_steps"],

    eval_strategy="steps",
    eval_steps=config["training"]["eval_steps"],

    save_strategy="steps",
    save_steps=config["training"]["save_steps"],
    save_total_limit=config["training"]["save_total_limit"],

    load_best_model_at_end=True,

    seed=config["dataset"]["seed"],

    report_to="none",
)




trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    peft_config=lora_config,
)



print("\nStarting QLoRA training...\n")

trainer.train()




trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\nTraining complete.")
print(f"Adapter saved to: {OUTPUT_DIR}")