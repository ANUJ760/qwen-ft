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

DATASET_PATH = (
    ROOT / "dataset" / "training_pairs.jsonl"
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_dataset(dataset):
    required_columns = {"messages"}
    missing = required_columns - set(dataset.column_names)

    if missing:
        raise ValueError(
            f"Dataset is missing required fields: {missing}"
        )

    for i, example in enumerate(dataset):
        messages = example["messages"]

        if not isinstance(messages, list):
            raise ValueError(
                f"Example {i}: messages must be a list"
            )

        roles = [message["role"] for message in messages]

        if roles != ["system", "user", "assistant"]:
            raise ValueError(
                f"Example {i}: expected roles "
                f"['system', 'user', 'assistant'], got {roles}"
            )


def main():
    config = load_config()
    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]
    output_dir = (
        ROOT / config["output"]["directory"]
    ).resolve()

    dataset = load_dataset(
        "json",
        data_files=str(DATASET_PATH),
        split="train",
    )

    print(f"Loaded {len(dataset)} total examples")

    validate_dataset(dataset)
    print("Dataset structure validated")

    split = dataset.train_test_split(
        test_size=config["dataset"]["validation_split"],
        seed=config["dataset"]["seed"],
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
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
        model_name,
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
        output_dir=str(output_dir),
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
        warmup_steps=config["training"]["warmup_steps"],
        optim="paged_adamw_8bit",
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=(
            config["training"]["gradient_checkpointing"]
        ),
        max_length=max_seq_length,
        assistant_only_loss=True,
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

    print()
    print("=" * 60)
    print("Starting QLoRA fine-tuning")
    print("=" * 60)
    print()

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print()
    print("=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Adapter: {output_dir}")


if __name__ == "__main__":
    main()
