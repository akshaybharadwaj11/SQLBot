import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from model_utils import load_base_model
from data import preprocess_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--use_8bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_base_model(args.base_model, use_8bit=args.use_8bit)
    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, lora_config)

    raw_datasets = load_dataset("json", data_files={
        "train": args.train_file, "validation": args.valid_file
    })
    tokenized_datasets = preprocess_dataset(raw_datasets, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        save_strategy="epoch",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()