def preprocess_dataset(datasets, tokenizer, max_length=512):
    def tokenize_fn(examples):
        inputs = [q for q in examples["nl"]]
        targets = [s for s in examples["sql"]]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
        labels = tokenizer(targets, max_length=max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return datasets.map(tokenize_fn, batched=True)