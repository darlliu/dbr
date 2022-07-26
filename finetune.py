from data import TRAIN, TEST, get_batch, get_data, get_char_names
import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

batch_size = 32
max_source_length = 128
max_target_length = 32
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")
# model.resize_token_embeddings(len(tokenizer))

tokenizer.add_tokens(get_char_names())
model.resize_token_embeddings(len(tokenizer))


class FTDataSet(torch.utils.data.Dataset):
    def __init__(self, ds, task_prefix):
        self.ds = ds
        self.task_prefix = task_prefix
        return

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Out of bound")
        src, tgt, tgt2, _ = get_data(self.ds[idx])
        src_enc = tokenizer(self.task_prefix + src, padding="max_length",
                            max_length=max_source_length, truncation=True, return_tensors='pt')
        input_ids = src_enc.input_ids
        tgt_enc = tokenizer(tgt, padding="max_length",
                            max_length=max_target_length, truncation=True, return_tensors='pt')
        labels = tgt_enc.input_ids
        labels = torch.tensor(labels)
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": torch.squeeze(input_ids), "labels": torch.squeeze(labels)}

    def __len__(self):
        return len(self.ds)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    logits = predictions[0]
    idxs = np.argmax(logits, 2)
    decoded_preds = tokenizer.batch_decode(
        idxs, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    empty = 0.
    success = 0.
    passes = 0.
    failure = 0.
    for pred, label in zip(decoded_preds, decoded_labels):
        if label.strip() != "" and pred.strip() == "":
            empty += 1
        elif label.strip() == "" and pred.strip() == "":
            passes += 1
        elif label.strip == pred.strip():
            success += 1
        else:
            failure += 1
    sz = len(labels)
    return {"Empty": empty/sz, "Success": success/sz, "Passes": passes/sz, "Failure": failure/sz}


def tags_to_char_finetune():
    """
    Task 1: fine tune seq2seq task of predicting char name
    """
    task_prefix = "predict char from tags: "
    DSTrain = FTDataSet(TRAIN, task_prefix)
    DSTest = FTDataSet(TEST, task_prefix)
    args = TrainingArguments(
        "t5-small-finetuned-dbr",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=10,
        log_level="debug",
        logging_steps=20
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=DSTrain,
        eval_dataset=DSTest,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )
    trainer.train()
    return


if __name__ == "__main__":
    task_prefix = "predict char from tags: "
    tags_to_char_finetune()
    # for idx in TRAIN[:10]:
    #     src, tgt, tgt2, _ = get_data(idx)
    #     src_enc = tokenizer(task_prefix + src, padding="max_length",
    #                         max_length=max_source_length, truncation=True, return_tensors='pt')
    #     input_ids = src_enc.input_ids
    #     tgt_enc = tokenizer(tgt, padding="max_length",
    #                         max_length=max_target_length, truncation=True, return_tensors='pt')
    #     labels = tgt_enc.input_ids
    #     labels = torch.tensor(labels)
    #     labels[labels == tokenizer.pad_token_id] = -100
    #     res = model(input_ids=input_ids, labels=labels).loss
    #     print(res)

    for t in TEST:
        tokenizer = T5Tokenizer.from_pretrained(
            "t5-small-finetuned-dbr/checkpoint-2000")
        model = T5ForConditionalGeneration.from_pretrained(
            "t5-small-finetuned-dbr/checkpoint-2000")
        task_prefix = "predict char from tags: "
        src, tgt, tgt2, _ = get_data(t)
        src_enc = tokenizer(task_prefix + src, padding="max_length",
                            max_length=max_source_length, truncation=True, return_tensors='pt')
        input_ids, input_masks = src_enc.input_ids, src_enc.attention_mask
        outputs = model.generate(input_ids)
        print(f"tags = {src} , label = {tgt}")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
