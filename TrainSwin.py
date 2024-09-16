import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import Dataset, load_from_disk
from evaluate import load as load_metric
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True #imagenes ligeramente corruptas OK


model_name="microsoft/swinv2-base-patch4-window16-256"
dataset=load_from_disk("popocatepetl-dataset/")
print(dataset["train"].features["labels"].int2str(dataset["train"][0]["labels"]))

image_processor = AutoImageProcessor.from_pretrained(model_name)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


dataset["train"].set_transform(preprocess_train)
dataset["test"].set_transform(preprocess_val)

label2id={v:i for i,v in enumerate(dataset["train"].features["labels"].names)}
id2label={v:k for k,v in label2id.items()}
print(label2id,id2label)


model = AutoModelForImageClassification.from_pretrained(
    model_name,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True
).cuda()

#metric = load_metric("accuracy")
#def compute_metrics(eval_pred):
#    predictions = np.argmax(eval_pred.predictions, axis=1)
#    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    label_ids = eval_pred.label_ids
    accuracy = accuracy_metric.compute(predictions=predictions, references=label_ids)
    f1 = f1_metric.compute(predictions=predictions, references=label_ids, average='weighted')
    return {
        "accuracy": accuracy["accuracy"],
        "f1_weighted": f1["f1"]
    }
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
    
args = TrainingArguments(
    f"MimModels/swinv2-base-patch4-window16-256-popocatepetl-reclassified",
    remove_unused_columns=False,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    fp16=True,
    per_device_train_batch_size=12,
    #gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=25,
    warmup_ratio=0.1,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to=[]
)

trainer = Trainer(
    model,
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)
print(trainer.model.device)
#print(trainer.evaluate())
print(trainer.train())
trainer.save_state()




