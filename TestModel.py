
import argparse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from evaluate import load as load_metric
from datasets import Dataset, load_from_disk
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True #imagenes ligeramente corruptas OK

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process model and dataset locations.")
    parser.add_argument('-m', '--model_location', type=str, required=False, default="MimModels/swinv2-base-patch4-window16-256-popocatepetl-reclassified/", help='Path to the model location')
    parser.add_argument('-d', '--dataset_location', type=str, required=False, default="popocatepetl-dataset/", help='Path to the dataset location')
    parser.add_argument('-c', '--cuda', type=bool, required=False, default=False, help='Whether to use GPU for evaluation')
    return parser.parse_args()
dataset_location=None
model_location=None
cuda=False

import torch
import torch.nn as nn
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def test_model():
    dataset=load_from_disk(dataset_location)
    model = AutoModelForImageClassification.from_pretrained(model_location)
    processor = AutoImageProcessor.from_pretrained(model_location)
    if(cuda):
        model=model.cuda()
    
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    if "height" in processor.size:
        size = (processor.size["height"], processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in processor.size:
        size = processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = processor.size.get("longest_edge")
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )
    #def preprocess_val(example_batch):
    #    if("image" in example_batch.keys()):
    #        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    #    return example_batch
    #dataset["test"].set_transform(preprocess_val)
    y=dataset["test"]["labels"]
    l2id={l:i for i,l in enumerate(dataset["test"].features["labels"].names)}
    yp=[]
    for i,row in enumerate(dataset["test"]):
        print(f"{i/len(dataset['test'])*100:.02f}%",end="\r")
        try:
            inputs=processor(images=row["image"], return_tensors="pt")["pixel_values"].to(model.device)
            logits=model(pixel_values=inputs).logits
            probabilities = nn.functional.softmax(logits, dim=-1)
            probabilities = probabilities.detach().cpu().numpy().flatten()
            predicted_class_idx = logits.argmax(-1).item()
            yp.append(l2id[model.config.id2label[predicted_class_idx]])
        except:
            yp.append(l2id["UNK"])
    accuracy = accuracy_metric.compute(predictions=yp, references=y)
    f1 = f1_metric.compute(predictions=yp, references=y, average='weighted')
    print(accuracy,f1)

if __name__ == "__main__":
    args = parse_arguments()
    model_location=args.model_location
    dataset_location=args.dataset_location
    cuda=args.cuda
    
    print(f"Model Location: {model_location}")
    print(f"Dataset Location: {dataset_location}")
    print(f"Using CUDA?: {cuda}")
    test_model()