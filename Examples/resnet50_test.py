from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

dataset = load_dataset("microsoft/cats_vs_dogs")
image = dataset["train"]["image"][0]
print(dataset)
plt.imshow(image)
plt.axis('off')
plt.show()

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
