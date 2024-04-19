import os
from CLIP.clip import clip
import torch
import torchvision
import cv2
from torchvision.datasets import CIFAR100

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model, preprocess = clip.load('ViT-B/32', device)
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

image, label = cifar100[3237]
image = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

values, indices = similarity[0].topk(5)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
print("\nTrue label:{}".format(cifar100.classes[label]))

cv2.imshow(image)
cv2.waitKey(0)