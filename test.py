from CLIP import clip
import torch
import torch.nn.functional as F


phrases = ["A photo of a cat", "A picture of dog next to a cat in the woods"]

net, _ = clip.load("RN50", device="cpu")

caption_tokens = clip.tokenize(phrases)
caption_features = net.encode_text(caption_tokens) 

print(F.normalize(caption_features) @ F.normalize(caption_features).T)