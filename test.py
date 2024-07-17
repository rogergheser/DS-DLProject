import torch
import torch.nn.functional as F

image_features = torch.randn(16,512)
caption_features = torch.randn(16,512)
text_features = torch.randn(200,512)

image_features = image_features.cuda()
caption_features = caption_features.cuda()
text_features = text_features.cuda()

ice_k=5

##### ICE method #####
# Obtain image and caption logits wrt. class embeddings
i_logits = F.normalize(image_features) @ F.normalize(text_features).T
c_logits = F.normalize(caption_features) @ F.normalize(text_features).T

# Softmax the image and caption logits
i_logits = F.softmax(i_logits, dim=1)
c_logits = F.softmax(c_logits, dim=1)


# Precompute the top-K values and class indices 
# for image and caption predictions
top_k_preds = i_logits.topk(ice_k, dim=1).indices
top_k_vals = i_logits.topk(ice_k, dim=1).values
top_k_c_vals = c_logits.topk(ice_k, dim=1).values

# For each prediction in the top-K image predictions, 
# obtain the corresponding image and caption logits
image_top_scores = torch.zeros(i_logits.shape[0], ice_k)
caption_top_scores = torch.zeros(i_logits.shape[0], ice_k)
for j in range(ice_k):
    top_j_preds = top_k_preds[:, j]
    j_scores = i_logits[torch.arange(i_logits.shape[0]), top_j_preds]
    image_top_scores[:, j] = j_scores

    j_scores = c_logits[torch.arange(c_logits.shape[0]), top_j_preds]
    caption_top_scores[:, j] = j_scores

# Compute the ICE coefficient as \lambda * normalize(std(top-K image scores))
# We find this to perform the best empirically
coef = 0.08 * F.normalize(
    torch.stack(
        (top_k_vals.std(1), top_k_c_vals.std(1))
        , dim=1
    )
    , dim=1
).cuda()
coef = coef[:, 1][:, None].repeat(1, caption_top_scores.shape[1])

# Sum the image and caption scores to obtain the ICE scores
ice_scores = image_top_scores.cuda() + coef * caption_top_scores.cuda()

# Lambda computed as a normalization term
std_devs = torch.stack((top_k_vals.std(dim=1), top_k_c_vals.std(dim=1)), dim=1)
coef = 0.08 * F.normalize(std_devs, dim=1)
coef = coef[:, 1].unsqueeze(1).expand(-1, caption_top_scores.size(1))

# Sum the image and caption scores to obtain the ICE scores
ice_scores_1 = image_top_scores.cuda() + coef * caption_top_scores.cuda()
ice_scores_1