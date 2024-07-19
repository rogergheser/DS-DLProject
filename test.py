import torch




ice_probabilities, ice_predictions = ice_scores.topk(5)
cap_probabilities = torch.zeros(ice_predictions.shape)
for i in range(ice_predictions.shape[0]):
    cap_probabilities[i,:] = caption_logits[i,ice_predictions[0]]