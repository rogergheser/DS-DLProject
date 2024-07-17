import io
import torch
import os
import numpy as np
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from CLIP import clip

from COOP.models import OurCLIP
from COOP.utils import get_optimizer, log_values, get_loss_function
from COOP.functions import training_step, test_step
from COOP.dataloader import get_data
from coca_model import Captioner
from loaders import Augmixer, load_pretrained_coop
from tqdm import tqdm
from utils import (entropy, avg_entropy, batch_report, filter_on_entropy,
                report_predictions, make_histogram, compute_accuracies, caption_report)
from copy import deepcopy
import torch.nn.functional as F

DEBUG = True
RUN_NAME = "exp6"

def add_caption_loss(net: OurCLIP, captioner: Captioner, filtered_inputs, filtered_outputs, text_features, label, id2class, prompt="a ", _lambda=0.5, K=5, debug=False):
    """
    Adds caption loss to the filtered_outputs using the given captioner.

    Args:
        captioner (C): The captioner object used to generate captions.
        filtered_inputs: The filtered inputs.
        filtered_outputs: The filtered outputs.
        prompt (str): The prompt used for generating captions. Default is "a ".
        _lambda (float): The value of lambda used for computing the caption similarity. Default is 0.5.

    Returns:
        The updated filtered_outputs with caption loss added.
    """
    
    # TODO implement this function following the steps
    # Compute captions for each augmentation using coca functions
    device = filtered_inputs.device
    with torch.no_grad(), torch.cuda.amp.autocast():
        captions = captioner.generate_captions(filtered_inputs, prompt)
    
    # Encode all the captions using the clip encoder (batchfying the captions to save compute)
    caption_tokens = clip.tokenize(captions).to(device)
    caption_features = net.encode_text(caption_tokens).to(device) 
    caption_logits = (F.normalize(caption_features) @ text_features.T).softmax(-1)
    image_logits = filtered_outputs

    # Extract topk classes for each image/prompt and for each caption/prompt
    topk_image_values, topk_image_pred = image_logits.topk(K)
    topk_caption_values, topk_caption_pred = caption_logits.topk(K)

    # Compute the value of lambda following ice implementation row 193 main_ice.pyù
    if _lambda:
        ice_scores = (1-_lambda)*topk_image_values + _lambda*topk_caption_values
    else:
        # Lambda computed as a normalization term
        std_devs = torch.stack((topk_image_values.std(dim=1), topk_caption_values.std(dim=1)), dim=1)
        coef = 0.08 * F.normalize(std_devs, dim=1)
        coef = coef[:, 1].unsqueeze(1).expand(-1, topk_caption_values.size(1))

        # Sum the image and caption scores to obtain the ICE scores
        ice_scores = topk_image_values + coef * topk_caption_values

    if debug:
        caption_report(filtered_inputs, label, captions, id2classes, idx)
    

    return ice_scores
    

def tta_net_train(batch, net, optimizer, scaler, id2classes, device="cuda", captioner=None, debug=False):
    batch_idx, inputs, targets = batch

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs, text_features = net(inputs)
    outputs = outputs.softmax(dim=-1)

    filtered_inputs, filtered_outputs = filter_on_entropy(inputs, outputs, p_threshold=10, return_original=debug)
    if captioner is not None:
        filtered_outputs = add_caption_loss(net, captioner, filtered_inputs, filtered_outputs, text_features, targets, id2classes, debug=debug)

    avg_predictions = torch.mean(filtered_outputs, dim=0).unsqueeze(0)
    prediction_entropy = entropy(avg_predictions).item()

    optimizer.zero_grad()
    loss = avg_entropy(filtered_outputs)

    if scaler is None:        
        loss.backward()
        optimizer.step()
    else:
        with torch.cuda.amp.autocast():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
    # show batch
    if debug:
        batch_report(filtered_inputs, filtered_outputs, avg_predictions, targets, id2classes, batch_n=batch_idx)

    prediction = avg_predictions.argmax(dim=1)
    return loss.item(), prediction, prediction_entropy

def tpt_train_loop(data_loader, net, optimizer, cost_function, scaler, writer, id2classes, device="cuda", captioner=None, debug=False):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    top1 = 0
    top5 = 0

    no_tpt_class_acc = {c: [] for c in id2classes.values()}
    tpt_class_acc = {c: [] for c in id2classes.values()}
    loss_diff = 0.0

    optimizer_state = deepcopy(optimizer.state_dict())

    try:
        pbar = tqdm(data_loader, desc="Testing", position=0, leave=True, total=len(data_loader))
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            # Reset the prompt_learner to its initial state and the optimizer to its initial state
            with torch.no_grad():
                net.reset()
                optimizer.load_state_dict(optimizer_state)

            _loss, no_tpt_prediction, no_tpt_prediction_entropy = tta_net_train((batch_idx, inputs, targets), net, optimizer, scaler, id2classes, device=device, captioner=captioner, debug=debug)

            net.eval()
            with torch.no_grad():
                # Classification with the updated net
                inputs = inputs[0].unsqueeze(0).to(device)
                targets = targets.to(device)
                outputs, _ = net(inputs)
                loss = cost_function(outputs, targets)
                prediction = outputs.argmax(dim=1)
                prediction_entropy = entropy(prediction).item()

                cumulative_loss += loss.item()
                samples += 1

            # Update accuracies
            # ! this is not correct, we are not computing the accuracy 
            # TODO fix this
            # TODO create a specific class to handle the metrics operations hiding details
            if no_tpt_prediction.item() == targets.item():
                no_tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(1)
            else:
                no_tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(0)

            values, predictions = outputs.topk(5)
            if prediction == targets:
                top1 += 1
                tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(1)
            else:
                tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(0)
                pass
            if targets.item() in predictions:
                top5 += (targets.view(-1, 1) == predictions).sum().item()

            if debug:
                top5_str = [id2classes[pred] for pred in predictions[0].tolist()]
                target_str = id2classes[targets.item()]
                report_predictions(batch_idx, top5_str, values, target_str)

            loss_diff +=  _loss - loss.item() # comparison of loss with and without TPT
            entropy_diff = prediction_entropy - no_tpt_prediction_entropy # comparison of entropy with and without TPT
            # Log Values
            writer.add_scalar("Delta_loss/test", loss_diff, batch_idx)
            writer.add_scalar("Delta_entropy/test", entropy_diff, batch_idx)

            pbar.set_postfix(test_loss=loss.item(), top1=top1/samples * 100, top5=top5/samples * 100)
            pbar.update(1)

    except KeyboardInterrupt:
        print("User keyboard interrupt")

    except Exception:
        for c in id2classes.values():
            if len(no_tpt_class_acc[c]) == 0 or len(tpt_class_acc[c]) == 0:
                continue
            no_tpt_acc = sum(no_tpt_class_acc[c]) / len(no_tpt_class_acc[c])
            tpt_acc = sum(tpt_class_acc[c]) / len(tpt_class_acc[c])
            writer.add_scalar(f"Class accuracy/{c}", no_tpt_acc, 0)
            writer.add_scalar(f"Class accuracy/{c}", tpt_acc, 1)
        # TODO plot histogram
            pbar.close()
        raise

    # Draw histogram of class accuracies
    no_tpt_accuracies, accuracies = compute_accuracies(id2classes, no_tpt_class_acc, tpt_class_acc)
    image = make_histogram(no_tpt_accuracies, accuracies, 'No TPT','TPT', save_path=f"runs/{RUN_NAME}/accuracy_by_class.png")
    writer.add_image("Class accuracies", image, 0, dataformats="HWC")

    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def main(
    dataset_name="imagenet_a",
    backbone="RN50",
    device="mps",
    batch_size=64,
    learning_rate=0.005,
    tta_steps=2,
    run_name="exp6",
    n_ctx=4,
    ctx_init="a_photo_of_a",
    class_token_position="end",
    csc=False,
    ice_loss=True,
    debug=DEBUG
):
    print("Using manual seed")
    torch.manual_seed(0)
    # Create a logger for the experiment
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    RUN_NAME = run_name

    _, preprocess = clip.load(backbone, device=device)
    
    data_transform = Augmixer(preprocess, batch_size, augmix=True, severity=1)
    # Get dataloaders
    _, _, test_loader, classnames, id2class = get_data(
        dataset_name, 1, data_transform, train_size=0, val_size=0, shuffle=True
    )    

    # Instantiate the network and move it to the chosen device (GPU)
    net = OurCLIP(
        classnames=classnames,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        class_token_position=class_token_position,
        backbone=backbone,
        csc=csc,
    ).to(device)

    load_pretrained_coop(backbone, net, device)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in net.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(
        f"Total trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}"
    )

    trainable_param = net.prompt_learner.parameters()
    optimizer = get_optimizer(trainable_param, learning_rate)

    cost_function = get_loss_function()

    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    else:
        scaler = None

    # Instantiate the captioner if needed
    captioner = None
    if ice_loss:
        model_name = "coca_ViT-L-14"
        version = "laion2B-s13B-b90k"
        captioner = Captioner(model_name=model_name, version=version, device=device)

    print(f"Beginning testing with TPT + ice_loss={ice_loss}:")
    test_loss, test_accuracy = tpt_train_loop(test_loader, net, optimizer, cost_function, scaler, writer, id2classes=id2class, device=device, captioner=captioner, debug=debug)
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    # Closes the logger
    
    writer.close()


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    main(device=DEVICE)
