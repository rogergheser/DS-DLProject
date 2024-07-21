import torch, torchvision
torchvision.disable_beta_transforms_warning()
import sys
import numpy as np
import torch.amp
import os
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
from utils import (entropy, avg_entropy, batch_report, filter_on_entropy, AverageMeter,
                report_predictions, make_histogram, compute_accuracies, caption_report, create_run_info)
from copy import deepcopy
import torch.nn.functional as F
import logging

DEBUG = False
HARMONIC_MEAN=True
RUN_NAME = "exp6"
LOG_FREQUENCY = 10
logger = logging.getLogger(__name__)


def add_caption_loss(net: OurCLIP, captioner: Captioner, batch, text_features, id2classes, prompt="a ", _lambda=0, K=200, debug=False):
    """
    Adds caption loss to the filtered_outputs using the given captioner.

    Args:
        net (OurCLIP): The network used to generate the text features.
        captioner (Captioner): The captioner object used to generate captions.
        batch (tuple): Tuple containing filtered inputs and outputs, batch_idx and label
        text_features: The text features of the labels computed by the model.
        id2classes (dict): The mapping from class index to class name.
        prompt (str): The prompt used for generating captions. Default is "a ".
        _lambda (float): The value of lambda used for computing the weighted logit summation
        K (int): The number of top classes to consider. Default is 200.
        debug (bool): Whether to print debug information. Default is False.

    Returns:
        The updated filtered_outputs with caption loss added.
        The caption prediction from the average of all the logits
    """
    batch_idx, filtered_inputs, filtered_outputs, label = batch
    # Compute captions for each augmentation using coca functions
    device = filtered_inputs.device
    with torch.no_grad(), torch.cuda.amp.autocast():
        captions = captioner.generate_captions(filtered_inputs, prompt)
    
    # Encode all the captions using the clip encoder (batchfying the captions to save compute)
    caption_tokens = clip.tokenize(captions).to(device)
    caption_features = net.encode_text(caption_tokens).to(device)

    
    caption_logits = net.logit_scale.exp()*(F.normalize(caption_features) @ text_features.T)

    caption_logits = caption_logits.softmax(-1)
    image_logits = filtered_outputs

    # Compute the value of lambda following ice implementation row 193 main_ice.py
    assert K == 200, "For k != 200, function has to be implemented"

    if _lambda:
        ice_scores = (1-_lambda)*image_logits + _lambda*caption_logits
    else:
        # Lambda computed as a normalization term
        # std_devs = torch.stack((image_logits.std(dim=1), caption_logits.std(dim=1)), dim=1)
        # coef = 0.08 * F.normalize(std_devs, dim=1)
        # coef = coef[:, 1].unsqueeze(1).expand(-1, K)
        # Sum the image and caption scores to obtain the ICE scores
        # ice_scores = image_logits + coef * caption_logits
        ice_scores = torch.zeros_like(image_logits)
        for batch in range(image_logits.shape[0]):
            A = 1/(1 + entropy(image_logits[batch]).item())
            B = 1/(1 + entropy(caption_logits[batch]).item())
            C = A + B
            if HARMONIC_MEAN:
                ice_scores[batch] = (2 * image_logits[batch] * caption_logits[batch]).div(image_logits[batch] + caption_logits[batch])
            else:
                ice_scores[batch] = (A/C * image_logits[batch] + B/C * caption_logits[batch])

    caption_prediction = torch.mean(caption_logits, dim=0)
    if debug:
        caption_report(filtered_inputs, image_logits, caption_logits, ice_scores, label, captions, caption_prediction, id2classes, batch_idx)    

    return ice_scores
    

def tta_net_train(batch, net, optimizer, scaler, id2classes, device="cuda", captioner=None, debug=False):
    batch_idx, inputs, targets = batch

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs, text_features = net(inputs)
    outputs = outputs.softmax(dim=-1)

    filtered_inputs, filtered_outputs = filter_on_entropy(inputs, outputs, p_percentile=10, return_original=debug)
    if captioner is not None:
        batch = (batch_idx, filtered_inputs, filtered_outputs, targets)
        filtered_outputs = add_caption_loss(net, captioner, batch, text_features, id2classes, debug=debug)

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
    cumulative_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

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

                cumulative_loss.update(loss.item())

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
                top1.update(1)
                tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(1)
            else:
                top1.update(0)
                tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(0)

            if targets.item() in predictions:
                top5.update(1)
            else:
                top5.update(0)

            if debug:
                top5_str = [id2classes[pred] for pred in predictions[0].tolist()]
                target_str = id2classes[targets.item()]
                report_predictions(batch_idx, top5_str, values, target_str)

            loss_diff =  _loss - loss.item() # comparison of loss with and without TPT
            entropy_diff = prediction_entropy - no_tpt_prediction_entropy # comparison of entropy with and without TPT
            # Log Values
            writer.add_scalar("Delta_loss/test", loss_diff, batch_idx)
            writer.add_scalar("Delta_entropy/test", entropy_diff, batch_idx)
            writer.add_scalar("Top-1", top1.get_avg()*100.00, batch_idx)
            writer.add_scalar("Top-5", top5.get_avg()*100.00, batch_idx)
            if batch_idx % LOG_FREQUENCY == 0 :#and batch_idx > 10:
                logger.info(f"[LOSS] Batch {batch_idx} - Delta loss: {loss_diff:.5f}, Delta entropy: {entropy_diff:.5f}")
                no_tpt_accuracies, accuracies = compute_accuracies(id2classes, no_tpt_class_acc, tpt_class_acc)
                histogram = make_histogram(no_tpt_accuracies, accuracies, 
                                        'No TPT', 'TPT', save_path=f"runs/{RUN_NAME}/class_accuracy%{batch_idx}e.png")
                writer.add_image(f"Class accuracies%{batch_idx}e", histogram, batch_idx, dataformats="HWC")
                logger.info(f"[ACC] Batch num:{batch_idx} - Top1: {top1.get_avg() * 100:.2f}, Top5: {top5.get_avg() * 100:.2f}")
            pbar.set_postfix(test_loss=loss.item(), top1=top1.get_avg() * 100.00, top5=top5.get_avg() * 100.00)
            pbar.update(1)

    except KeyboardInterrupt:
        print("User keyboard interrupt")

    # Draw histogram of class accuracies
    no_tpt_accuracies, accuracies = compute_accuracies(id2classes, no_tpt_class_acc, tpt_class_acc)
    image = make_histogram(no_tpt_accuracies, accuracies, 'No TPT','TPT', save_path=f"runs/{RUN_NAME}/accuracy_by_class.png")
    image = make_histogram(no_tpt_accuracies, accuracies, 'No TPT','TPT', save_path=f"runs/{RUN_NAME}/accuracy_by_worst_class.png", worst_case=True)
    writer.add_image("Class accuracies", image, 0, dataformats="HWC")

    return cumulative_loss.get_avg() , top1.get_avg() * 100

def main(
    dataset_name="imagenet_a",
    backbone="ViT-B/16",
    device="mps",
    batch_size=64,
    learning_rate=0.005,
    tta_steps=2,
    run_name=RUN_NAME,
    n_ctx=4,
    ctx_init="a_photo_of_a",
    class_token_position="end",
    csc=False,
    ice_loss=False,
    harmonic_mean=HARMONIC_MEAN,
    debug=DEBUG
):
    HARMONIC_MEAN = harmonic_mean
    DEBUG = debug
    RUN_NAME = run_name

    seed = 0
    print("Using manual seed {}".format(seed))
    torch.manual_seed(seed)
    # Create a logger for the experiment
    run_name = RUN_NAME
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

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
    
    create_run_info(dataset_name, backbone, ice_loss, test_accuracy, run_name, harmonic_mean)
    
    writer.close()


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    logger.setLevel(logging.DEBUG)
    os.makedirs(f"runs/{RUN_NAME}", exist_ok=True)
    
    log_path = f"runs/{RUN_NAME}/log.log"
    if os.path.isfile(log_path):
        os.remove(log_path)

    file_handler = logging.FileHandler(log_path)
    stderr_handler = logging.StreamHandler(sys.stderr)

    file_handler.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.ERROR)

    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stderr_formatter = logging.Formatter('\r%(levelname)s - %(message)s')

    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)

    main(device=DEVICE)
