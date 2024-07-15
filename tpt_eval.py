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
from COOP.utils import get_optimizer, get_cost_function, log_values
from COOP.functions import training_step, test_step
from COOP.dataloader import get_data
from loaders import Augmixer, load_pretrained_coop
from tqdm import tqdm
from utils import (entropy, avg_entropy, batch_report, filter_on_entropy,
                report_predictions, make_histogram, compute_accuracies)
from copy import deepcopy

DEBUG = True
RUN_NAME = "exp6"

def tta_net_train(batch, net, optimizer, scaler, cost_function, id2classes, device="cuda", debug=False):
    batch_idx, inputs, targets = batch

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = net(inputs).softmax(-1)

    filtered_inputs, filtered_outputs = filter_on_entropy(inputs, outputs, p_threshold=10, return_original=debug)

    avg_predictions = torch.mean(filtered_outputs, dim=0).unsqueeze(0)
    prediction_entropy = entropy(avg_predictions).item()

    optimizer.zero_grad()
    loss = cost_function(avg_predictions, targets)
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

def tpt_train_loop(data_loader, net, optimizer, scaler, cost_function, writer, id2classes, device="cuda", debug=False):
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

            _loss, no_tpt_prediction, no_tpt_prediction_entropy = tta_net_train((batch_idx, inputs, targets), net, optimizer, scaler, cost_function, id2classes, device=device, debug=debug)

            net.eval()
            with torch.no_grad():
                # Classification with the updated net
                inputs = inputs[0].unsqueeze(0).to(device)
                targets = targets.to(device)
                outputs = net(inputs)
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

    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    else:
        scaler = None
    # Define the cost function
    cost_function = get_cost_function()

    print("Beginning testing with TPT:")
    test_loss, test_accuracy = tpt_train_loop(test_loader, net, optimizer, scaler, cost_function, writer, id2classes=id2class, device=device, debug=debug)
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
