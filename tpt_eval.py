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
from loaders import Augmixer
from tqdm import tqdm
from utils import entropy, avg_entropy
from copy import deepcopy

DEBUG = True

def load_pretrained_coop(backbone, _model):
    if backbone.lower() == "rn50":
        _backbone = "rn50"
    elif backbone.lower() == "rn101":
        _backbone = "rn101"
    elif backbone.lower() == "vit_b16" or backbone.lower() == "vit-b/16":
        _backbone = "vit_b16"
    elif backbone.lower() == "vit_b32" or backbone.lower() == "vit-b/32":
        _backbone = "vit_b32"
    else:
        raise ValueError(f"Unknown backbone {backbone}")

    path = f"bin/coop/{_backbone}_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"
    assert os.path.exists(path), f"Path {path} does not exist"

    pretrained_ctx = torch.load(path, DEVICE)['state_dict']['ctx']
    assert pretrained_ctx.size()[0] == _model.prompt_learner.n_ctx, f"Number of context tokens mismatch: {_model.prompt_learner.n_ctx} vs {pretrained_ctx.size()[0]}"
    with torch.no_grad():
        _model.prompt_learner.ctx.copy_(pretrained_ctx)
        _model.prompt_learner.ctx_init_state = pretrained_ctx


def batch_report(inputs, outputs, final_prediction, targets, id2classes, batch_n):
    from matplotlib import pyplot as plt
    probabilities, predictions = outputs.cpu().topk(5)
    probabilities = probabilities.detach().numpy()
    predictions = predictions.detach()

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    mean = torch.tensor(clip_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(clip_std).reshape(1, 3, 1, 1)

    # Denormalize the batch of images
    # denormalized_images = inputs.cpu() * std + mean
    # denormalized_images = denormalized_images.numpy().astype('uint8')
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    denormalized_images = unnormalize(inputs)

    # Visualise the input using matplotlib
    images = [image.numpy().transpose(1, 2, 0) for image in denormalized_images.cpu()] # Convert to numpy and transpose to (H, W, C)

    # Visualise the input using matplotlib
    label = id2classes[targets[0].item()]
    plt.figure(figsize=(16,16))
    plt.title(f"Image batch of {label} - min entropy 10 samples selected")
    plt.axis('off')

    for i, image in enumerate(images[:10]):
        plt.subplot(6,4, 2*i+1)
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(6,4, 2*i+2)
        y = np.arange(probabilities.shape[-1])
        plt.grid()
        plt.barh(y, probabilities[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [id2classes[pred] for pred in predictions[i].numpy()])
        plt.xlabel("probability")
    
    avg_prob, avg_pred = final_prediction.cpu().topk(5)
    avg_prob = avg_prob.detach().numpy()
    avg_pred = avg_pred.detach()
    plt.subplot(6,4,22)
    y = np.arange(avg_prob.shape[-1])
    plt.grid()
    plt.barh(y, avg_prob[0])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [id2classes[index] for index in avg_pred[0].numpy()])
    plt.xlabel("Final prediction (avg entropy)")    

    plt.savefig(f"batch_reports/Batch{batch_n}.png")
    plt.close()


def make_histogram(no_tpt_acc: dict, tpt_acc: dict, no_tpt_label: str, tpt_label: str, save_path:str=None)-> Image:
    """
    Creates histogram for class accuracies and log it with tensorboard and saves the plot
    """
    classes = list(no_tpt_acc.keys())
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(dpi=500)
    ax.bar(x - width/2, no_tpt_acc.values(), width, color='b', label=no_tpt_label)
    ax.bar(x + width/2, tpt_acc.values(), width, color='r', label=tpt_label)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Class accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=-90, fontsize=2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image = np.array(image)

    if save_path:
        plt.savefig(save_path)

    return image

def tta_net_train(batch, net, optimizer, scaler, cost_function, id2classes, device="cuda", debug=False):
    batch_idx, inputs, targets = batch
    # Set the network to training mode
    net.train()

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = net(inputs)

    # Filter out the predictions with high entropy
    entropies = [entropy(t).item() for t in outputs.softmax(-1)]
    # Calculate the threshold for the lowest entropies values
    threshold = np.percentile(entropies, 15)
    if scaler is None:
        outputs = outputs.softmax(-1)
        entropies = [0 if val > threshold else val for val in entropies]
        indices = torch.nonzero(torch.tensor(entropies)).squeeze(1)
        filtered_outputs = outputs[indices]
        filtered_inputs = inputs[indices]
        avg_predictions = torch.mean(filtered_outputs, dim=0).unsqueeze(0)
        prediction_entropy = entropy(avg_predictions).item()

        optimizer.zero_grad()
        # loss = cost_function(avg_predictions, targets)
        loss = avg_entropy(filtered_outputs)
        
        loss.backward()
        if debug:
            if torch.isnan(net.prompt_learner.ctx.grad).any():
                print("NaN in context tokens gradient")
                raise ValueError("NaN in context tokens gradient")
            if torch.isinf(net.prompt_learner.ctx.grad).any():
                print("Inf in context tokens gradient")
                raise ValueError("Inf in context tokens gradient")

        optimizer.step()
    else:
        with torch.cuda.amp.autocast():
            outputs = outputs.softmax(-1)
            entropies = [0 if val > threshold else val for val in entropies]
            indices = torch.nonzero(torch.tensor(entropies)).squeeze(1)
            filtered_outputs = outputs[indices]
            filtered_inputs = inputs[indices]
            avg_predictions = torch.mean(filtered_outputs, dim=0).unsqueeze(0)
            prediction_entropy = entropy(avg_predictions).item()
            loss = avg_entropy(filtered_outputs)
            scaler.scale(loss).backward()
            loss.backward()
            if debug:
                if torch.isnan(net.prompt_learner.ctx.grad).any():
                    print("NaN in context tokens gradient")
                    raise ValueError("NaN in context tokens gradient")
                if torch.isinf(net.prompt_learner.ctx.grad).any():
                    print("Inf in context tokens gradient")
                    raise ValueError("Inf in context tokens gradient")
            
            scaler.step(optimizer)
            scaler.update()
    
    if torch.isnan(net.prompt_learner.ctx).any():
        print("NaN in context tokens")
        raise ValueError("NaN in context tokens")
    
    if torch.isinf(net.prompt_learner.ctx).any():
        print("Inf in context tokens")
        raise ValueError("Inf in context tokens")
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
        # Disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
        pbar = tqdm(data_loader, desc="Testing", position=0, leave=True, total=len(data_loader))
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Reset the prompt_learner to its initial state and the optimizer to its initial state
            with torch.no_grad():
                net.reset()
                optimizer.load_state_dict(optimizer_state)

            # Optimize prompts using TTA and augmentations
            # Get prediction without prompt optimization      
            _loss, no_tpt_prediction, no_tpt_prediction_entropy = tta_net_train((batch_idx, inputs, targets), net, optimizer, scaler, cost_function, id2classes, device=device, debug=debug)
            #_loss, no_tpt_prediction, no_tpt_prediction_entropy = 0, torch.tensor(-1), 0

            if no_tpt_prediction.item() == targets.item():
                no_tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(1)
            else:
                no_tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(0)

            # Evaluate the trained prompts on the single sample
            net.eval()
            with torch.no_grad():
                inputs = inputs[0].unsqueeze(0).to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                loss = cost_function(outputs, targets)
                cumulative_loss += loss.item()
                samples += 1
                prediction = outputs.argmax(dim=1)
                prediction_entropy = entropy(prediction).item()

                values, predictions = outputs.topk(5)
                if prediction == targets:
                    top1 += 1
                    tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(1)
                else:
                    tpt_class_acc[id2classes[no_tpt_prediction.item()]].append(0)
                    pass
                if targets.item() in predictions:
                    top5 += (targets.view(-1, 1) == predictions).sum().item()

                top1_str = id2classes[prediction.item()]
                top5_str = [id2classes[pred] for pred in predictions[0].tolist()]
                target_str = id2classes[targets.item()]
                loss_diff +=  _loss - loss.item() # comparison of loss with and without TPT
                entropy_diff = prediction_entropy - no_tpt_prediction_entropy # comparison of entropy with and without TPT
                
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
        raise
        
    pbar.close()
    # Log the final values and class accuracies
    # create histogram for class accuracies and log it with tensorboard
    # Create single histograms for each class with a column for TPT and one for no TPT
    
    no_tpt_accuracies = {}
    accuracies = {}

    for c in id2classes.values():
        if len(no_tpt_class_acc[c]) == 0 or len(tpt_class_acc[c]) == 0:
            continue
        no_tpt_accuracies[c] = sum(no_tpt_class_acc[c]) / len(no_tpt_class_acc[c])
        accuracies[c] = sum(tpt_class_acc[c]) / len(tpt_class_acc[c])
    
    image = make_histogram(no_tpt_accuracies, accuracies, 'No TPT','TPT', save_path="results/imagenet_A/plots/accuracy_by_class.png")
    writer.add_image("Class accuracies", image, 0, dataformats="HWC")

    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def main(
    dataset_name="imagenet_a",
    backbone="RN50",
    device="mps",
    batch_size=16,
    learning_rate=0.005,
    tta_steps=2,
    run_name="exp4",
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

    _, preprocess = clip.load(backbone, device=device)
    
    data_transform = Augmixer(preprocess, batch_size, severity=3)
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

    load_pretrained_coop(backbone, net)

    # Instantiate the optimizer
    #optimizer = get_optimizer(net, learning_rate)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in net.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(
        f"Total trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}"
    )

    trainable_param = net.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, learning_rate)
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
