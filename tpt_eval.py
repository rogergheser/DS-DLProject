import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
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
from utils import entropy
import numpy as np
from utils import show_image, show_image_batch

def tta_net_train(batch, net, optimizer, cost_function, device="cuda"):
    inputs, targets = batch
    # Set the network to training mode
    net.train()

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = net(inputs)

    # Filter out the predictions with high entropy
    entropies = [entropy(t).item() for t in outputs.softmax(-1)]
    # Calculate the threshold for the lowest 10% entropies
    threshold = np.percentile(entropies, 15)

    outputs = outputs.softmax(-1)
    entropies = [0 if val > threshold else val for val in entropies]
    indices = torch.nonzero(torch.tensor(entropies)).squeeze(1)
    filtered_outputs = outputs[indices]
    avg_predictions = torch.mean(filtered_outputs, dim=0).unsqueeze(0) 

    loss = cost_function(avg_predictions, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return net

def batch_report(input, outputs, targets, id2classes):
    # Fetch prediction and loss value
    prediction = outputs.argmax(dim=1)
    probabilities, predictions = outputs.topk(5)
    text_str = f"True label: {id2classes[targets[0].item()]}\n"
    text_str += "Predicted labels:\n"
    for i, (p, pred) in enumerate(zip(probabilities[0], predictions[0])):
        text_str += f"{id2classes[pred.item()]}: {p:.2f}\n"
    # Visualise the input using matplotlib
    show_image_batch(input, id2classes[targets[0].item()], text_str)
    


def tpt_train_loop(data_loader, net, optimizer, cost_function, writer, id2classes, device="cuda"):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    top1 = 0
    top5 = 0
    original_net = net
    original_optimizer = optimizer

    # Disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    pbar = tqdm(data_loader, desc="Testing", position=0, leave=True, total=len(data_loader))
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        #Optimize prompts using TTA and augmentations
        trained_net = tta_net_train((inputs, targets), original_net, original_optimizer, cost_function, device=device)

        #Evaluate the trained prompts on the single sample
        original_sample = inputs[0].unsqueeze(0)
        with torch.no_grad():
            trained_net.eval()
            # Load data into GPU
            original_sample = original_sample.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = trained_net(original_sample)

            loss = cost_function(outputs, targets)

            # Fetch prediction and loss value
            # samples += inputs.shape[0]
            samples += 1 # In TTA we have augmentations of 64, so in reality we are passing a single sample
            cumulative_loss += loss.item()
            #! check this function that return shape [200], instead of [1]
            prediction = outputs.argmax(dim=1)
            values, predictions = outputs.topk(5)

            if prediction == targets:
                top1 += 1
            if targets.item() in predictions:
                top5 += (targets.view(-1, 1) == predictions).sum().item()
            
            # show batch
            batch_report(inputs, outputs, targets, id2classes)

            pbar.set_postfix(test_loss=loss.item(), top1=top1/samples * 100, top5=top5/samples * 100)
            pbar.update(1)

    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def main(
    dataset_name="imagenet_a",
    backbone="RN50",
    device="mps",
    batch_size=64,
    learning_rate=0.002,
    weight_decay=0.0005,
    momentum=0.9,
    tta_steps=2,
    run_name="exp1",
    n_ctx=4,
    ctx_init="a_photo_of_a",
    class_token_position="end",
    csc=False,
):
    # Create a logger for the experiment
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    _, preprocess = clip.load(backbone, device=device)

    
    data_transform = Augmixer(preprocess, batch_size)
    # Get dataloaders
    _, _, test_loader, classnames, id2class = get_data(
        dataset_name, 1, data_transform, train_size=0, val_size=0
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

    print("Turning off gradients in both the image and the text encoder")
    for name, param in net.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(
        f"Total trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}"
    )

    # Instantiate the optimizer
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    # Define the cost function
    cost_function = get_cost_function()


    print("Beginning testing with TPT:")
    test_loss, test_accuracy = tpt_train_loop(test_loader, net, optimizer, cost_function, writer, id2classes=id2class, device=device)
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
