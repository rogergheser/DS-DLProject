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

def batch_report(inputs, outputs, targets, id2classes, batch_n):
    from matplotlib import pyplot as plt
    # Fetch prediction and loss value
    # prediction = outputs.argmax(dim=1)
    probabilities, predictions = outputs.cpu().topk(5)
    probabilities = probabilities.detach().numpy()
    predictions = predictions.detach()

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    mean = torch.tensor(clip_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(clip_std).reshape(1, 3, 1, 1)

    # Denormalize the batch of images
    denormalized_images = inputs.cpu() * std + mean

    # Visualise the input using matplotlib
    images = [image.numpy().astype('uint8').transpose(1, 2, 0) for image in denormalized_images] # Convert to numpy and transpose to (H, W, C)

    # Visualise the input using matplotlib
    label = id2classes[targets[0].item()]
    plt.figure(figsize=(16,10))
    plt.title(f"Image batch of {label} - 8/{len(images)} selected")

    for i, image in enumerate(images[:8]):
        plt.subplot(4,4, 2*i+1)
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(4,4, 2*i+2)
        y = np.arange(probabilities.shape[-1])
        plt.grid()
        plt.barh(y, probabilities[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [id2classes[index] for index in predictions[i].numpy()])
        plt.xlabel("probability")

    plt.savefig(f"batch_reports/Batch{batch_n}.png")

def tta_net_train(batch, net, optimizer, cost_function, id2classes, device="cuda"):
    batch_idx, inputs, targets = batch
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
    filtered_inputs = inputs[indices]
    avg_predictions = torch.mean(filtered_outputs, dim=0).unsqueeze(0)

    # show batch
    batch_report(filtered_inputs, filtered_outputs, targets, id2classes, batch_n=batch_idx)

    loss = cost_function(avg_predictions, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return net

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
        trained_net = tta_net_train((batch_idx, inputs, targets), original_net, original_optimizer, cost_function, id2classes, device=device)

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
