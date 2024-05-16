import torch
from torch.utils.tensorboard import SummaryWriter
from CLIP import clip

from COOP.models import OurCLIP
from COOP.utils import get_optimizer, get_cost_function, log_values
from COOP.functions import training_step, test_step
from COOP.dataloader import get_data
import glob

def find_files(wildcard_pattern):
    return glob.glob(wildcard_pattern)

def load_pretrained_coop(backbone, _model):
    if backbone.lower() == "rn50":
        _backbone = "rn50*"
    elif backbone.lower() == "rn101":
        _backbone = "rn101*"
    elif backbone.lower() == "vit_b16":
        _backbone = "vit_b16*"
    elif backbone.lower() == "vit_b32":
        _backbone = "vit_b32*"
    else:
        raise ValueError(f"Unknown backbone {backbone}")

    #### !TODO #### Fix path string builder
    path = "bin/coop/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"

    pretrained_ctx = torch.load(path, map_location='mps')['state_dict']['ctx']
    assert pretrained_ctx.size()[0] == _model.prompt_learner.n_ctx, f"Number of context tokens mismatch: {_model.n_ctx} vs {pretrained_ctx.size()[0]}"
    with torch.no_grad():
        _model.prompt_learner.ctx.copy_(pretrained_ctx)
        _model.prompt_learner.ctx_init_state = pretrained_ctx

    return _model
    
def main_coop(
    dataset_name="imagenet_a",
    backbone="RN50",
    batch_size=64,
    num_classes=10,
    device="mps",
    learning_rate=0.002,
    weight_decay=0.0005,
    momentum=0.9,
    epochs=2,
    run_name="exp1",
    n_ctx=4,
    ctx_init="",
    class_token_position="end",
    csc=False,
):
    # Create a logger for the experiment
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    _, preprocess = clip.load(backbone, device=device)
    
    # Get dataloaders
    train_loader, val_loader, test_loader, classnames, id2class = get_data(
        dataset_name, batch_size, preprocess
    )

    ###### TODO: Finish Augmentation procedure ######

    # Instantiate the network and move it to the chosen device (GPU)
    net = OurCLIP(
        classnames=classnames,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        class_token_position=class_token_position,
        backbone=backbone,
        csc=csc,
    ).to(device)

    net =  load_pretrained_coop(backbone, net)

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

    # Computes evaluation results before training
    print("Before training:")
    train_loss, train_accuracy = test_step(
        net, train_loader, cost_function, device=device
    )
    val_loss, val_accuracy = test_step(net, val_loader, cost_function, device=device)
    test_loss, test_accuracy = test_step(net, test_loader, cost_function, device=device)

    # Log to TensorBoard
    log_values(writer, -1, train_loss, train_accuracy, "train")
    log_values(writer, -1, val_loss, val_accuracy, "validation")
    log_values(writer, -1, test_loss, test_accuracy, "test")

    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")

    # For each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        train_loss, train_accuracy = training_step(
            net, train_loader, optimizer, cost_function, device=device
        )
        val_loss, val_accuracy = test_step(
            net, val_loader, cost_function, device=device
        )

        log_values(writer, e, train_loss, train_accuracy, "train")
        log_values(writer, e, val_loss, val_accuracy, "validation")

    # Compute final evaluation results
    print("After training:")
    train_loss, train_accuracy = test_step(
        net, train_loader, cost_function, device=device
    )
    val_loss, val_accuracy = test_step(net, val_loader, cost_function, device=device)
    test_loss, test_accuracy = test_step(net, test_loader, cost_function, device=device)

    log_values(writer, epochs, train_loss, train_accuracy, "train")
    log_values(writer, epochs, val_loss, val_accuracy, "validation")
    log_values(writer, epochs, test_loss, test_accuracy, "test")
    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
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

    main_coop(device=DEVICE)
