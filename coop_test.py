import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import tqdm
_tokenizer = _Tokenizer()

def get_optimizer(model, lr, wd, momentum):
    optimizer = torch.optim.SGD([
        {"params": model.classifier.parameters(), "lr": lr}
    ], lr=lr / 10, weight_decay=wd, momentum=momentum)

    return optimizer

def get_cost_function():
    cost_function = torch.nn.CrossEntropyLoss()
    return cost_function

def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)

def training_step(net, data_loader, optimizer, cost_function, device="cuda"):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    
    # Set the network to training mode
    net.train()
    
    # Iterate over the training set
    pbar = tqdm(data_loader, desc="Training", position=0, leave=True, total=len(data_loader))
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = net(inputs)
        
        # Loss computation
        loss = cost_function(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Parameters update
        optimizer.step()
        
        # Gradients reset
        optimizer.zero_grad()
        
        # Fetch prediction and loss value
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1) # max() returns (maximum_value, index_of_maximum_value)
        
        # Compute training accuracy
        cumulative_accuracy += predicted.eq(targets).sum().item()

        pbar.set_postfix(train_loss=loss.item(), train_acc=cumulative_accuracy / samples * 100)
        pbar.update(1)
    
    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def test_step(net, data_loader, cost_function, device="cuda"):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    
    # Set the network to evaluation mode
    net.eval()
    
    # Disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    pbar = tqdm(data_loader, desc="Testing", position=0, leave=True, total=len(data_loader))
    with torch.no_grad():
        # Iterate over the test set
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = net(inputs)
            
            # Loss computation
            loss = cost_function(outputs, targets)
            
            # Fetch prediction and loss value
            samples += inputs.shape[0]
            cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
            _, predicted = outputs.max(1)
            
            # Compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

            pbar.set_postfix(test_acc=cumulative_accuracy / samples * 100)
            pbar.update(1)

    return cumulative_loss / samples, cumulative_accuracy / samples * 100

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # [batch_size, n_ctx, transformer.width] -> [n_ctx, batch_size, transformer.width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [n_ctx, batch_size, transformer.width] -> [batch_size, n_ctx, transformer.width]
        x = self.ln_final(x)

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx, ctx_init, class_token_position, csc=False):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        # Use given words to initialize context vectors
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim)

            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"Initial context: '{prompt_prefix}'")
        print(f"Number of context words (tokens): {n_ctx}")

        # These are the `prompts` we want to optimize
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # print("+++")
        # print("Prompts:")
        # for p in prompts:
        #     print(p)
        # print("+++")

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(clip_model.token_embedding.weight.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        
        # If CoOp, expand the ctx for all classes
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
class OurCLIP(nn.Module):
    def __init__(self, classnames, n_ctx, ctx_init, class_token_position, csc=False):
        super().__init__()
        clip_model, _ = clip.load("RN50")
        # clip_model = clip_model.cpu()
        clip_model = clip_model.float()
        
        self.prompt_learner = PromptLearner(clip_model, classnames, n_ctx, ctx_init, class_token_position, csc=csc)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        image_features = self.image_encoder(image)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    
def get_optimizer(model, lr, wd, momentum):
    optimizer = torch.optim.SGD([
        {"params": model.parameters()}
    ], lr=lr, weight_decay=wd, momentum=momentum)

    return optimizer

def main_coop(
    dataset_name="cifar10",
    batch_size=16,
    num_classes=10,
    device="cuda:0",
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
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_data(dataset_name, transform=preprocess, batch_size=batch_size)
    classnames, _ = embed_dataset_classnames(dataset_name)
    
    # Instantiate the network and move it to the chosen device (GPU)
    net = OurCLIP(
        classnames=classnames, n_ctx=n_ctx, ctx_init=ctx_init, class_token_position=class_token_position, csc=csc
    ).to(device)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in net.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Total trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    
    # Instantiate the optimizer
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)
    
    # Define the cost function
    cost_function = get_cost_function()
    
    # Computes evaluation results before training
    print("Before training:")
    train_loss, train_accuracy = test_step(net, train_loader, cost_function)
    val_loss, val_accuracy = test_step(net, val_loader, cost_function)
    test_loss, test_accuracy = test_step(net, test_loader, cost_function)
    
    # Log to TensorBoard
    log_values(writer, -1, train_loss, train_accuracy, "train")
    log_values(writer, -1, val_loss, val_accuracy, "validation")
    log_values(writer, -1, test_loss, test_accuracy, "test")
    
    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    
    # For each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        train_loss, train_accuracy = training_step(net, train_loader, optimizer, cost_function)
        val_loss, val_accuracy = test_step(net, val_loader, cost_function)

        log_values(writer, e, train_loss, train_accuracy, "train")
        log_values(writer, e, val_loss, val_accuracy, "validation")

    # Compute final evaluation results
    print("After training:")
    train_loss, train_accuracy = test_step(net, train_loader, cost_function)
    val_loss, val_accuracy = test_step(net, val_loader, cost_function)
    test_loss, test_accuracy = test_step(net, test_loader, cost_function)
    
    log_values(writer, epochs, train_loss, train_accuracy, "train")
    log_values(writer, epochs, val_loss, val_accuracy, "validation")
    log_values(writer, epochs, test_loss, test_accuracy, "test")
    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    
    # Closes the logger
    writer.close()