import torch

def get_optimizer(params, lr):
    optimizer = torch.optim.AdamW(params, lr)
    return optimizer

def get_loss_function():
    loss_function = torch.nn.CrossEntropyLoss()
    return loss_function

def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)