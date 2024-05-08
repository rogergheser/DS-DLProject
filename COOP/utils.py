import torch

def get_optimizer(model, lr, wd, momentum):
    optimizer = torch.optim.SGD([
        {"params": model.parameters()}
    ], lr=lr, weight_decay=wd, momentum=momentum)

    return optimizer

def get_cost_function():
    cost_function = torch.nn.CrossEntropyLoss()
    return cost_function

def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)