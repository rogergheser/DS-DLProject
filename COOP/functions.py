import torch
from tqdm import tqdm

def training_step(net, data_loader, optimizer, cost_function, device="cuda"):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    
    # Set the network to training mode
    net.train()
    
    # Iterate over the training set
    pbar = tqdm(data_loader, desc="Training", position=0, leave=True, total=len(data_loader))
    for batch_idx, (inputs, targets, _) in enumerate(data_loader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs, _ = net(inputs)
        
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
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            # Load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs, _ = net(inputs)
            
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