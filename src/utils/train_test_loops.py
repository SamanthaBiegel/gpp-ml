import torch
import torch.nn as nn

def train_loop(dataloader, model, optimizer, DEVICE, steps, writer=None, has_categorical_features=False):
    """
    Train the model for one epoch.
    
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        DEVICE (torch.device): The device to use for training.
        steps (int): The current step in the training process.
        writer (SummaryWriter, optional): TensorBoard writer for logging. Defaults to None.
        has_categorical_features (bool, optional): Whether the model has categorical features. Defaults to False.
    Returns:
        tuple: Average training loss and the trained model.
    """
    model.train()
    train_loss = 0.0
    n_batches = len(dataloader)
    criterion = nn.MSELoss()

    for batch in dataloader:
        if has_categorical_features:
            x, y, c = batch
            x, y, c = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True), c.to(DEVICE, non_blocking=True)
        else:
            x, y = batch
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        
        if has_categorical_features:
            y_pred = model(x, c)
        else:
            y_pred = model(x)
        
        loss = criterion(y_pred.squeeze(-1), y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for name, param in model.named_parameters():
            if param.grad is not None and writer is not None:
                writer.add_scalar(f"gradients/{name}", param.grad.norm(), steps)        
        train_loss += loss.item()

    model.eval()

    return train_loss / n_batches, model


def test_loop(dataloader, model, DEVICE, has_categorical_features=False):
    """
    Evaluate the model on the validation or test set.
    
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation/test data.
        model (torch.nn.Module): The model to evaluate.
        DEVICE (torch.device): The device to use for evaluation.
        has_categorical_features (bool, optional): Whether the model has categorical features. Defaults to False.
    Returns:
        tuple: Average test loss and predictions.
    """

    model.eval()
    total_loss = 0.0
    total_count = 0
    criterion = nn.MSELoss()
    all_y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            if has_categorical_features:
                x, y, c = batch
                x, y, c = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True), c.to(DEVICE, non_blocking=True)
            else:
                x, y = batch
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            T = x.shape[1]
            
            if has_categorical_features:
                preds = model(x, c)
            else:
                preds = model(x)
            loss = criterion(preds.squeeze(-1), y)
            total_loss += loss.item() * T
            total_count += T
            
            all_y_pred.append(preds.squeeze().detach().cpu().numpy())
     
    return total_loss / total_count, all_y_pred
