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
            x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
        else:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        
        if has_categorical_features:
            y_pred = model(x, c)
        else:
            y_pred = model(x)
        
        loss = criterion(y_pred.flatten(), y.flatten())

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
    test_loss = 0.0
    n_batches = len(dataloader)
    criterion = nn.MSELoss()
    all_y_pred = []
    count_preds = 0

    with torch.no_grad():
        for batch in dataloader:
            if has_categorical_features:
                x, y, c = batch
                x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
            else:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)

            if has_categorical_features:
                y_pred = model(x, c)
            else:
                y_pred = model(x)

            test_loss += criterion(y_pred.squeeze().flatten(), y.squeeze().flatten())
            count_preds += y_pred.squeeze().flatten().shape[0]
            all_y_pred.append(y_pred.squeeze().detach().cpu().numpy())       

    return test_loss / n_batches, all_y_pred

