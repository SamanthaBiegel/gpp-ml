import numpy as np
from data.dataset import *
from utils.train_test_loops import *
from utils.evaluate_model import evaluate_model
import copy


def train_model(train_dl, val_dl, model, optimizer, scheduler, n_epochs, DEVICE, patience, writer=None, early_stopping = True, cat = False):
    """
    Train the model using the provided training and validation data loaders.

    Args:
        train_dl (DataLoader): DataLoader for training data.
        val_dl (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        n_epochs (int): Number of epochs to train the model.
        DEVICE (str): Device to use for training.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        writer (SummaryWriter, optional): TensorBoard writer for logging. Defaults to None.
        early_stopping (bool, optional): Flag to enable early stopping. Defaults to True.
        cat (bool, optional): Flag indicating if categorical features are used. Defaults to False.
    Returns:
        dict: Dictionary containing validation metrics.
        model (nn.Module): The trained model.
    """
    best_loss = np.inf

    # Train the model
    for epoch in range(n_epochs):
        
        # Perform one round of training, doing backpropagation for each training batch
        global_steps = len(train_dl)*epoch
        train_loss, model = train_loop(train_dl, model, optimizer, DEVICE, global_steps, writer, cat)

        # Evaluate model on val set
        val_loss, y_pred = test_loop(val_dl, model, DEVICE, cat)
        val_metrics, _ = evaluate_model(val_dl, y_pred)

        scheduler.step(val_loss)
        
        if writer is not None:
            writer.add_scalar("mse_loss/train", train_loss, epoch)
            writer.add_scalar("mse_loss/validation", val_loss, epoch)
            writer.add_scalar("r2_mean/validation", val_metrics["r2"], epoch)
            writer.add_scalar("rmse/validation", val_metrics["rmse"], epoch)
            writer.add_scalar("nmae/validation", val_metrics["nmae"], epoch)
            writer.add_scalar("abs_bias/validation", val_metrics["abs_bias"], epoch)
            writer.add_scalar("nse/validation", val_metrics["nse"], epoch)

        if early_stopping:
            # Save the model from the best epoch, based on the validation loss
            if val_loss <= best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            # Check for early stopping
            if patience_counter >= patience:
                model.load_state_dict(best_model)
                return best_metrics, model

    if early_stopping:
        return best_metrics, model
    else:
        return val_metrics, model