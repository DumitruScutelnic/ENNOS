import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from .metrics import *

def training_loop(epochs: int, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, processing: bool=False, device: torch.device=None) -> None:
    """
    Training loop for the model.

    Args:
        epochs: Number of training epochs.
        model: The model to train.
        dataloader: The training data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
        processing: Whether to use processing mode (default: False).
        device: The device to train on (default: None).
    """
    # Put model into training mode
    model.train()

    print(f"Start training on {device} [...]\n")
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch+1}/{epochs}")
        train_loss = 0
        train_dice = 0
        train_iou = 0
        train_mse = 0
        train_psnr = 0
        
        # Add a loop to loop through the training batches
        for _, (X, y) in enumerate(dataloader):
            # Put data on target device
            X, y = X.to(device), y.to(device)

            # if processing == False:
            #     y = torch.where(y > 0, 1, 0).type(torch.float)

            # 1. Forward pass (outputs the raw logits from the model)
            y_logits = model(X)
            y_pred = torch.sigmoid(y_logits)

            # 2. Calculate loss and accuracy(per batch)
            loss = loss_fn(y_logits, y)
            train_loss += loss.item() # accumulate train loss

            if processing == False:
                train_dice += dice_coef(y, torch.round(y_pred))
                train_iou += calculate_iou(y, torch.round(y_pred))
            else:
                train_mse += F.mse_loss(y_logits, y).item()
                train_psnr += (10 * torch.log10(1 / F.mse_loss(y_logits, y))).item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        # Divide total train loss and acc by length of train dataloader
        train_loss /= len(dataloader)
        if processing == False:
            train_dice /= len(dataloader)
            train_iou /= len(dataloader)
            print(f"Train loss: {train_loss:.5f}  Train DICE: {train_dice*100:.2f}% | Train IoU: {train_iou*100:.2f}%")
        else:
            train_mse /= len(dataloader)
            train_psnr /= len(dataloader)
            print(f"Train loss: {train_loss:.5f} | Train MSE score: {train_mse:.4f} | Train PSNR score: {train_psnr:.4f}")

def testing_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, processing: bool=False, device: torch.device=None, test: bool=True, show: bool=False) -> None:
    """
    Testing loop for the model.

    Args:
        model: The model to test.
        dataloader: The test data loader.
        loss_fn: The loss function.
        processing: Whether to use processing mode (default: False).
        device: The device to test on (default: None).
        test: Whether this is a test loop (default: True).
        show: Whether to display the results (default: False).
    """
    test_loss = 0
    test_dice = 0
    test_iou = 0
    test_ap = 0
    test_mse = 0
    test_psnr = 0

    # Put the model in eval mode
    model.eval()

    print("Testing...")
    # Turn on inference mode context manager
    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            # Send the data to the target device
            X, y = X.to(device), y.to(device)

            print(y.unique())

            # if processing == False:
            #     y = torch.where(y > 0, 1, 0).type(torch.float)

            # 1. Forward pass (outputs raw logits)
            y_logits = model(X)
            y_pred = torch.sigmoid(y_logits)

            # 2. Calculate the loss/acc
            test_loss += loss_fn(y_logits, y).item()

            if processing == False:
                test_dice += dice_coef(y, torch.round(y_pred))
                test_iou += calculate_iou(y, torch.round(y_pred))
                test_ap += calculate_average_precision(predictions=y_pred, targets=y)
            else:
                test_mse += F.mse_loss(y_logits, y).item()
                test_psnr += (10 * torch.log10(1 / F.mse_loss(y_logits, y))).item()

            if show:
                display_batch_images_and_masks(images=X, gts=y, preds=y_logits, processing=processing, alpha=0.5)
                

        # Adjust metrics and print out
        test_loss /= len(dataloader)
        if processing == False:
            test_dice /= len(dataloader)
            test_iou /= len(dataloader)
            test_ap /= len(dataloader)
            if test:
                print(f"Test loss: {test_loss:.5f} | Test DICE: {test_dice*100:.2f}% | Test IoU: {test_iou*100:.2f} | Test mAP: {test_ap*100:.2f}%")
            else:
                print(f"Val loss: {test_loss:.5f} | Val DICE: {test_dice*100:.2f}% | Val IoU: {test_iou*100:.2f}% | Val mAP: {test_ap*100:.2f}%")
        else:
            test_mse /= len(dataloader)
            test_psnr /= len(dataloader)
            if test:
                print(f"Test loss: {test_loss:.5f} | Test MSE score: {test_mse:.4f} | Test PSNR score: {test_psnr:.4f}")
            else:
                print(f"Val loss: {test_loss:.5f} | Val MSE score: {test_mse:.4f} | Val PSNR score: {test_psnr:.4f}")
                
def save_model(model: torch.nn.Module, model_name: str) -> None:
    """
    Saves the model state dictionary to the specified directory.
    
    Args:
        model (torch.nn.Module): The model to save.
        target_dir (str): Directory to save the model.
        model_name (str): Name of the model file.
    """
    # Create target directory if it doesn't exist
    if not os.path.exists('weights'):
        os.makedirs('weights')
    
    # Save the model state dictionary
    torch.save(model.state_dict(), os.path.join('weights', model_name))
    print(f"Model saved to {os.path.join('weights', model_name)}")
    
def load_model(model: torch.nn.Module, model_name: str, device: torch.device) -> None:
    """
    Loads the model state dictionary from the specified directory.
    
    Args:
        model (torch.nn.Module): The model to load the state dictionary into.
        model_name (str): Name of the model file.
    """
    # Load the model state dictionary
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device), weights_only=True))
    print(f"Model loaded from {model_name}")

def display_batch_images_and_masks(
    images: torch.Tensor, 
    gts: torch.Tensor, 
    preds: torch.Tensor, 
    processing: bool = False, 
    alpha: float = 0.5, 
    delay: float = 2.0
) -> None:
    """
    Displays images, ground truths, and predictions as a slideshow.

    Args:
        images (torch.Tensor): Batch of input images, shape (B, C, H, W).
        gts (torch.Tensor): Batch of ground-truth masks, shape (B, C, H, W).
        preds (torch.Tensor): Batch of predictions, shape (B, C, H, W).
        processing (bool): If True, shows raw predictions. If False, applies sigmoid + threshold.
        alpha (float): Transparency for overlay in non-processing mode.
        delay (float): Seconds to pause between images.
    """
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(len(images)):
        # Convert tensors to numpy
        img = images[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        gt = gts[i].cpu().numpy().transpose(1, 2, 0).squeeze()

        if processing:
            pred = preds[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        else:
            pred = torch.sigmoid(preds[i]).cpu().numpy().transpose(1, 2, 0).squeeze()
            pred = np.round(pred)

        # Clear previous plots
        for ax in axes:
            ax.clear()

        # Show images
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(img, cmap="gray")
        axes[2].imshow(pred, cmap="jet" if processing else "gray", alpha=alpha)
        axes[2].set_title("Prediction Overlay" if not processing else "Prediction")
        axes[2].axis("off")

        plt.pause(delay)

    plt.ioff()
    plt.close(fig)
    
def print_execution_time(start: float, end: float, device: torch.device=None):
    """
    Prints difference between start and end time.

    Args:
        start (float): The start time.
        end (float): The end time.
        device (torch.device, optional): The device used for computation.
    """
    total_time = end - start
    print(f"\nExecution time on {device} : {total_time:.3f} seconds")