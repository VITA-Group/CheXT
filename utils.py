"""
Author: Greg Holste
Last Modified: 12/9/21
Description: Utility functions + train/val/test loops for training models on the NIH ChestXRay14 dataset.
"""

import os
import random
import shutil
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils import compute_class_weight

from dataset import ChestXRay14
from models.loss import NTXentLoss

import torch
import torchvision

def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def unsupervised_train(model, device, loss_fxn, ls, optimizer, data_loader, history, epoch, model_dir):
    """Unsupervised Train PyTorch model for one epoch on NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    running_loss = 0.
    mean_auc = 0.
    for i, batch in pbar:
        x = batch['x'].to(device)
        _, _, _, zs = model.forward(x, threshold=0.65)
        # _, _, _, zs = model.forward(x, threshold=None)
        loss = loss_fxn(zs[0], zs[1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (i + 1)})
        current_metrics = pd.DataFrame([[epoch, 'train', running_loss/(i+1)]], columns=history.columns)
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)
    
    best_model_wts = deepcopy(model.state_dict())
    torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))

    return history.append(current_metrics)

def train(model, device, loss_fxn, ls, optimizer, data_loader, history, epoch, model_dir, classes, fusion=False, meta_only=False, chext=False):
    """Train PyTorch model for one epoch on NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        classes : list[str]
            Ordered list of names of output classes
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    running_loss = 0.
    mean_auc = 0.
    y_true, y_hat = [], []
    for i, batch in pbar:
        y = batch['y'].to(device)

        # # Forward pass
        # if meta_only:
        #     meta = batch['meta'].to(device)

        #     yhat = model.forward(meta)
        # elif fusion:
        #     x = batch['x'].to(device)
        #     meta = batch['meta'].to(device)

        #     yhat = model.forward(x, meta)
        # else:
        #     x = batch['x'].to(device)

        #     yhat = model.forward(x)

        if chext:
            x = batch['x'].to(device)

            _, _, _, zs = model.forward(x)

            # loss1 = loss_fxn(output, y)

            loss = NTXentLoss(device, 0.5)(zs[0], zs[1])

            # loss = 0.7*loss1 + 0.3*loss2

        else:
            x = batch['x'].to(device)
            yhat, _ = model.forward(x)

            # Compute loss (with optional label smoothing)
            if ls == 0:
                loss = loss_fxn(yhat, y)
            else:
                loss = loss_fxn(yhat, y * (1-ls) + 0.5 * ls)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep running sum of mean batch losses
        running_loss += loss.item()

        # Collect true and predicted labels
        y_true.append(y.detach().cpu().numpy())
        y_hat.append(torch.sigmoid(yhat).detach().cpu().numpy())

        # # Update progress bar w/ running mean loss and, every 50 iterations, mean AUC
        # if (i + 1) % 50 == 0:
        #     aucs = roc_auc_score(np.concatenate(y_true), np.concatenate(y_hat), average=None)
        #     mean_auc = np.mean(aucs[np.where(np.array(classes) != 'No Finding')[0]])  # important that classes is np array
        
        pbar.set_postfix({'loss': running_loss / (i + 1), 'auc': mean_auc})

    # Collect true and predicted labels into numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)  # n x 15

    # Compute metrics
    aucs = roc_auc_score(y_true, y_hat, average=None)
    mean_auc = np.mean(aucs[np.where(np.array(classes) != 'No Finding')[0]])

    print(f'Mean AUC: {round(mean_auc, 3)}')

    current_metrics = pd.DataFrame([[epoch, 'train', running_loss/(i+1), mean_auc] + list(aucs)], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return history.append(current_metrics)

def validate(model, device, loss_fxn, ls, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, classes, fusion=False, meta_only=False):
    """Evaluate PyTorch model on validation set of NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        early_stopping_dict : dict
            Dictionary of form {'epochs_no_improve': <int>, 'best_loss': <float>} for early stopping
        best_model_wts : PyTorch state_dict
            Model weights from best epoch
        classes : list[str]
            Ordered list of names of output classes
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
        early_stopping_dict : dict
            Updated early stopping metrics
        best_model_wts : PyTorch state_dict
            (Potentially) updated model weights (if best validation loss achieved)
    """
    model.eval()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')
    running_loss = 0.
    mean_auc = 0.
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, batch in pbar:
            y = batch['y'].to(device)

            # Forward pass
            if meta_only:
                meta = batch['meta'].to(device)

                yhat = model.forward(meta)
            elif fusion:
                x = batch['x'].to(device)
                meta = batch['meta'].to(device)

                yhat = model.forward(x, meta)
            else:
                x = batch['x'].to(device)

                yhat = model.forward(x)

            # Compute loss (with optional label smoothing)
            if ls == 0:
                loss = loss_fxn(yhat, y)
            else:
                loss = loss_fxn(yhat, y * (1-ls) + 0.5 * ls)

            # Keep running sum of mean batch losses
            running_loss += loss.item()

            # Collect true and predicted labels
            y_true.append(y.detach().cpu().numpy())
            y_hat.append(torch.sigmoid(yhat).detach().cpu().numpy())

            # # Update progress bar w/ running mean loss and, every 50 iterations, mean AUC
            # if (i + 1) % 50 == 0:
            #     aucs = roc_auc_score(np.concatenate(y_true), np.concatenate(y_hat), average=None)
            #     mean_auc = np.mean(aucs[np.where(np.array(classes) != 'No Finding')[0]])  # important that classes is np array
            
            pbar.set_postfix({'val_loss': running_loss / (i + 1), 'val_auc': mean_auc})

    # Collect true and predicted labels into numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)  # n x 15

    # Compute metrics
    val_loss = running_loss / (i + 1)
    aucs = roc_auc_score(y_true, y_hat, average=None)
    mean_auc = np.mean(aucs[np.where(np.array(classes) != 'No Finding')[0]])

    print(f'Mean VAL AUC: {round(mean_auc, 3)}')

    current_metrics = pd.DataFrame([[epoch, 'val', val_loss, mean_auc] + list(aucs)], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    # Early stopping: save model weights only when val loss has improved
    if val_loss < early_stopping_dict['best_loss']:
        print(f'EARLY STOPPING: Loss has improved from {round(early_stopping_dict["best_loss"], 3)} to {round(val_loss, 3)}! Saving weights.')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_loss'] = val_loss
        best_model_wts = deepcopy(model.state_dict())
        torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))
    else:
        print(f'EARLY STOPPING: Loss has not improved from {round(early_stopping_dict["best_loss"], 3)}')
        early_stopping_dict['epochs_no_improve'] += 1


    return history.append(current_metrics), early_stopping_dict, best_model_wts


def evaluate(model, data_dir, device, loss_fxn, ls, batch_size, history, model_dir, weights, n_TTA=0, fusion=False, meta_only=False):
    """Evaluate PyTorch model on test set of NIH ChestXRay14 dataset. Saves training history csv, summary text file, training curves, etc.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        batch_size : int
        history : pandas DataFrame
            Data frame containing history of training metrics
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        weights : PyTorch state_dict
            Model weights from best epoch
        n_TTA : int
            Number of augmented copies to use for test-time augmentation (0-K)
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    """
    model.load_state_dict(weights)  # load best weights
    model.eval()

    ## INFERENCE
    test_dataset = ChestXRay14(data_dir=data_dir, split="test", augment=False, n_TTA=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=4, worker_init_fn=val_worker_init_fn)

    # Evaluation with no test-time augmentation
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc=f'TEST EVAL')
        running_loss = 0.
        auc = 0
        y_true, y_hat = [], []
        for i, batch in pbar:
            y = batch['y'].to(device)

            # Forward pass
            if meta_only:
                meta = batch['meta'].to(device)

                yhat = model.forward(meta)
            elif fusion:
                x = batch['x'].to(device)
                meta = batch['meta'].to(device)

                yhat = model.forward(x, meta)
            else:
                x = batch['x'].to(device)

                yhat = model.forward(x)

            # Compute loss
            if ls == 0:
                loss = torch.nn.BCEWithLogitsLoss()(yhat, y)
            else:
                loss = torch.nn.BCEWithLogitsLoss()(yhat, y * (1-ls) + 0.5 * ls)

            running_loss += loss.item()

            y_true.append(y.detach().cpu().numpy())
            y_hat.append(torch.sigmoid(yhat).detach().cpu().numpy())

            # if (i + 1) % 50 == 0:
            #     aucs = roc_auc_score(np.concatenate(y_true), np.concatenate(y_hat), average=None)
            #     auc = np.mean(aucs[np.where(test_dataset.CLASSES != 'No Finding')[0]])

            pbar.set_postfix({'test_loss': running_loss / (i + 1), 'test_auc': auc})

    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    aucs = roc_auc_score(y_true, y_hat, average=None)
    auc = np.mean(aucs[np.where(np.array(test_dataset.CLASSES) != 'No Finding')[0]])

    print(aucs)
    print(np.mean(aucs))

    print(f'TEST AUC: {round(auc, 3)}')

    # Evaluation with test-time augmentation
    if n_TTA > 0:
        set_seed(0)  # for fixed TTA
        test_dataset = ChestXRay14(data_dir=data_dir, split="test", augment=False, n_TTA=n_TTA)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // n_TTA * 2, shuffle=False, num_workers=4, worker_init_fn=val_worker_init_fn)

        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc=f'TEST EVAL')
            running_loss = 0.
            # auc = 0
            y_true, y_hat = [], []
            for i, batch in pbar:
                y = batch['y'].to(device)

                # Forward pass
                if meta_only:
                    meta = batch['meta'].to(device)

                    yhat = torch.sigmoid(model.forward(meta))
                elif fusion:
                    x = batch['x'].to(device)
                    meta = batch['meta'].to(device)

                    yhat = torch.stack([model.forward(x[..., i], meta) for i in range(n_TTA)], dim=0)
                    yhat = torch.sigmoid(yhat).mean(dim=0)
                else:
                    x = batch['x'].to(device)

                    yhat = torch.stack([model.forward(x[..., i]) for i in range(n_TTA)], dim=0)
                    yhat = torch.sigmoid(yhat).mean(dim=0)

                # Compute loss
                if ls == 0:
                    loss = torch.nn.BCELoss()(yhat, y)
                else:
                    loss = torch.nn.BCELoss()(yhat, y * (1-ls) + 0.5 * ls)

                running_loss += loss.item()

                y_true.append(y.detach().cpu().numpy())
                y_hat.append(yhat.detach().cpu().numpy())

                # if (i + 1) % 50 == 0:
                #     aucs = roc_auc_score(np.concatenate(y_true), np.concatenate(y_hat), average=None)
                #     auc = np.mean(aucs[np.where(test_dataset.CLASSES != 'No Finding')[0]])

                pbar.set_postfix({'test_loss': running_loss / (i + 1), 'test_auc': auc})

        y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

        aucs_tta = roc_auc_score(y_true, y_hat, average=None)
        auc_tta = np.mean(aucs_tta[np.where(np.array(test_dataset.CLASSES) != 'No Finding')[0]])
        print(f'TEST AUC (x{n_TTA} TTA): {round(auc_tta, 3)}')

    # Collect and save true and predicted disease labels for test set
    pred_df = pd.DataFrame(y_hat, columns=test_dataset.CLASSES)
    true_df = pd.DataFrame(y_true, columns=test_dataset.CLASSES)

    pred_df.to_csv(os.path.join(model_dir, 'test_pred.csv'), index=False)
    true_df.to_csv(os.path.join(model_dir, 'test_true.csv'), index=False)

    # Plot loss curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'loss'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'loss.png'), dpi=300, bbox_inches='tight')

    # Plot AUC curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'mean_auc'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'mean_auc'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'auc.png'), dpi=300, bbox_inches='tight')
        
    # Create summary text file describing final performance
    summary = f'Mean AUC: {round(auc, 3)}\n'
    if n_TTA > 0:
        summary += f'Mean AUC (x{n_TTA} TTA): {round(auc_tta, 3)}\n\n'
    for i in range(len(test_dataset.CLASSES)):
        summary += f'{test_dataset.CLASSES[i]}:| {round(aucs[i], 3)}'
        if n_TTA > 0:
            summary += f' | {round(aucs_tta[i], 3)} |'
        summary += '\n'
    f = open(os.path.join(model_dir, 'summary.txt'), 'w')
    f.write(summary)
    f.close()