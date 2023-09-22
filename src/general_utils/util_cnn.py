from tqdm import trange
import torch
import os
import torch.nn as nn
import time
import copy
import numpy as np
from tqdm import tqdm
import pandas as pd

# Figure properties.
import seaborn as sns
import matplotlib.pyplot as plt
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=2, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
sns.set_context("paper")

def test(model, evaluator, data_loader, criterion, device, run, save_folder=None):
    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]

def train(model, train_loader, criterion, optimizer, device, batch_size, n_samples):
    total_loss = []

    model.train()
    for curr_iter, (inputs, targets) in enumerate(train_loader):
        if curr_iter == n_samples // batch_size:
            break

        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss

def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs, early_stopping, warmup_epoch, model_dir, device, class_real=None, n_samples=None, to_disk=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epochs_no_improve = 0
    best_epoch = 0
    early_stop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            current_samples = 0

            # Iterate over data.
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
                for inputs, labels in data_loaders[phase]:
                    if phase=='train' and n_samples is not None:
                        if current_samples > n_samples:
                            break

                    if phase == 'val' and class_real is not None:
                        labels = torch.full_like(labels.detach(), class_real)

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(inputs.shape[0])
                    current_samples += inputs.shape[0]

            if phase=='train' and n_samples is not None:
                epoch_loss = running_loss / n_samples
                epoch_acc = running_corrects.double() / n_samples
            else:
                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':

                if epoch > warmup_epoch:
                    if epoch_loss < best_loss:
                        best_epoch = epoch
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= early_stopping:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break
        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    if to_disk:
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_best_epoch_{best_epoch}.pt"))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history

def evaluate(dataset_name, model, data_loader, device):

    # Global and Class Accuracy
    if dataset_name == 'AIforCOVID':
        idx_to_class = data_loader.dataset.idx_to_class

    elif dataset_name == 'AERTS':
        raise NotImplementedError
    elif dataset_name == 'CLARO':
        raise NotImplementedError
    else:
        idx_to_class = data_loader.dataset.data.info['label']
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}


    classes_name = list(idx_to_class.values())
    correct_pred = {classname: 0 for classname in classes_name + ["all"]}
    total_pred = {classname: 0 for classname in classes_name + ["all"]}
    labels_list = []
    preds_list = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_list.append(labels.cpu().numpy())
            # Prediction
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            preds_list.append(preds.cpu().numpy())
            # global
            correct_pred['all'] += (preds == labels).sum().item()
            total_pred['all'] += labels.size(0)
            # class
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_pred[idx_to_class[label.item()]] += 1
                total_pred[idx_to_class[label.item()]] += 1

    labels_list = np.concatenate(labels_list)
    preds_list = np.concatenate(preds_list)

    # Accuracy
    test_results = {k: correct_pred[k]/total_pred[k] for k in correct_pred.keys() & total_pred}

    test_results = compute_metrics(test_results, labels_list, preds_list, keys=['recall', 'precision', 'f1_score', 'specificity', 'geometric_mean', 'auc'])

    return test_results

def compute_metrics(data, label, pred, keys=None):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    import math
    if keys is None:
        keys = ['acc', 'recall', 'precision', 'f1_score', 'specificity', 'geometric_mean', 'auc']

    # Conf matrix
    #    predicted
    #    TN     FP
    #
    #    FN     TP

    conf_mat = confusion_matrix(label, pred)
    tn, fp, fn, tp = conf_mat.ravel()

    if 'acc' in keys:
        data['acc'] = (tp + tn) / (tp + tn + fp + fn)
    if 'recall' in keys:
        data['recall'] = tp / (tp + fn)
    if 'precision' in keys:
        data['precision'] = tp / (tp + fp)
    if 'f1_score' in keys:
        data['f1_score'] = (2 * data['recall'] * data['precision']) /  (data['recall'] + data['precision'])
    if 'specificity' in keys:
        data['specificity'] = tn / (tn + fp)
    if 'geometric_mean' in keys:
        data['geometric_mean'] = math.sqrt(data['recall'] * data['specificity'])
    if 'auc' in keys:
        data['auc'] = roc_auc_score(label, pred)

    return data

def plot_training(history, plot_training_dir, plot_name_loss='Loss', plot_name_acc='Acc'):

    # Training results Loss function
    if 'train_loss' in history.columns and 'val_loss' in history.columns:
        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'val_loss']:
            plt.plot(history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')

        # Delete white space
        plt.tight_layout()
        plt.savefig(os.path.join(plot_training_dir, f"{plot_name_loss}.pdf"),  dpi=400, format='pdf')
        plt.show()

    # Training results Accuracy
    if 'train_acc' in history.columns and 'val_acc' in history.columns:
        plt.figure(figsize=(8, 6))
        for c in ['train_acc', 'val_acc']:
            plt.plot(100 * history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.title('Training and Validation Accuracy')

        # Delete white space
        plt.tight_layout()
        plt.savefig(os.path.join(plot_training_dir, f"{plot_name_acc}.pdf"),  dpi=400, format='pdf')
        plt.show()