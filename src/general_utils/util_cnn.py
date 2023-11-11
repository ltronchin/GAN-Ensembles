from tqdm import trange
import torch
import os
import torch.nn as nn
import time
import copy
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
import math

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

def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs, early_stopping, warmup_epoch, model_dir, device, n_samples=None, to_disk=True, transfer_learning_inception_v3=False):
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
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if transfer_learning_inception_v3 and phase == 'train':
                            outputs, _ = model(inputs.float())
                        else:
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
                epoch_acc = running_corrects.double().item() / len(data_loaders[phase].dataset)

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
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_best.pt"))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history

def evaluate(model, data_loader, device, idx_to_class):

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

    test_results = {k: correct_pred[k]/total_pred[k] for k in correct_pred.keys() & total_pred}
    acc, prec, rec, spec, f1, gmean = compute_metrics(labels_list, preds_list)
    test_results['accuracy'] = acc
    test_results['precision'] = prec
    test_results['recall'] = rec
    test_results['specificity'] = spec
    test_results['f1_score'] = f1
    test_results['g_mean'] = gmean

    return test_results

def compute_metrics(y_test_real, y_pred, aggregation='mean'):
    cm = confusion_matrix(y_test_real, y_pred)
    num_classes = cm.shape[0]

    accuracy = np.trace(cm) / float(np.sum(cm))

    if num_classes == 2:

        tp = cm[0, 0]
        fp = np.sum(cm[:, 0]) - tp
        fn = np.sum(cm[0, :]) - tp
        tn = np.sum(cm) - (fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        g_mean = np.sqrt(recall * specificity)

        return accuracy, precision, recall, specificity, f1, g_mean

    else:

        precision_list = []
        recall_list = []
        f1_score_list = []
        specificity_list = []
        g_mean_list = []

        for j in range(num_classes):

            tp = cm[j, j]
            fp = np.sum(cm[:, j]) - tp
            fn = np.sum(cm[j, :]) - tp
            tn = np.sum(cm) - (fp + fn + tp)

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            g_mean = np.sqrt(recall * specificity)

            precision_list.append(precision)
            recall_list.append(recall)
            specificity_list.append(specificity)
            f1_score_list.append(f1)
            g_mean_list.append(g_mean)

        if aggregation == 'mean':
            precision = np.mean(precision_list)
            recall = np.mean(recall_list)
            specificity = np.mean(specificity_list)
            f1_score = np.mean(f1_score_list)
            g_mean = np.mean(g_mean_list)
        else:
            raise NotImplementedError

        return accuracy, precision, recall, specificity, f1_score, g_mean

def plot_training(history, plot_training_dir, plot_name_loss='Loss', plot_name_acc='Acc'):

    # Training results Loss function
    if 'train_loss' in history.columns and 'val_loss' in history.columns:
        plt.figure()
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
        plt.figure()
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