import warnings
import os
import numpy as np

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from tqdm import tqdm
from torch.utils import data
import argparse
import shutil
from sklearn.model_selection import train_test_split


def create_dir(path):
    if os.path.exists(path) == False:
        os.mkdir(path)


class VIT(nn.Module):
    def __init__(self, pretrained=True):
        super(VIT, self).__init__()
        self.model = models.vit_b_32(pretrained=pretrained)
        self.classify = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.classify(x)
        return x


class CIFARLoader(data.Dataset):
    def __init__(self, image, label, transform=None):

        self.img_name, self.labels = image, label
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        self.img = self.img_name[index]
        self.label = self.labels[index]

        if self.transform:
            self.img = self.transform(self.img)

        return self.img, self.label


def predict(test_loader, model, device):

    global y_pred_All_val_batch
    global y_true_All_val_batch
    y_pred = []
    model.eval()
    with torch.no_grad():
        curloader = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, (x_test, y_test) in curloader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_pred_test = model(x_test)
            y_pred += (torch.max(y_pred_test, 1)
                       [1].cpu().numpy().tolist())
            y_true_All_val_batch += (y_test.cpu().numpy().tolist())
            y_pred_All_val_batch += (torch.max(y_pred_test, 1)
                                     [1].cpu().numpy().tolist())
    return y_pred


def plot_confusion_matrix_figure(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_confusion_matrix(y_pred, y_true):

    target_names = list(range(10))
    plt.figure()
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_figure(
        cnf_matrix, classes=target_names, normalize=True, title='confusion matrix')
    plt.savefig('confusion_matrix.jpg', dpi=300)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test',
                    help='two options, train or test')
parser.add_argument('--train_model_name', type=str, default='VIT_CIFAR.rar',
                    help='model name to save the weight')
parser.add_argument('--test_model_name', type=str, default='BEST_VIT_CIFAR.rar',
                    help='model name to save the weight')
opt = parser.parse_args()

y_pred_All_val_batch = []
y_true_All_val_batch = []

image_size = 224
num_classes = 10
min_loss = 1
number_worker = 4
max_accuracy = 0
batch_size = 64
epochs = 10
lr = 2e-5

create_dir(os.path.abspath(os.path.dirname(__file__)) +
           "/checkpoint")
create_dir(os.path.abspath(os.path.dirname(__file__)) +
           "/history_csv")

train_filepath = os.path.abspath(os.path.dirname(__file__)) + \
    "/checkpoint/"+opt.train_model_name

train_filepath_csv = os.path.abspath(os.path.dirname(
    __file__))+"/history_csv/VIT_CIFAR.csv"

test_filepath = os.path.abspath(os.path.dirname(__file__)) + \
    "/checkpoint/"+opt.test_model_name

device = torch.device("cuda:0")


if opt.mode == "train":

    # Load data
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")

    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_train /= 255

    # It's a multi-class classification problem
    class_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                   'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    print(np.unique(y_train))

    # ![image](https://img-blog.csdnimg.cn/20190623084800880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lqcDE5ODcxMDEz,size_16,color_FFFFFF,t_70)

    # ## Data preprocess
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Resize((image_size, image_size)),
    ])

    y_train = y_train.reshape(y_train.shape[0],)

    train_dataset = CIFARLoader(
        x_train, y_train, transform=train_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=number_worker)

    warnings.filterwarnings('ignore')
    # Builde model

    model = VIT()
    print(model)

    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True

    model.cuda(0)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_batch = []
    accuracy_batch = []
    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for epoch in range(epochs):

        curloader = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (data, target) in curloader:

            model.train()
            data, target = data.to(device), target.to(device)

            y_pred = model(data)
            loss = criterion(y_pred, target)
            loss_batch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = target.shape[0]
            correct = (torch.max(y_pred, 1)[1] == target).sum().item()
            train_accuracy = correct / n
            accuracy_batch.append(train_accuracy)
            curloader.set_postfix(loss=loss.item(), accuracy=train_accuracy)

        train_accuracy = sum(accuracy_batch)/len(accuracy_batch)
        train_loss = sum(loss_batch)/len(loss_batch)

        print("\n epochs:", epoch, "Training loss:", train_loss, "Training Accuracy:",
              train_accuracy)
        loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        loss_batch = []
        accuracy_batch = []

        if train_accuracy > max_accuracy:
            max_accuracy = train_accuracy
            torch.save(model.state_dict(), train_filepath)

    df = pd.DataFrame(
        {"loss": loss_history, "train_accuracy_history": train_accuracy_history})

    df.to_csv(train_filepath_csv, encoding="utf-8-sig")

# testing mode -> input is dataloader, output is y_pred
elif opt.mode == "test":

    # ## DO NOT MODIFY CODE BELOW!
    # please screen shot your results and post it on your report
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    x_test = x_test.astype('float32')
    x_test /= 255

    y_test = y_test.reshape(y_test.shape[0],)

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Resize((image_size, image_size))
    ])

    test_dataset = CIFARLoader(
        x_test, y_test, transform=test_transform)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=number_worker)

    model = VIT()
    model.cuda(0)

    model.load_state_dict(torch.load(test_filepath))
    y_pred = predict(test_loader, model, device)
    y_pred = np.array(y_pred)

    assert y_pred.shape == (10000,)

    y_test = np.load("y_test.npy")
    print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))

    plot_confusion_matrix(y_pred_All_val_batch, y_true_All_val_batch)
