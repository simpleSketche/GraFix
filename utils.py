import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from dataset import *


# accuracy

def binary_classification_accracy(pred, y):
    pred = torch.round(pred)
    acc = torch.sum(pred == y) / torch.numel(y)
    return acc

def R2_score(pred, y):
    # print("in r2 score: y")
    # print(y)
    # print()
    y_bar = torch.sum(torch.mean(y))
    SS_tot = torch.sum((y - y_bar)**2)
    SS_res = torch.sum((y - pred)**2)
    R2 = 1 - SS_res / SS_tot
    return R2




def plot_learningCurve(accuracy_record, save_dir):
    train_acc, valid_acc = accuracy_record["train"], accuracy_record["valid"]    
    epochs = list(range(1, len(train_acc)+1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, label=f'best train: {np.max(train_acc):.4f}')
    plt.plot(epochs, valid_acc, label=f'best valid: {np.max(valid_acc):.4f}')
    plt.legend()
    plt.grid(alpha=.7)
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.ylim([-0.1, 1.05])
    plt.title("Accuracy (R2) for vertices repairment")
    plt.savefig(save_dir / "LearningCurve.png")
    plt.close()



def plot_lossCurve(loss_record, save_dir, title=None):
    train_loss, valid_loss = loss_record["train"], loss_record["valid"] 
    epochs = list(range(1, len(train_loss)+1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label=f'best train: {np.min(train_loss):.4f}')
    plt.plot(epochs, valid_loss, label=f'best valid: {np.min(valid_loss):.4f}')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(save_dir / "lossCurve.png")
    plt.close()


