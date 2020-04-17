import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


def _get_loss_acc(model, train_loader, valid_loader):
    """
    Get losses and validation accuracy
    """
    n_epochs = 2
    lr = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)
    loss_batch = []

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())

    correct = 0
    total = 0
    for data, target in valid_loader:
        output = model(data)
        _, predicted = max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()

    valid_acc = correct.item() / total

    return loss_batch, valid_acc

    def compare_init_weights(
            model_list,
            plot_title,
            train_loader,
            valid_loader,
            plot_n_batches=100):
        """
        Plot loss and print stats of weights using an example neural network
        """
        colors = ['r', 'b', 'g', 'c', 'y', 'k']
        label_accs = []
        label_loss = []

        assert len(model_list) <= len(colors), 'Too many initial weights to plot'

        for i, (model, label) in enumerate(model_list):
            loss, val_acc = _get_loss_acc(model, train_loader, valid_loader)

            plt.plot(loss[:plot_n_batches], colors[i], label=label)
            label_accs.append((label, val_acc))
            label_loss.append((label, loss[-1]))

        plt.title(plot_title)
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        print('After 2 Epochs:')
        print('Validation Accuracy')
        for label, val_acc in label_accs:
            print('  {:7.3f}% -- {}'.format(val_acc * 100, label))
        print('Training Loss')
        for label, loss in label_loss:
            print('  {:7.3f}  -- {}'.format(loss, label))


def hist_dist(title, distribution_tensor, hist_range=(-4, 4)):
    """
    Display histogram of values in a given distribution tensor
    """
    plt.title(title)
    plt.hist(distribution_tensor, np.linspace(*hist_range, num=len(distribution_tensor) / 2))
    plt.show()
