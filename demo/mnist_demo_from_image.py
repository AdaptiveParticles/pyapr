import argparse
from torchvision import datasets

import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pyapr
import pyapr.nn


def mnist_apr_loader(train=True, batch_size=32, shuffle=True, transform=None):

    data = datasets.MNIST('../data', train=train, download=True, transform=transform)

    pars = pyapr.APRParameters()

    pars.Ip_th = 5
    pars.sigma_th = 10
    pars.sigma_th_max = 2
    pars.gradient_smoothing = 1
    pars.rel_error = 0.1
    pars.auto_parameters = False

    conv = pyapr.converter.FloatConverter()
    conv.set_parameters(pars)
    conv.set_verbose(False)

    part_list = []
    apr_list = []
    targets = torch.zeros(len(data), dtype=torch.long)

    i = 0
    for dd in data:
        x, y = dd
        x = np.array(x).astype(np.float32)

        targets[i] = y

        apr_list.append(pyapr.APR())
        conv.get_apr(apr_list[i], x)
        apr_list[i].init_tree()
        part_list.append(pyapr.FloatParticles())
        part_list[i].sample_image(apr_list[i], x)

        i += 1

    apr_data = APRDataset(apr_list, part_list, targets)

    return DataLoader(apr_data, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)


class APRDataset(Dataset):
    def __init__(self, apr_list, part_list, labels):
        self.apr_list = apr_list
        self.part_list = part_list
        self.labels = labels

    def __getitem__(self, index):
        apr = self.apr_list[index]
        part = self.part_list[index]
        target = self.labels[index]

        return apr, part, target

    def __len__(self):
        return len(self.apr_list)


def custom_collate_fn(data):

    x, y, z = zip(*data)

    apr_arr = np.stack(x)
    targets = torch.stack(z)

    return apr_arr, y, targets


def train(model, train_loader, loss_fn, optimizer, epoch, log_interval=10):
    model.train()

    loss_list = []
    acc_list = []

    loss_log = []
    acc_log = []

    for batch_idx, (aprs, parts, target) in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            output = model(aprs, parts)
            loss = loss_fn(output, target)
            loss_list.append(float(loss.item()) / len(aprs))
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc_list.append(float(correct) / len(target))
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0 and batch_idx is not 0:
                avg_loss = np.mean(np.array(loss_list))
                avg_acc = np.mean(np.array(acc_list))
                loss_list = []
                acc_list = []
                loss_log.append(avg_loss)
                acc_log.append(avg_acc)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {}'.format(
                    epoch, batch_idx * len(aprs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), avg_loss, avg_acc)
                )

    return acc_log, loss_log


def test(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return float(correct) / len(test_loader.dataset), test_loss


def main(args):
    device = torch.device("cpu")

    train_loader = mnist_apr_loader(train=True, batch_size=args.batch_size, shuffle=True)
    test_loader = mnist_apr_loader(train=False, batch_size=args.batch_size, shuffle=False)

    model = APRConvNet().to(device)

    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-4, patience=5)

    nepoch = args.nepoch

    for epoch in range(start_epoch + 1, start_epoch + nepoch + 1):

        start = time.time()
        train(model, train_loader, criterion, optimizer, epoch, log_interval=10)
        acc, ts_loss = test(model, test_loader, criterion)
        end = time.time()
        print('epoch time: {}'.format(end - start))

        scheduler.step(ts_loss)


class APRConvNet(nn.Module):
    def __init__(self):
        super(APRConvNet, self).__init__()

        self.input_layer = pyapr.nn.APRInputLayer()

        self.conv1 = pyapr.nn.APRConv(in_channels=1, out_channels=16, kernel_size=3, nstencils=2)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.pool1 = pyapr.nn.APRMaxPool()

        self.conv2 = pyapr.nn.APRConv(in_channels=16, out_channels=32, kernel_size=3, nstencils=1)
        self.conv2_bn = nn.BatchNorm1d(32)
        self.pool2 = pyapr.nn.APRMaxPool()

        self.fc1 = pyapr.nn.APRConv(in_channels=32, out_channels=64, kernel_size=1, nstencils=1)
        self.fc2 = pyapr.nn.APRConv(in_channels=64, out_channels=10, kernel_size=1, nstencils=1)

        self.globavg = nn.AdaptiveAvgPool1d(1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias.size:
                    nn.init.uniform_(m.bias, 0, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, apr_arr, parts_arr):

        x, level_deltas = self.input_layer(apr_arr, parts_arr)

        x = self.conv1(x, apr_arr, level_deltas)
        x = self.pool1(x, apr_arr, level_deltas)
        x = F.relu(x)
        x = self.conv1_bn(x)

        x = self.conv2(x, apr_arr, level_deltas)
        #x = self.pool2(x, apr_arr, level_deltas)  # FIXME
        x = F.relu(x)
        x = self.conv2_bn(x)

        x = self.fc1(x, apr_arr, level_deltas)
        x = self.fc2(x, apr_arr, level_deltas)

        x = self.globavg(x).view(-1, 10)

        return F.softmax(x, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()

    main(args)
