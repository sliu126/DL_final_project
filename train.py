import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import tqdm
import io
import sys
import logging
import pickle


class Args(object):
    def __init__(self, epochs=10, lr=0.01,
                optimizer='sgd', momentum=0.5, seed=1):
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.momentum = momentum
        self.seed = seed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
args = Args()
data_input = pickle.load(open('data_input.pkl', 'rb'))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(100, 10)

    def forward(self):
        pass

def train(model, optimizer, train_data, epoch, total_minibatch_count, train_losses, train_accs):
    
    model.train()
    correct_count, total_loss, total_acc = 0., 0., 0.
    progress_bar = tqdm.tqdm(train_data, desc='Training')

    for idx, (data, target) in enumerate(progress_bar):
        data = torch.LongTensor(data)
        target = torch.LongTensor([target])
        # print(data)
        # print(target)
        '''
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        # Forward prediction step
        output = model(data)
        loss = F.nll_loss(output, target)

        # Backprop step
        loss.backward()
        optimizer.step()

        pred = output.data.max(1)[1]

        if target == pred:
            correct_count += 1

        total_loss += loss.data
        total_acc += accuracy.data

        progress_bar.set_description(
            'Epoch: {} loss: {:.4f}, acc: {:.2f}'.format(
                epoch, total_loss / (idx + 1), total_acc / (idx + 1)
            )
        )
        total_minibatch_count += 1
        '''

    return total_minibatch_count


def test(model, test_data, epoch, total_minibatch_count, dev_losses, dev_accs):
    
    model.eval()
    test_loss, correct = 0., 0.
    progress_bar = tqdm.tqdm(test_data, desc='Validation')
    with torch.no_grad():
        for data, target in progress_bar:
            data = torch.LongTensor(data)
            target = torch.LongTensor([target])
            data, target = Variable(data), Variable(target)
            # print(data)
            # print(target)
            '''
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').data
            pred = output.data.max(1)[1]
            if target == pred:
                correct += 1
            '''




def run_experiment(args, data):

    train_data = data["train"]
    dev_data = data["dev"]
    epochs_to_run = args.epochs

    total_minibatch_count = 0
    model = Net()

    # args.optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError('Unsupported optimizer: ' + args.optimizer)


    dev_acc = 0
    train_losses, train_accs = [], []
    dev_losses, dev_accs = [], []

    for epoch in range(1, epochs_to_run + 1):
        total_minibatch_count = train(model, optimizer, train_data, epoch, total_minibatch_count, train_losses, train_accs)
        dev_acc = test(model, dev_data, epoch, total_minibatch_count, dev_losses, dev_accs)


run_experiment(args, data_input)
