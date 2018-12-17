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
from make_vocab import Vocab
import time
import matplotlib.pyplot as plt
import os


class Args(object):
    def __init__(self, epochs=10, lr=0.01,
                optimizer='sgd', momentum=0.5, seed=1, model="ConvNet", conv_level = "word", cuda = True):
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.momentum = momentum
        self.seed = seed
        self.model = model
        self.conv_level = conv_level
        self.cuda = cuda and torch.cuda.is_available()

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')


class ConvNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, windows):
        super(ConvNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, 5, (window, embed_dim)) for window in windows])
        self.fc1 = nn.Linear(5*len(windows), 4)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.windows = windows

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(0) # add a batch dimension
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = F.dropout(x, training = self.training)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(LSTMNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 4)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.h0 = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.c0 = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1) # add a batch dimension
        output, (hn, cn) = self.lstm(x, (self.h0, self.c0))
        x = F.dropout(hn, training = self.training)
        x = self.fc1(x)
        x = x.squeeze(0)
        return F.log_softmax(x, dim = 1)

class LSTMAttentionNet(nn.Module):
	def __init__(self, vocab_size, embed_dim):
		super(LSTMAttentionNet, self).__init__()
		self.embed = nn.Embedding(vocab_size, embed_dim)
		self.lstm = nn.LSTM(embed_dim, embed_dim)
		self.att_fc1 = nn.Linear(embed_dim, embed_dim)
		self.att_fc2 = nn.Linear(embed_dim, 1)
		self.fc1 = nn.Linear(embed_dim, 4)
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.h0 = nn.Parameter(torch.randn(1, 1, embed_dim))
		self.c0 = nn.Parameter(torch.randn(1, 1, embed_dim))

	def forward(self, x):
		x = self.embed(x)
		x = x.unsqueeze(1)
		output, (hn, cn) = self.lstm(x, (self.h0, self.c0))
		alpha = torch.tanh(self.att_fc1(output))
		#print(alpha)
		alpha = self.att_fc2(alpha).squeeze(1)
		#print(alpha)
		alpha = F.softmax(alpha.t(), dim = 1)
		#print(alpha.size())
		c = torch.mm(alpha, output.squeeze(1)) # 1 by embed_dim
		c = self.fc1(c)
		return F.log_softmax(c, dim = 1)



def train(args, model, optimizer, train_data, epoch, total_minibatch_count, train_losses, train_accs):
    
    model.train()
    correct_count, total_loss, total_acc = 0., 0., 0.
    progress_bar = tqdm.tqdm(train_data, desc='Training')
    data_size = len(train_data)

    for idx, (data, target) in enumerate(progress_bar):
        if len(data) < 3:
            continue
        data = torch.LongTensor(data)
        target = torch.LongTensor([target])
        # print(data)
        # print(target)
        if args.cuda:
        	data, target = data.cuda(), target.cuda()
        
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        # Forward prediction step
        output = model(data)
        loss = F.nll_loss(output, target)

        # Backprop step
        loss.backward()
        optimizer.step()

        pred = output.data.max(1)[1]
        #print(output)
        #print(pred)

        if target == pred:
            correct_count += 1

        total_loss += loss.data

        progress_bar.set_description(
            'Epoch: {} loss: {:.4f}, acc: {:.2f}'.format(
                epoch, total_loss / (idx + 1), correct_count * 1.0 / (idx + 1)
            )
        )

        if idx % 10000 == 100:
            train_losses.append(total_loss / (idx + 1))
            train_accs.append(correct_count * 1.0 / (idx + 1))

        total_minibatch_count += 1

    return total_minibatch_count


def test(args, model, test_data, epoch, total_minibatch_count, dev_losses, dev_accs):
    
    model.eval()
    test_loss, correct = 0., 0.
    progress_bar = tqdm.tqdm(test_data, desc='Validation')
    with torch.no_grad():
        for idx, (data, target) in enumerate(progress_bar):
            if len(data) < 3:
                continue
            data = torch.LongTensor(data)
            target = torch.LongTensor([target])
            if args.cuda:
            	data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # print(data)
            # print(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').data
            pred = output.data.max(1)[1]
            if target == pred:
                correct += 1

            progress_bar.set_description(
                'Test Epoch: {} loss: {:.4f}, acc: {:.2f}'.format(
                    epoch, test_loss / (idx + 1), correct * 1.0 / (idx + 1)
                )
            )

    test_loss /= len(test_data)
    acc = correct / len(test_data)
    dev_losses.append(test_loss)
    dev_accs.append(acc)

    # progress_bar.clear()
    '''
    print('Epoch: %d validation test results - Average val_loss: %.4f, val_acc: %d/%d (%.2f)%%)' % (
            epoch, test_loss, correct, len(test_data),
            100. * correct / len(test_data)))
    '''

    return acc




def run_experiment(args):

    if args.conv_level == 'word':
        data = pickle.load(open('data_input.pkl', 'rb'))
        vocab_size = 57634
    elif args.conv_level == 'char':
        data = pickle.load(open('char_data_input.pkl', 'rb'))
        vocab_size = 176  

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]
    epochs_to_run = args.epochs
    torch.manual_seed(args.seed)
    if args.cuda:
    	torch.cuda.manual_seed(args.seed)

    total_minibatch_count = 0
    if args.model == 'ConvNet':
        model = ConvNet(vocab_size, 10, [3])
    elif args.model == 'LSTMNet':
        model = LSTMNet(vocab_size, 10)
    elif args.model == 'LSTMAttentionNet':
    	model = LSTMAttentionNet(vocab_size, 10)
    else:
        raise ValueError('Unsupported model: ' + args.model)

    if args.cuda:
    	model.cuda()

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
    best_dev_acc = 0
    train_losses, train_accs = [], []
    dev_losses, dev_accs = [], []
    test_losses, test_accs = [], []

    # dev_acc = test(model, dev_data, 0, total_minibatch_count, dev_losses, dev_accs)
    # time.sleep(0.5)
    for epoch in range(1, epochs_to_run + 1):
        total_minibatch_count = train(args, model, optimizer, train_data, epoch, total_minibatch_count, train_losses, train_accs)
        dev_acc = test(args, model, dev_data, epoch, total_minibatch_count, dev_losses, dev_accs)
        if dev_acc > best_dev_acc:
        	torch.save(model.state_dict(), os.path.join(os.getcwd() + '/' + args.model + args.conv_level))

    fig, axes = plt.subplots(1,4, figsize=(16,4))
    # plot the losses and acc
    plt.title(args.model)
    axes[0].plot(train_losses)
    axes[0].set_title("Loss")
    axes[1].plot(train_accs)
    axes[1].set_title("Acc")
    axes[2].plot(dev_losses)
    axes[2].set_title("Val loss")
    axes[3].plot(dev_accs)
    axes[3].set_title("Val Acc")
    plt.tight_layout()
    plt.show()

    # final test
    test_acc = test(args, model, test_data, epochs_to_run, total_minibatch_count, test_losses, test_accs)
    print("final test accuracy is %.2f%%." % (test_acc * 100))


run_experiment(Args(model='ConvNet', conv_level='char', epochs=5))
