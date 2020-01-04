# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.allconv import AllConvNet
from models.wrn import WideResNet
import json
import attacks
from trades import trades_loss, kld_loss

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=60, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--epoch_step', default='[30,40,50]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
# WRN Architecture
parser.add_argument('--layers', default=28, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./logs/cifar10_semidir', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./logs/cifar10_warm10k', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--supdata', default='/mnt/lustrenew/yanglei/haonan/data/cinic_imgnet5000', type=str)
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

# # mean and standard deviation of channels of CIFAR-10 images
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor()])
test_transform = trn.Compose([trn.ToTensor()])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.dataroot, train=True, transform=train_transform)
    test_data = dset.CIFAR10(args.dataroot, train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100(args.dataroot, train=True, transform=train_transform)
    test_data = dset.CIFAR100(args.dataroot, train=False, transform=test_transform)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size//2, shuffle=True,
    num_workers=args.prefetch, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=torch.cuda.is_available())
sup_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(args.dataroot+'/cinic_imgnet5000/train', transform=train_transform), batch_size=args.batch_size//2, shuffle=True,
    num_workers=args.prefetch, pin_memory=torch.cuda.is_available())

# Create model
if args.model == 'allconv':
    net = AllConvNet(1000)
else:
    net = WideResNet(args.layers, 10, args.widen_factor, dropRate=args.droprate)

start_epoch = 0

if args.ngpu > 0:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

# Restore model if desired
if args.load != '':
    for i in range(300 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + args.model +
                                  '_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

# net.module.fc = nn.Linear(640, num_classes)

#if args.ngpu > 1:
#    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


# def cosine_annealing(step, total_steps, lr_max, lr_min):
#     return lr_min + (lr_max - lr_min) * 0.5 * (
#             1 + np.cos(step / total_steps * np.pi))
#
#
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer,
#     lr_lambda=lambda step: cosine_annealing(
#         step,
#         args.epochs * len(train_loader),
#         1,  # since lr_lambda computes multiplicative factor
#         1e-6 / args.learning_rate))  # originally 1e-6
#
#
adversary = attacks.PGD(epsilon=8./255, num_steps=7, step_size=2./255).cuda()

# /////////////// Training ///////////////


def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    iter_sup = iter(sup_loader)
    for bx, by in train_loader:
        bx, by = bx.cuda(), by.cuda()

        sup_bx, sup_by = iter_sup.next()
        sup_bx = sup_bx.cuda()

        net.eval()
        adv_bx = adversary(net, bx, by)
        domainb_loss = kld_loss(model=net,
                           x_natural=sup_bx,
                           optimizer=optimizer,
                           beta=3.0)
        net.train()

        logits = net(adv_bx * 2 - 1)
        domaina_loss = F.cross_entropy(logits, by)

        optimizer.zero_grad()
        loss = (domaina_loss + domainb_loss) / 2
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary(net, data, target)

            # forward
            output = net(adv_data * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

# overall_test function
def test_in_testset():
    net.eval()
    loss_avg = 0.0
    correct = 0
    adv_loss_avg = 0.0
    adv_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary(net, data, target)

            # forward
            output = net(data * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            # forward
            adv_output = net(adv_data * 2 - 1)
            adv_loss = F.cross_entropy(adv_output, target)

            # accuracy
            adv_pred = adv_output.data.max(1)[1]
            adv_correct += adv_pred.eq(target.data).sum().item()

            # test loss average
            adv_loss_avg += float(adv_loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)
    state['adv_test_loss'] = adv_loss_avg / len(test_loader)
    state['adv_test_accuracy'] = adv_correct / len(test_loader.dataset)

def test_in_trainset():
    train_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(args.dataroot + '/cinic-10-cifar/train10k', transform=test_transform),
        batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.prefetch, pin_memory=torch.cuda.is_available())
    net.eval()
    loss_avg = 0.0
    correct = 0
    adv_loss_avg = 0.0
    adv_correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary(net, data, target)

            # forward
            output = net(data * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            # forward
            adv_output = net(adv_data * 2 - 1)
            adv_loss = F.cross_entropy(adv_output, target)

            # accuracy
            adv_pred = adv_output.data.max(1)[1]
            adv_correct += adv_pred.eq(target.data).sum().item()

            # test loss average
            adv_loss_avg += float(adv_loss.data)

    state['train_loss'] = loss_avg / len(train_loader)
    state['train_accuracy'] = correct / len(train_loader.dataset)
    state['adv_train_loss'] = adv_loss_avg / len(train_loader)
    state['adv_train_accuracy'] = adv_correct / len(train_loader.dataset)

# def robust_test():
#     net.eval()
#     loss_avg = 0.0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.cuda(), target.cuda()
#
#             data = adversary(net, data, target)
#
#             # forward
#             output = net(data * 2 - 1)
#             loss = F.cross_entropy(output, target)
#
#             # accuracy
#             pred = output.data.max(1)[1]
#             correct += pred.eq(target.data).sum().item()
#
#             # test loss average
#             loss_avg += float(loss.data)
#
#     state['robust_test_loss'] = loss_avg / len(test_loader)
#     state['robust_test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + args.model +
                                  '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

epoch_step = json.loads(args.epoch_step)

# Main loop
best_test_accuracy = 0
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    if epoch in epoch_step:
        lr = optimizer.param_groups[0]['lr'] * args.lr_decay_ratio
        optimizer = torch.optim.SGD(
            net.parameters(), lr, momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)
        print('At epoch: ' + str(epoch) + ', lr: ' + str(lr))

    train()
    test()

    # Save model
    if epoch % 10 == 9:
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + args.model +
                                '_epoch_' + str(epoch) + '.pt'))

    if state['test_accuracy'] > best_test_accuracy:
        best_test_accuracy = state['test_accuracy']
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + args.model +
                                '_best_epoch.pt'))


    # # Let us not waste space and delete the previous model
    # prev_path = os.path.join(args.save, args.dataset + args.model +
    #                          '_adv_epoch_' + str(epoch - 1) + '.pt')
    # if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + args.model +
                                      '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
