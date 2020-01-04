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
from models.zip import ZipNet, WideResNet, NoiseZipNet, BlurZipCNN
import json
from PIL import Image
import attacks
from temp.canny_net import Canny_CNN

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='zip_wrn',
                    choices=['zip_wrn', 'wrn'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
# WRN Architecture
parser.add_argument('--layers', default=28, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./logs/cifar10_warm', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--dataroot', default='.', type=str)
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

class ThreeNets(nn.Module):
    def __init__(self, net1, net2, net3):
        super(ThreeNets, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3

    def forward(self, x):
        return self.net3(self.net2(self.net1(x)) * 2 - 1)

torch.manual_seed(1)
np.random.seed(1)

# # mean and standard deviation of channels of CIFAR-10 images
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor()])
test_transform = trn.Compose([trn.ToTensor()])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.dataroot, train=True, transform=train_transform, download=False)
    test_data = dset.CIFAR10(args.dataroot, train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100(args.dataroot, train=True, transform=train_transform, download=False)
    test_data = dset.CIFAR100(args.dataroot, train=False, transform=test_transform)
    num_classes = 100


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=torch.cuda.is_available())

# Create model
net_zip = BlurZipCNN()
net = WideResNet(args.layers, 10, args.widen_factor, dropRate=args.droprate)

start_epoch = 0

if args.ngpu > 0:
    net_zip = torch.nn.DataParallel(net_zip, device_ids=list(range(args.ngpu)))
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

# Restore model if desired
if args.load != '':
    for i in range(300 - 1, -1, -1):
        model_zip_name = os.path.join(args.load, args.dataset + args.model +
                                  '_zip_baseline_epoch_' + str(i) + '.pt')
        model_name = os.path.join(args.load, args.dataset + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net_zip.load_state_dict(torch.load(model_zip_name))
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

# net.module.fc = nn.Linear(640, num_classes)

#if args.ngpu > 1:
#    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

canny_net = Canny_CNN()

if args.ngpu > 0:
    net_zip.cuda()
    canny_net.cuda()
    net.cuda()
    torch.cuda.manual_seed(1)


cudnn.benchmark = True  # fire on all cylinders

optimizer_zip = torch.optim.SGD(
    net_zip.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)
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
# adversary = attacks.PGD(epsilon=8./255, num_steps=10, step_size=2./255).cuda()

# /////////////// Training ///////////////


def train():
    net_zip.train()  # enter train mode
    net.train()  # enter train mode
    loss_avg = 0.0
    for bx, by in train_loader:
        bx, by = bx.cuda(), by.cuda()

        # adv_bx = adversary(net, bx, by)

        # forward
        zip_map = net_zip(bx * 2 - 1)
        edge = canny_net(zip_map)
        logits = net(edge * 2 - 1)

        # backward
        # scheduler.step()
        optimizer.zero_grad()
        optimizer_zip.zero_grad()
        loss = F.cross_entropy(logits, by)
        loss.backward()
        optimizer.step()
        optimizer_zip.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg


# test function
def test():
    net_zip.eval()
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            zip_map = net_zip(data * 2 - 1)
            edge = canny_net(zip_map)
            output = net(edge * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

def vis_test():

    if not os.path.exists(args.save + "/edge_sample/"):
        os.makedirs(args.save + "/edge_sample/")

    net_zip.eval()
    net.eval()
    three_nets = ThreeNets(net_zip, canny_net, net).cuda()
    adversary = attacks.PGD(epsilon=8. / 255, num_steps=20, step_size=2. / 255).cuda()
    loss_avg = 0.0
    correct = 0
    adv_loss_avg = 0.0
    adv_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary(three_nets, data, target)

            # forward
            output = three_nets(data * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            # forward
            adv_output = three_nets(adv_data * 2 - 1)
            adv_loss = F.cross_entropy(adv_output, target)

            # accuracy
            adv_pred = adv_output.data.max(1)[1]
            adv_correct += adv_pred.eq(target.data).sum().item()

            # test loss average
            adv_loss_avg += float(adv_loss.data)

            # zip_map = net_zip(data * 2 - 1)
            # adv_zip_map = net_zip(adv_data * 2 - 1)
            # for single in range(16):
            #     image_tensor = zip_map[single].repeat(3,1,1)
            #     image_numpy = image_tensor.cpu().float().numpy()
            #     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + ".png")
            #
            #     image_tensor = adv_zip_map[single].repeat(3, 1, 1)
            #     image_numpy = image_tensor.cpu().float().numpy()
            #     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_adv.png")

            zip_map = canny_net(net_zip(data * 2 - 1))
            adv_zip_map = canny_net(net_zip(adv_data * 2 - 1))
            for single in range(16):
                image_tensor = zip_map[single].repeat(3,1,1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + ".png")

                image_tensor = adv_zip_map[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_adv.png")

                image_numpy = data[single].cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_ori.png")

            break

    state['test_loss'] = loss_avg
    state['test_accuracy'] = correct / 128
    state['adv_test_loss'] = adv_loss_avg
    state['adv_test_accuracy'] = adv_correct / 128


# overall_test function
def test_in_testset():
    net_zip.eval()
    net.eval()
    three_nets = ThreeNets(net_zip, canny_net, net).cuda()
    adversary = attacks.PGD(epsilon=8. / 255, num_steps=20, step_size=2. / 255).cuda()
    loss_avg = 0.0
    correct = 0
    adv_loss_avg = 0.0
    adv_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary(three_nets, data, target)

            # forward
            output = three_nets(data * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            # forward
            adv_output = three_nets(adv_data * 2 - 1)
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



if args.test:
    vis_test()
    # test_in_testset()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + args.model +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

epoch_step = json.loads(args.epoch_step)

# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    if epoch in epoch_step:
        lr = optimizer.param_groups[0]['lr'] * args.lr_decay_ratio
        optimizer_zip = torch.optim.SGD(
            net_zip.parameters(), lr, momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)
        optimizer = torch.optim.SGD(
            net.parameters(), lr, momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)

    begin_epoch = time.time()

    train()
    test()

    # Save model
    if epoch % 10 == 9:
        torch.save(net_zip.state_dict(),
                   os.path.join(args.save, args.dataset + args.model +
                                '_zip_baseline_epoch_' + str(epoch) + '.pt'))
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + args.model +
                                '_baseline_epoch_' + str(epoch) + '.pt'))

    # # Let us not waste space and delete the previous model
    # prev_path = os.path.join(args.save, args.dataset + args.model +
    #                          '_baseline_epoch_' + str(epoch - 1) + '.pt')
    # if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + args.model +
                                      '_baseline_training_results.csv'), 'a') as f:
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
