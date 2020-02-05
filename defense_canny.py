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
from models.zip import ZipNet, WideResNet, NoiseZipNet, Zip16Net
import json
from PIL import Image
import attacks
from temp.canny_net import Canny_Net
from temp.robust_canny import get_edge, vis_edge

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

parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--high_threshold', type=float, default=0.2)
parser.add_argument('--low_threshold', type=float, default=0.1)
parser.add_argument('--robust_threshold', type=float, default=0.2)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

class New_TwoNets(nn.Module):
    def __init__(self, net1, net2):
        super(New_TwoNets, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        return self.net2(2 * self.net1((x+1)/2.0) - 1)

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
net = WideResNet(args.layers, 10, args.widen_factor, dropRate=args.droprate)

start_epoch = 0

if args.ngpu > 0:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

# Restore model if desired
if args.load != '':
    for i in range(300 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
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

canny_net = Canny_Net(args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)

if args.ngpu > 0:
    net.cuda()
    canny_net.cuda()
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
# adversary = attacks.PGD(epsilon=8./255, num_steps=10, step_size=2./255).cuda()

# /////////////// Training ///////////////


def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for bx, by in train_loader:
        bx, by = bx.cuda(), by.cuda()

        # adv_bx = adversary(net, bx, by)

        # forward
        edge = get_edge(bx, args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)
        logits = net(edge * 2 - 1)

        # backward
        # scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, by)
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

            # forward
            edge = get_edge(data, args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)
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

    net.eval()
    two_nets = New_TwoNets(canny_net, net).cuda()
    adversary = attacks.PGD(epsilon=8. / 255, num_steps=20, step_size=2. / 255).cuda()
    loss_avg = 0.0
    correct = 0
    adv_loss_avg = 0.0
    adv_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary(two_nets, data, target)

            # forward
            edge = get_edge(data, args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)
            output = net(edge * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            # forward
            adv_edge = get_edge(adv_data, args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)
            adv_output = net(adv_edge * 2 - 1)
            adv_loss = F.cross_entropy(adv_output, target)

            # accuracy
            adv_pred = adv_output.data.max(1)[1]
            adv_correct += adv_pred.eq(target.data).sum().item()

            # test loss average
            adv_loss_avg += float(adv_loss.data)

            edge_map, vis1, vis2, vis3, vis4, vis5 = vis_edge(data, args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)
            adv_edge_map, adv_vis1, adv_vis2, adv_vis3, adv_vis4, adv_vis5 = vis_edge(adv_data, args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)
            for single in range(16):
                image_tensor = edge_map[single].repeat(3,1,1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + ".png")

                image_tensor = adv_edge_map[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_adv.png")

                image_tensor = vis1[single].repeat(3,1,1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis1.png")

                image_tensor = adv_vis1[single].repeat(3,1,1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis1_adv.png")

                image_tensor = vis2[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis2.png")

                image_tensor = adv_vis2[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis2_adv.png")

                image_tensor = vis3[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis3.png")

                image_tensor = adv_vis3[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis3_adv.png")

                image_tensor = vis4[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis4.png")

                image_tensor = adv_vis4[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis4_adv.png")

                image_tensor = vis5[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis5.png")

                image_tensor = adv_vis5[single].repeat(3, 1, 1)
                image_numpy = image_tensor.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_vis5_adv.png")

                image_numpy = data[single].cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_ori.png")

                image_numpy = adv_data[single].cpu().float().numpy()
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
                image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
                single_image = Image.fromarray(image_numpy)
                single_image.save(args.save + "/edge_sample/" + str(single) + "_adv_ori.png")

            # edge_map, vis1, vis2, vis3, vis4 = canny_net.vis_forward(data)
            # adv_edge_map, adv_vis1, adv_vis2, adv_vis3, adv_vis4 = canny_net.vis_forward(adv_data)
            # for single in range(16):
            #     image_tensor = edge_map[single].repeat(3,1,1)
            #     image_numpy = image_tensor.cpu().float().numpy()
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + ".png")
            #
            #     image_tensor = adv_edge_map[single].repeat(3, 1, 1)
            #     image_numpy = image_tensor.cpu().float().numpy()
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_adv.png")
            #
            #     image_tensor = vis1[single].repeat(3,1,1)
            #     image_numpy = image_tensor.cpu().float().numpy()
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_vis1.png")
            #
            #     image_tensor = adv_vis1[single].repeat(3, 1, 1)
            #     image_numpy = image_tensor.cpu().float().numpy()
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_vis1_adv.png")
            #
            #     image_tensor = vis2[single].repeat(3,1,1)
            #     image_numpy = image_tensor.cpu().float().numpy()
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_vis2.png")
            #
            #     image_tensor = adv_vis2[single].repeat(3, 1, 1)
            #     image_numpy = image_tensor.cpu().float().numpy()
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_vis2_adv.png")
            #
            #     image_numpy = vis3[single].repeat(3, 0)
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_vis3.png")
            #
            #     image_numpy = adv_vis3[single].repeat(3, 0)
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_vis3_adv.png")
            #
            #     image_numpy = vis4[single].repeat(3, 0)
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_vis4.png")
            #
            #     image_numpy = adv_vis4[single].repeat(3, 0)
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_vis4_adv.png")
            #
            #     image_numpy = data[single].cpu().float().numpy()
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_ori.png")
            #
            #     image_numpy = adv_data[single].cpu().float().numpy()
            #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            #     image_numpy = np.clip(image_numpy.astype(np.uint8), 0, 255)
            #     single_image = Image.fromarray(image_numpy)
            #     single_image.save(args.save + "/edge_sample/" + str(single) + "_adv_ori.png")

            break

    state['test_loss'] = loss_avg
    state['test_accuracy'] = correct / 128
    state['adv_test_loss'] = adv_loss_avg
    state['adv_test_accuracy'] = adv_correct / 128


# overall_test function
def test_in_testset():
    net.eval()
    two_nets = New_TwoNets(canny_net, net).cuda()
    adversary = attacks.PGD(epsilon=8. / 255, num_steps=20, step_size=2. / 255).cuda()
    loss_avg = 0.0
    correct = 0
    adv_loss_avg = 0.0
    adv_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary(two_nets, data, target)

            # forward
            edge = get_edge(data, args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)
            output = net(edge * 2 - 1)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            # forward
            adv_edge = get_edge(adv_data, args.sigma, args.high_threshold, args.low_threshold, args.robust_threshold)
            adv_output = net(adv_edge * 2 - 1)
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
    # vis_test()
    test_in_testset()
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
        optimizer = torch.optim.SGD(
            net.parameters(), lr, momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)

    begin_epoch = time.time()

    train()
    test_in_testset()
    # test()

    # Save model
    if epoch % 10 == 9:
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
