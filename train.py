# reference: https://github.com/pytorch/examples/tree/42e5b996718797e45c46a25c55b031e6768f8440/imagenet
import argparse
import os
import shutil
import time
import csv
from random import shuffle
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import setup_logger
from datasets import StanfordCarsDataset


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Stanford Cars Training')
parser.add_argument('--data', metavar='DIR',
                    required=True, help='path to dataset')
parser.add_argument('--logdir', default='logs',
                    metavar='DIR', help='path to log')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet152)')
parser.add_argument('--num-classes', default=196, type=int,
                    metavar='N',  help='number of output class')
parser.add_argument('-j', '--workers', default=10, type=int,
                    metavar='N',  help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=200, type=int,
                    metavar='N',  help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N',  help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float,  metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M',  help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4,
                    type=float,  metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',
                    action='store_true',  help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test',
                    action='store_true',  help='predict the class of test data')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model (pytorch official)')
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    datetime_now = datetime.now()
    logger = setup_logger('datetime{}'.format(datetime_now),
                          os.path.join(args.logdir,
                                       '{}_{}_log.txt'.format(args.arch, datetime.now())))
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.to(device)
    else:
        model = torch.nn.DataParallel(model).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    cudnn.benchmark = True

    # Data loading code
    if args.test:
        testdir = os.path.join(args.data, 'testing_data/testing_data')

        test_dataset = StanfordCarsDataset(testdir, os.listdir(testdir), False, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        with open(os.path.join(args.data, 'labels_seri_map.csv'), 'r') as file:
            cin = csv.reader(file)
            mapping = [v for (k, v) in cin]

        result = [('id', 'label')]
        with torch.no_grad():
            for i, (input, image_id) in enumerate(test_loader, start=1):
                input = input.to(device)
                output = model(input)
                preds = torch.argmax(output, dim=1).cpu().tolist()
                preds_text = [mapping[pred] for pred in preds]
                result += [res for res in zip(image_id, preds_text)]
                print("[{} / {}]".format(i, len(test_loader)))

        with open('result.csv', 'w') as file:
            cout = csv.writer(file)
            cout.writerows(result)

        return

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.evaluate:
        traindir = os.path.join(args.data, 'training_data/training_data')
        imglist = os.listdir(traindir)
        shuffle(imglist)

        val_dataset = StanfordCarsDataset(traindir, imglist[:3000], transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        validate(val_loader, model, criterion)
        return

    # training
    traindir = os.path.join(args.data, 'training_data/training_data')
    imglist = os.listdir(traindir)
    shuffle(imglist)
    nr_train = round(len(imglist)*0.8)

    train_dataset = StanfordCarsDataset(traindir, imglist[:nr_train], transform=transforms.Compose([
        transforms.RandomAffine(degrees=20),
        transforms.RandomPerspective(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_dataset = StanfordCarsDataset(traindir, imglist[nr_train:], transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader))

    for epoch in range(args.start_epoch, args.epochs):
        # change validation set
        if epoch % 10 == 0:
            shuffle(imglist)
            nr_train = round(len(imglist)*0.8)

            train_dataset = StanfordCarsDataset(traindir, imglist[:nr_train], transform=transforms.Compose([
                transforms.RandomAffine(degrees=15),
                transforms.RandomPerspective(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            val_dataset = StanfordCarsDataset(traindir, imglist[nr_train:], transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

        # train for one epoch
        train(train_loader, model, criterion, optimizer,
              epoch, logger, scheduler.get_last_lr())

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, logger)

        # adjust_learning_rate(optimizer, epoch)
        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger, lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader, start=1):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.squeeze().to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]  '
                        'Lr {3:.6f}  '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch, i, len(train_loader), lr[0], batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader, start=1):
        with torch.no_grad():
            input_var = input
            target_var = target.squeeze().to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]  '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n\n\n\n'.format(
                top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='model_saved/checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_saved/model_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t_().to(device)
    traget_expend = target.view(1, -1).expand_as(pred).to(device)
    correct = pred.eq(traget_expend)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
