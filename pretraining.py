import argparse
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import hico.builder
from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining_set_pointcloud, get_pretraining_set_intra
from losses import MultiPositiveInfoNCE

from torchsummary import summary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


parser = argparse.ArgumentParser(description='PyTorch NTU-60 Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=101, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[351], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=47, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--checkpoint-path', default='./checkpoints', type=str,
                    help="/root/autodl-tmp/hico-point-checkpoints or ./checkpoints")
parser.add_argument('--skeleton-representation', default='joint', type=str,
                    help='input skeleton-representation  for self supervised training (joint or motion or bone)')
parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120 or pointcloud)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='traiining protocol cross_view/cross_subject/cross_setup')

# hico specific configs:
parser.add_argument('--hico-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--hico-k', default=2048, type=int,
                    help='queue size; number of negative keys (default: 16384)')
parser.add_argument('--hico-m', default=0.999, type=float,
                    help='momentum of updating key encoder (default: 0.999)')
parser.add_argument('--hico-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
log_file = './' + str(int(time.time())) + '.txt'

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # pretraining dataset and protocol
    from options import options_pretraining as options 
    if args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':  # default
        opts = options.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
    elif args.pre_dataset == 'pku_part1' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part1_cross_subject()
    elif args.pre_dataset == 'pku_part2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part2_cross_subject()
    elif args.pre_dataset == 'pointcloud':
        opts = options.opts_pointcloud()

    opts.train_feeder_args['input_representation'] = args.skeleton_representation

    # create model
    print("=> creating model")

    model = hico.builder.HiCo(opts.encoder_args, args.hico_dim, args.hico_k, args.hico_m, args.hico_t, args.pre_dataset)
    print("options", opts.train_feeder_args)
    print(model)
    
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    
    # single gpu training
    model = model.cuda()

    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = MultiPositiveInfoNCE().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                #checkpoint = torch.load(args.resume, map_location=loc)
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading code
    if args.pre_dataset == "pointcloud":
        train_dataset = get_pretraining_set_pointcloud(opts)
    else:
        train_dataset = get_pretraining_set_intra(opts)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    writer = SummaryWriter(args.checkpoint_path)

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss, acc1 = train(train_loader, model, criterion1, criterion2, optimizer, epoch, args)
        writer.add_scalar('train_loss', loss.avg, global_step=epoch)
        writer.add_scalar('acc', acc1.avg, global_step=epoch)

        if epoch % 50 == 0:
                  save_checkpoint({
                      'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
                  }, is_best=False, filename=args.checkpoint_path+'/checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion1, criterion2, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1,],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch,optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (qc_input, qp_input, kc_input, kp_input) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        qc_input = qc_input.float().cuda(non_blocking=True)
        qp_input = qp_input.float().cuda(non_blocking=True)
        kc_input = kc_input.float().cuda(non_blocking=True)
        kp_input = kp_input.float().cuda(non_blocking=True)
        # print(qc_input.shape)
        # compute output
        output1, output2, output3, output4, output5, target1, target2, target3,  target4, target5, \
            = model(qc_input, qp_input, kc_input, kp_input)
        batch_size = output5.size(0)

        # compute hierarchical contrast loss
        
        # clip&part level
        loss1 = criterion2(output1, target1) + criterion2(output2, target2)
        # domain level
        loss2 = criterion1(output3, target3) + criterion1(output4, target4)
        # instance level
        loss3 = criterion1(output5, target5)

        loss = 1. * loss1 + 1. * loss2 + 1. * loss3
        losses.update(loss.item(), batch_size)
 
        # measure accuracy of model m1 and m2 individually
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, _ = accuracy(output5, target5, topk=(1, 5))
        top1.update(acc1[0], batch_size)

        # print("input output size",output.size(),images[0].size(),half_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
       
    return losses, top1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_path='./'):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)
        with open(log_file, 'a') as f:
            f.writelines('\t'.join(entries)+'\n')


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
