import warnings
from sklearn import metrics

from models.QDDA import QDDANet
from utils.AttentionLoss import AttentionLoss

warnings.filterwarnings("ignore")
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import torchvision.transforms as transforms
import datetime
from data_processing.sam import SAM
from data_processing.dataset_q0 import Dataset, collate_fn, config
from torch.utils.data import DataLoader
from utils.AttentionFunc import *
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='RAF-DB', choices=['RAF-DB', 'AffectNet-7', 'FERPlus', 'AffectNet-8'],
                    type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_raf_db/')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint_raf_db/')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=0.000003, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--num_classes', type=int, default=7)

args = parser.parse_args()
args.checkpoint_path = args.checkpoint_path + time_str + 'checkout.pth'
args.checkpoint_path = Path(args.checkpoint_path)
args.best_checkpoint_path = args.best_checkpoint_path + time_str + 'model_best.pth'
args.best_checkpoint_path = Path(args.best_checkpoint_path)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    model = QDDANet(num_class=args.num_classes, num_head=2, embed_dim=786,pretrained=True)

    model = torch.nn.DataParallel(model).cuda()

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_at = AttentionLoss()

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    recorder = RecorderMeter_loss(args.epochs)
    recorder_m = RecorderMeter_matrix(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            recorder_m = checkpoint['recorder_m']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code

    train_root, test_root, train_pd, test_pd, cls_num = config(dataset=args.dataset)

    data_transforms = {
        'train': transforms.Compose([transforms.Resize((112, 112)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                     transforms.RandomErasing(scale=(0.02, 0.1))]),
        'test': transforms.Compose([transforms.Resize((112, 112)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]),
    }

    # data_transforms = {
    #     'train': transforms.Compose([transforms.Resize((112, 112)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation(8),
    #         transforms.RandomCrop((112, 112)),
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         transforms.RandomErasing(scale=(0.02, 0.1))]),
    #
    #     'test': transforms.Compose([transforms.Resize((112, 112)),
    #         transforms.CenterCrop((112, 112)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),
    # }


    train_dataset = Dataset(train_root, train_pd, train=True, transform=data_transforms['train'], num_positive=1,
                            num_negative=1)
    test_dataset = Dataset(test_root, test_pd, train=False, transform=data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    if args.evaluate is not None:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            best_acc = checkpoint['best_acc']
            best_acc = best_acc.to()
            print(f'best_acc:{best_acc}')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        validate(val_loader, model, criterion_cls, criterion_at, args)
        return

    matrix = None

    for epoch in range(args.start_epoch, args.epochs):

        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        log_dir = Path("log_raf_db")
        log_dir.mkdir(parents=True, exist_ok=True)  # create the directory if it doesn't exist
        txt_name = log_dir / f"{time_str}log.txt"
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_los_1, train_los_2, train_los_3, train_los_4 = train(train_loader, model, criterion_cls, criterion_at, optimizer, epoch,
                                                                   args)

        # evaluate on a validation set
        val_acc, val_los, output, target, D = validate(val_loader, model, criterion_cls, criterion_at, args)

        scheduler.step()

        recorder.update(epoch, train_los_1, train_los_2, train_los_3, train_los_4)
        recorder_m.update(output, target)

        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log_raf_db/', curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())

        if is_best:
            matrix = D
            recorder_m.plot_confusion_matrix(cm=matrix)

        print('Current best matrix: ', matrix)

        log_dir = Path("log_raf_db")
        log_dir.mkdir(parents=True, exist_ok=True)  # create the directory if it doesn't exist

        txt_name = log_dir / f"{time_str}log.txt"
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder_m': recorder_m,
                         'recorder': recorder}, is_best, args)


def train(train_loader, model, criterion_cls, criterion_at, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.5f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    losses_1 = AverageMeter('Loss_1', ':.5f')
    losses_2 = AverageMeter('Loss_2', ':.5f')
    losses_3 = AverageMeter('Loss_3', ':.5f')
    losses_4 = AverageMeter('Loss_4', ':.5f')

    progress = ProgressMeter(len(train_loader),
                             [losses, top1, losses_1, losses_2, losses_3, losses_4],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):

        anchor_image, positive_image, negative_image, negative_image2, label, neg_label = data
        # print(image.shape)
        anchor_image = anchor_image.cuda()
        positive_image = positive_image.cuda()
        negative_image = negative_image.cuda()
        negative_image2 = negative_image2.cuda()
        label = torch.Tensor(label).type(torch.int64).cuda()
        neg_label = torch.Tensor(neg_label).type(torch.int64).cuda()

        '''----------------------  first_step  ----------------------'''
        # compute output
        (output1, output2, output3, output4, output5, output6, output7, output8,
         output9, output10, output11, output12, output13, output14, output15, output16)  = model(anchor_image,
                                                                                                 positive_image, negative_image, negative_image2)
        loss1 = criterion_cls(output1, label)  + 0.1 * criterion_at(output9)
        loss2 = criterion_cls(output2, label) + 0.1 * criterion_at(output10)
        loss3 = criterion_cls(output3, neg_label) + 0.1 * criterion_at(output11)
        loss4 = criterion_cls(output4, neg_label) + 0.1 * criterion_at(output12)
        loss5 = criterion_cls(output5, label) + 0.1 * criterion_at(output13)
        loss6 = criterion_cls(output6, label) + 0.1 * criterion_at(output14)
        loss7 = criterion_cls(output7, neg_label) + 0.1 * criterion_at(output15)
        loss8 = criterion_cls(output8, neg_label) + 0.1 * criterion_at(output16)

        loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8) / 8

        # measure accuracy and record loss
        acc1, _ = accuracy(output1, label, topk=(1, 5))
        losses.update(loss.item(), anchor_image.size(0))
        top1.update(acc1[0], anchor_image.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        loss.backward()

        # for name, param in model.named_parameters():
        #    if (name == "module.cross_attention_3.proj.weight"):
        #        print(f"Layer: {name}, Grad:{param.grad}")

        optimizer.first_step(zero_grad=True)

        '''----------------------  second_step  ----------------------'''

        (output1, output2, output3, output4, output5, output6, output7, output8,
         output9, output10, output11, output12, output13, output14, output15, output16) = model(anchor_image,
                                                                                                positive_image,
                                                                                                negative_image,
                                                                                                negative_image2)

        loss1 = criterion_cls(output1, label) + 0.1 * criterion_at(output9)
        loss2 = criterion_cls(output2, label) + 0.1 * criterion_at(output10)
        loss3 = criterion_cls(output3, neg_label) + 0.1 * criterion_at(output11)
        loss4 = criterion_cls(output4, neg_label) + 0.1 * criterion_at(output12)
        loss5 = criterion_cls(output5, label) + 0.1 * criterion_at(output13)
        loss6 = criterion_cls(output6, label) + 0.1 * criterion_at(output14)
        loss7 = criterion_cls(output7, neg_label) + 0.1 * criterion_at(output15)
        loss8 = criterion_cls(output8, neg_label) + 0.1 * criterion_at(output16)

        loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8) / 8

        # measure accuracy and record loss
        acc1, _ = accuracy(output1, label, topk=(1, 5))
        losses.update(loss.item(), anchor_image.size(0))
        top1.update(acc1[0], anchor_image.size(0))
        '----------loss-----------'
        losses_1.update(loss1.item(), anchor_image.size(0))
        losses_2.update(loss2.item(), anchor_image.size(0))
        losses_3.update(loss5.item(), anchor_image.size(0))
        losses_4.update(loss6.item(), anchor_image.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        loss.backward()

        optimizer.second_step(zero_grad=True)

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return losses_1.avg, losses_2.avg, losses_3.avg, losses_4.avg


def validate(val_loader, model, criterion_cls, criterion_at, args):
    losses = AverageMeter('Loss', ':.5f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    D = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            output, head = model(images, None, None, None)
            loss = criterion_cls(output, target) + 0.1 * criterion_at(head)

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            topk = (1,)
            # """Computes the accuracy over the k top predictions for the specified values of k"""
            with torch.no_grad():
                maxk = max(topk)
                # batch_size = target.size(0)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

            output = pred
            target = target.squeeze().cpu().numpy()
            output = output.squeeze().cpu().numpy()

            im_re_label = np.array(target)
            im_pre_label = np.array(output)
            y_ture = im_re_label.flatten()
            im_re_label.transpose()
            y_pred = im_pre_label.flatten()
            im_pre_label.transpose()

            C = metrics.confusion_matrix(y_ture, y_pred, labels=[0, 1, 2, 3, 4, 5, 6])
            D += C

            if i % args.print_freq == 0:
                progress.display(i)

        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open('./log_raf_db/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    print(D)
    return top1.avg, losses.avg, output, target, D


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        # best_state = state.pop('optimizer')
        torch.save(state, args.best_checkpoint_path)


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
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log_raf_db/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print("预测:", pred[:10])
        # print("标签:", target[:10])
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter_matrix(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, output, target):
        self.y_pred = output
        self.y_true = target

    def plot_confusion_matrix(self, cm):
        D_raf_norm = [[x * 100 / sum(sublist) for x in sublist] for sublist in cm]
        D_raf_text = [['{:.1f}%'.format(x * 100 / sum(sublist)) for x in sublist] for sublist in cm]

        fig_raf, ax_raf = plt.subplots()
        sns.heatmap(D_raf_norm, cmap='Blues', square=True, annot=D_raf_text, fmt='', cbar=False, ax=ax_raf,
                    annot_kws={'size': 7, 'ha': 'center', 'va': 'center'})

        x_labels_raf = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
        y_labels_raf = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
        ax_raf.set_xticklabels(x_labels_raf, fontsize=7)
        ax_raf.set_yticklabels(y_labels_raf, fontsize=7)
        ax_raf.set_xlabel('Predicted', fontsize=10)
        ax_raf.set_ylabel('True', fontsize=10)
        ax_raf.set_title('RAF-DB', fontsize=12)
        fig_raf.savefig('./log_raf_db/' + time_str + '-matrix.png', dpi=300)

        print('Saved matrix')

    def matrix(self):
        target = self.y_true
        output = self.y_pred
        im_re_label = np.array(target)
        im_pre_label = np.array(output)
        y_ture = im_re_label.flatten()
        # im_re_label.transpose()
        y_pred = im_pre_label.flatten()
        im_pre_label.transpose()


class RecorderMeter_loss(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 4), dtype=np.float32)  # [epoch, train/val]
        # self.epoch_accuracy = np.zeros((self.total_epoch, 4), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss_1, train_loss_2, train_loss_3, train_loss_4):
        self.epoch_losses[idx, 0] = train_loss_1
        self.epoch_losses[idx, 1] = train_loss_2
        self.epoch_losses[idx, 2] = train_loss_3
        self.epoch_losses[idx, 3] = train_loss_4

        # self.epoch_accuracy[idx, 0] = train_acc
        # self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the losses curve of train'
        dpi = 80
        width, height = 1800, 1600
        legend_fontsize = 35
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 0.3)
        interval_y = 0.015
        interval_x = 10
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x), fontsize=15)
        plt.yticks(np.arange(0, 0.3 + interval_y, interval_y), fontsize=15)
        plt.grid()
        plt.title(title, fontsize=40)
        plt.xlabel('epoch', fontsize=35)
        plt.ylabel('loss', fontsize=35)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='r', linestyle='-', label='loss_base_a', lw=3)
        plt.legend(loc=1, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='loss_base_p', lw=3)
        plt.legend(loc=1, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 2]
        plt.plot(x_axis, y_axis, color='b', linestyle='-', label='loss_cross_a', lw=3)
        plt.legend(loc=1, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 3]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='loss_cross_p', lw=3)
        plt.legend(loc=1, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    print("Python path:", sys.executable)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")
    main()
