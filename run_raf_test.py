import warnings
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import torchvision.transforms as transforms
import datetime
from data_processing.dataset_q0 import Dataset, collate_fn, config
from torch.utils.data import DataLoader
from utils.AttentionFunc import *

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='RAF-DB', choices=['RAF-DB', 'AffectNet-7', 'FERPlus', 'AffectNet-8'],
                    type=str, help='dataset option')
parser.add_argument('--model_path', type=str, required=True, help='path to trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--save_results', action='store_true', help='save test results')
parser.add_argument('--output_dir', type=str, default='./test_results/', help='output directory for results')

args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Testing time: ' + now.strftime("%m-%d %H:%M"))

    # Create output directory
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # create model
    model = QDDANet(num_class=args.num_classes, num_head=2, embed_dim=786, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    # Load trained model
    if os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        if 'best_acc' in checkpoint:
            print("=> model best accuracy: {:.3f}".format(checkpoint['best_acc']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model_path))
        return

    cudnn.benchmark = True

    # Data loading code
    train_root, test_root, train_pd, test_pd, cls_num = config(dataset=args.dataset)

    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = Dataset(test_root, test_pd, train=False, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Test the model
    test_acc, test_loss, confusion_matrix, class_report = test(test_loader, model, args)

    print(f'\nTest Results:')
    print(f'Test Accuracy: {test_acc:.3f}%')
    print(f'Test Loss: {test_loss:.5f}')
    print(f'\nConfusion Matrix:')
    print(confusion_matrix)

    # Save results if requested
    if args.save_results:
        save_test_results(test_acc, test_loss, confusion_matrix, class_report, args)


def test(test_loader, model, args):
    criterion = torch.nn.CrossEntropyLoss()
    losses = AverageMeter('Loss', ':.5f')
    top1 = AverageMeter('Accuracy', ':6.3f')

    progress = ProgressMeter(len(test_loader), [losses, top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    all_predictions = []
    all_targets = []

    # Initialize confusion matrix
    confusion_matrix = np.zeros((args.num_classes, args.num_classes), dtype=int)

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            outputs, _ = model(images, None, None, None)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Store predictions and targets for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Update confusion matrix
            for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[t][p] += 1

            if i % args.print_freq == 0:
                progress.display(i)

    # Calculate per-class accuracy and other metrics
    class_report = calculate_class_metrics(confusion_matrix, args.num_classes)

    print(f'\n **** Test Accuracy {top1.avg:.3f}% **** ')
    print(f'Per-class Results:')
    for i, (precision, recall, f1) in enumerate(class_report):
        print(f'Class {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}')

    return top1.avg, losses.avg, confusion_matrix, class_report


def calculate_class_metrics(confusion_matrix, num_classes):
    """Calculate precision, recall, and F1-score for each class"""
    class_report = []

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_report.append((precision, recall, f1))

    return class_report


def save_test_results(test_acc, test_loss, confusion_matrix, class_report, args):
    """Save test results to files"""
    output_dir = Path(args.output_dir)

    # Save confusion matrix as heatmap
    plot_confusion_matrix(confusion_matrix, output_dir / f'{time_str}confusion_matrix.png')

    # Save numerical results
    results_file = output_dir / f'{time_str}test_results.txt'
    with open(results_file, 'w') as f:
        f.write(f'Test Results - {args.dataset}\n')
        f.write(f'Model: {args.model_path}\n')
        f.write(f'Test Time: {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Test Accuracy: {test_acc:.3f}%\n')
        f.write(f'Test Loss: {test_loss:.5f}\n\n')

        f.write('Per-class Results:\n')
        class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
        for i, (precision, recall, f1) in enumerate(class_report):
            class_name = class_names[i] if i < len(class_names) else f'Class_{i}'
            f.write(f'{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n')

        f.write(f'\nConfusion Matrix:\n')
        f.write(str(confusion_matrix))

    # Save confusion matrix as numpy array
    np.save(output_dir / f'{time_str}confusion_matrix.npy', confusion_matrix)

    print(f'Results saved to {output_dir}')


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create text annotations
    cm_text = [[f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)' for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=cm_text, fmt='', cmap='Blues', square=True,
                cbar_kws={'label': 'Percentage (%)'})

    class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45, ha='right')
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Confusion matrix saved to {save_path}')


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
        print('\t'.join(entries))

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
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    print("Python path:", sys.executable)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")
    main()