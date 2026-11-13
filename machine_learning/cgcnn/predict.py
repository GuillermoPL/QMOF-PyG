import argparse
import os
import sys
import time
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from cgcnn.data import CIFData, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet
from cgcnn.utils import AverageMeter, Normalizer, class_eval, mae

parser = argparse.ArgumentParser(
    description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--disable-save-torch', action='store_true',
                    help='Do not save processed crystal graphs as .pt files')
parser.add_argument('--clean-torch', action='store_true',
                    help='Clean processed crystal graph .pt files')
parser.add_argument('--train-val-test', action='store_true',
                    help='Return training/validation/testing results')

args = parser.parse_args(sys.argv[1:])
args.cuda = not args.disable_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=device)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

if model_args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, model_args, best_mae_error

    # load data
    dataset = CIFData(args.cifpath, disable_save_torch=args.disable_save_torch)

    if args.train_val_test:
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            batch_size=model_args.batch_size,
            train_ratio=model_args.train_ratio,
            num_workers=args.workers,
            val_ratio=model_args.val_ratio,
            test_ratio=model_args.test_ratio,
            pin_memory=args.cuda,
            train_size=model_args.train_size,
            val_size=model_args.val_size,
            test_size=model_args.test_size,
            return_test=True)
    else:
        test_loader = DataLoader(dataset, batch_size=model_args.batch_size,
                                 shuffle=True, num_workers=args.workers,
                                 pin_memory=args.cuda)

    # make output folder if needed
    if not os.path.exists('output'):
        os.mkdir('output')

    # make and clean torch files if needed
    torch_data_path = os.path.join(args.cifpath, 'cifdata')
    if args.clean_torch and os.path.exists(torch_data_path):
        shutil.rmtree(torch_data_path)
    if os.path.exists(torch_data_path) and not args.clean_torch:
        warnings.warn('Found cifdata folder at ' +
                      torch_data_path+'. Will read in .pt files as-available')

    # build model
    sample_graph = dataset[0]
    orig_atom_fea_len = sample_graph.x.shape[-1]
    nbr_fea_len = sample_graph.edge_attr.shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task ==
                                'classification' else False)
    model.to(device)

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    normalizer = Normalizer(torch.zeros(3))

    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    if args.train_val_test:
        print('---------Evaluate Model on Train Set---------------')
        validate(train_loader, model, criterion, normalizer, test=True,
                 csv_name='train_results.csv')
        print('---------Evaluate Model on Val Set---------------')
        validate(val_loader, model, criterion, normalizer, test=True,
                 csv_name='val_results.csv')
        print('---------Evaluate Model on Test Set---------------')
        validate(test_loader, model, criterion, normalizer, test=True,
                 csv_name='test_results.csv')
    else:
        print('---------Evaluate Model on Dataset---------------')
        validate(test_loader, model, criterion, normalizer, test=True,
                 csv_name='predictions.csv')


def validate(val_loader, model, criterion, normalizer, test=False,
             csv_name='test_results.csv'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, batch in enumerate(val_loader):
        batch = batch.to(device, non_blocking=True)
        target = batch.y.cpu()
        batch_cif_ids = batch.cif_id
        if model_args.task == 'regression':
            target_var = normalizer.norm(batch.y)
        else:
            target_var = batch.y.view(-1).long()

        # compute output
        with torch.no_grad():
            output = model(batch.x, batch.edge_index, batch.edge_attr,
                           batch.batch)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        if model_args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if model_args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                          i+1, len(val_loader), batch_time=batch_time, loss=losses,
                          mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                          i+1, len(val_loader), batch_time=batch_time, loss=losses,
                          accu=accuracies, prec=precisions, recall=recalls,
                          f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open(os.path.join('output', csv_name), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if model_args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


if __name__ == '__main__':
    main()
