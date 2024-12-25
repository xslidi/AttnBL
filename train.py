import numpy as np
import os
import time
import torch
import torch.nn as nn
from argparse import ArgumentParser
from datasets import Train_Dataset, Eval_Dataset
from models import HDRnetModel
from torch.optim import lr_scheduler, AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import psnr, print_params, load_train_ckpt, save_model_stats, AvgMeter
import random
from tqdm import tqdm
    
def train(params, train_loader, valid_loader, model, start_epoch=0):
    nodecay = 20
    # Optimization
    optimizer = AdamW(model.parameters(), params['learning_rate'], weight_decay=1e-8)
    # # Learning rate adjustment
    scheduler = lr_scheduler.LinearLR(optimizer,
        start_factor=1.0, end_factor=0.0, total_iters=params['epochs'] - nodecay, verbose=True)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'] - nodecay, eta_min=1e-6, verbose=True)

    # Loss function
    criterion_l1 = nn.L1Loss()
    # Training
    train_loss_meter = AvgMeter()
    train_psnr_meter = AvgMeter()
    stats = {'train_loss': [],
             'train_psnr': [],
             'valid_psnr': []}
    iteration = 0
    old_time = time.time()
    for epoch in range(start_epoch, params['epochs']):
        for batch_idx, (low, full, target) in enumerate(train_loader):
            iteration += 1
            model.train()

            low = low.to(device)
            full = full.to(device)
            target = target.to(device)


            # Normalize to [0, 1] on GPU
            if params['hdr']:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
            if params['ppr']:
                low = torch.div(low, 255.0)
                full = torch.div(full, 65535.0)
            else:
                low = torch.div(low, 255.0)
                full = torch.div(full, 255.0)
            target = torch.div(target, 255.0)
            output = model(low, full)

            loss = 20 * criterion_l1(output, target)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()


            if iteration % params['summary_interval'] == 0:
                save_image(target, 'target.jpg')
                save_image(full, 'full.jpg')
                save_image(output, 'output.jpg')
                train_loss_meter.update(loss.item())
                train_psnr = psnr(output, target).item()
                train_psnr_meter.update(train_psnr)
                new_time = time.time()
                print('[%d/%d] Iteration: %d | Loss: %.4f | PSNR: %.4f | Time: %.2fs' %
                        (epoch+1, params['epochs'], iteration, loss, train_psnr, new_time-old_time))
                old_time = new_time

            if iteration % params['ckpt_interval'] == 0:

                stats['train_loss'].append(train_loss_meter.avg)
                train_loss_meter.reset()
                stats['train_psnr'].append(train_psnr_meter.avg)
                train_psnr_meter.reset()
                valid_psnr = eval(params, valid_loader, model, device)
                stats['valid_psnr'].append(valid_psnr)
                ckpt_fname = "epoch_" + str(epoch+1)+'_iter_' + str(iteration) + ".pth"
                save_model_stats(model, params, ckpt_fname, stats)
        if epoch > nodecay:
            scheduler.step()


def eval(params, valid_loader, model, device):
    model.eval()
    psnr_meter = AvgMeter()
    with torch.no_grad():
        for (low, full, target, fname) in tqdm(valid_loader):
            low = low.to(device)
            full = full.to(device)
            target = target.to(device)

            # Normalize to [0, 1] on GPU
            if params['hdr']:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
            if params['ppr']:
                low = torch.div(low, 255.0)
                full = torch.div(full, 65535.0)
            else:
                low = torch.div(low, 255.0)
                full = torch.div(full, 255.0)
            target = torch.div(target, 255.0)

            output = model(low, full)
            # save_image(output, os.path.join(params['eval_out'], fname[0]))
            eval_psnr = psnr(output, target).item()
            psnr_meter.update(eval_psnr)

    print ("Validation PSNR: ", psnr_meter.avg)
    return psnr_meter.avg


def parse_args():
    parser = ArgumentParser(description='HDRnet training')
    # Training, logging and checkpointing parameters
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--ckpt_interval', default=int(8876/4*10), type=int, help='Interval for saving checkpoints, unit is iteration')
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str, help='Checkpoint directory')
    parser.add_argument('--stats_dir', default='./stats', type=str, help='Statistics directory')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # Data pipeline and data augmentation
    parser.add_argument('--batch_size', default=4, type=int, help='Size of a mini-batch')
    parser.add_argument('--train_data_dir', type=str, required=True, help='Dataset path')
    parser.add_argument('--eval_data_dir', default=None, type=str, help='Directory with the validation data.')
    parser.add_argument('--eval_out', default='./outputs', type=str, help='Validation output path')
    parser.add_argument('--hdr', action='store_true', help='Handle HDR image')
    parser.add_argument('--ppr', action='store_true', help='Handle ppr image')

    # Model parameters
    parser.add_argument('--batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--input_res', default=256, type=int, help='Resolution of the down-sampled input')
    parser.add_argument('--output_res', default=(512, 512), type=int, nargs=2, help='Resolution of the guidemap/final output')


    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':


    # Parse training parameters
    params = vars(parse_args())
    print_params(params)

    # Random seeds
    if params['seed'] > 0:
        set_seed(params['seed'])
        print('Set random seed %s' % params['seed'])

    # Folders
    os.makedirs(params['ckpt_dir'], exist_ok=True)
    os.makedirs(params['stats_dir'], exist_ok=True)
    os.makedirs(params['eval_out'], exist_ok=True)

    if params['gpu_ids']:
        device = torch.device('cuda:{}'.format(params['gpu_ids'][0]))
    else:
        device = torch.device("cpu")

    # Dataloader for training
    train_dataset = Train_Dataset(params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=8,  pin_memory=True)

    # Dataloader for validation
    valid_dataset = Eval_Dataset(params)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=8, pin_memory=True)

    # Model for training
    model = HDRnetModel(params)
    print(model)

    start = load_train_ckpt(model, params['ckpt_dir'])
    start = start if start else 0

    model.to(device)

    train(params, train_loader, valid_loader, model, start_epoch=start)
