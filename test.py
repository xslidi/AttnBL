import numpy as np
import torch
from argparse import ArgumentParser
from models import HDRnetModel
from torchvision.utils import save_image
from utils import psnr, load_test_ckpt, AvgMeter
from datasets import Eval_Dataset
from torch.utils.data import DataLoader
import os
import time
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

def test(params, model, val_loader):
    model.to(device)
    model.eval()
    psnr_meter = AvgMeter()

    start = time.time()
    for batch_idx, (low, full, target, fname) in enumerate(val_loader):
        with torch.no_grad():

            low = low.to(device)
            full = full.to(device)
            target = target.to(device)

            # Normalize to [0, 1] on GPU
            if params['hdr']:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
            else:
                low = torch.div(low, 255.0)
                full = torch.div(full, 255.0)

            output = model(low, full)
            target = torch.div(target, 255.0)
            
            output_fname = params['save_path'] + '/' + fname[0]
            if params['speed']:
                avg = (time.time() - start) *1000 / (batch_idx+1) 
                print('Throughout: %0.2f ms' % avg)
            else:
                if os.path.exists(output_fname):
                    print('file_exist')
                    continue
                save_image(output, output_fname)
                eval_psnr = psnr(output, target).item()
                print(fname[0], eval_psnr)
                psnr_meter.update(eval_psnr)

def test_speed(params, model, val_loader):
    model.to(device)
    model.eval()
    dummy_full = torch.randn(1, 3, 3840, 2160, dtype=torch.float).to(device)
    dummy_low = resize(dummy_full, (256, 256), InterpolationMode.NEAREST)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_low, dummy_full)    
    for rep in range(repetitions):
        with torch.no_grad():
            starter.record()

            # Normalize to [0, 1] on GPU
            if params['hdr']:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
            else:
                low = torch.div(dummy_low, 255.0)
                full = torch.div(dummy_full, 255.0)

            _ = model(low, full)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time   
            mean_syn = np.sum(timings) / (rep+1)          
            print('Throughout: %0.2f ms' % mean_syn)


def parse_args():
    parser = ArgumentParser(description='HDRnet testing')
    parser.add_argument('--eval_data_dir', type=str, required=True, help='Test image path')
    parser.add_argument('--save_path', type=str, default='./results', help='Test image path')
    parser.add_argument('--ckpt_path', type=str, help='Checkpoint path')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--input_res', default=256, type=int, help='Resolution of the down-sampled input')
    parser.add_argument('--hdr', action='store_true', help='Handle HDR image')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--speed', action='store_true', help='test the model speed')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse test parameters
    new_params = vars(parse_args())
    if new_params['gpu_ids']:
        device = torch.device('cuda:{}'.format(new_params['gpu_ids'][0]))
    else:
        device = torch.device("cpu")
    # Dataloader for validation
    valid_dataset = Eval_Dataset(new_params)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, pin_memory=True)

    # Test model
    state_dict, params = load_test_ckpt(new_params['ckpt_path'], map_location=device)
    params.update(new_params)
    model = HDRnetModel(params)
    model.load_state_dict(state_dict)

    test_speed(params, model, valid_loader)