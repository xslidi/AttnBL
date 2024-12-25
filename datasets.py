import numpy as np
import os
import rawpy
import torch
from imageio import imread
from torchvision import transforms
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset
from utils import get_files
from torchvision.transforms import InterpolationMode


class BaseDataset(Dataset):

    def load_img(self, fname):
        input = imread(os.path.join(self.data_path, 'input', fname))
        output = imread(os.path.join(self.data_path, 'output', fname))
        if fname.split('.')[1] == 'tif':
            input = input.astype(np.int32)
           
        input = torch.from_numpy(input.transpose((2, 0, 1)))
        output = torch.from_numpy(output.transpose((2, 0, 1)))
        return input, output

    def load_img_hdr(self, fname):
        input = rawpy.imread(os.path.join(self.data_path, 'input', fname))
        input = input.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        input = np.asarray(input, dtype=np.float32)
        # Output may not necessarily be .jpg images, so change when necessary.
        output = imread(os.path.join(self.data_path, 'output', fname.split('.')[0] + '.png'))

        input = torch.from_numpy(input.transpose((2, 0, 1)))
        output = torch.from_numpy(output.transpose((2, 0, 1)))

        return input, output

    def __len__(self):
        return len(self.input_paths)


class Train_Dataset(BaseDataset):
    """Class for training images."""

    def __init__(self, params=None):
        self.data_path = params['train_data_dir']
        self.input_paths = get_files(os.path.join(self.data_path, 'input'))
        self.input_res = params['input_res']
        self.output_res = params['output_res']
        self.augment = transforms.Compose([
            transforms.RandomCrop(self.output_res),
            # transforms.RandomResizedCrop(self.output_res, scale=(0.6, 1.0), ratio=(1, 1), antialias=True),            
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
        ])
        self.params = params

    def __getitem__(self, idx):
        fname = self.input_paths[idx].split('/')[-1]
        if self.params['hdr']:
            input, output = self.load_img_hdr(fname)
        else:
            input, output = self.load_img(fname)
        # Check dimensions before crop
        assert input.shape == output.shape
        assert self.output_res[0] <= input.shape[2]
        assert self.output_res[1] <= input.shape[1]
        # Crop
        inout = torch.cat([input,output],dim=0)
        inout = self.augment(inout)

        full = inout[:3,:,:]
        full = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(full)
        low = resize(full, (self.input_res, self.input_res), InterpolationMode.NEAREST)
        output = inout[3:,:,:]

        return low, full, output


class Eval_Dataset(BaseDataset):
    """Class for validation images."""

    def __init__(self, params=None):
        self.data_path = params['eval_data_dir']
        self.input_paths = get_files(os.path.join(self.data_path, 'input'))
        self.input_res = params['input_res']
        self.params = params

    def __getitem__(self, idx):
        fname = self.input_paths[idx].split('/')[-1]
        if self.params['hdr']:
            full, output = self.load_img_hdr(fname)
        else:
            full, output = self.load_img(fname)
        low = resize(full, (self.input_res, self.input_res), InterpolationMode.NEAREST)
        return low, full, output, fname
