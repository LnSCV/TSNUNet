import os
import sys
import time

import tqdm
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torch.utils.data.dataloader import DataLoader

import random

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.misc import *
from data.dataloader import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# device_num = torch.cuda.device_count()
# print('device_num', device_num)
# os.environ.setdefault("CUDA_VISIBLE_DEVICES","1")

def set_seed(seed):
    """
    Set random seeds for reproducibility across different libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(opt, args):
    print('args.device', args.device)
    # 设置随机种子以确保测试的可复现性
    SEED = 1024
    set_seed(SEED)
    model = (eval(opt.Model.name)(**opt.Model))
    model.to(args.device)
    # state_dict = torch.load(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'latest.pth'),map_location='cuda:1')
    state_dict = torch.load(os.path.join(opt.Test.Checkpoint.checkpoint_dir, 'latest.pth'))
    model.load_state_dict(state_dict, strict=True)
    model.cuda(args.device)

    model.eval()

    if args.verbose is True:
        sets = tqdm.tqdm(opt.Test.Dataset.sets, desc='Total TestSet', total=len(
            opt.Test.Dataset.sets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        sets = opt.Test.Dataset.sets

    total_time = 0.0
    num_images = 0

    for set in sets:
        save_path = os.path.join(opt.Test.Checkpoint.checkpoint_dir, set)

        os.makedirs(save_path, exist_ok=True)
        test_dataset = eval(opt.Test.Dataset.type)(opt.Test.Dataset.root, [set], opt.Test.Dataset.transforms)
        test_loader  = DataLoader(dataset=test_dataset, batch_size=1, num_workers=opt.Test.Dataloader.num_workers, pin_memory=opt.Test.Dataloader.pin_memory)

        if args.verbose is True:
            samples = tqdm.tqdm(test_loader, desc=set + ' - Test', total=len(test_loader),
                                position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = test_loader

        for sample in samples:
            sample = to_cuda_device(sample,args.device)
            with torch.no_grad():
                time_start = time.time()
                out = model(sample)
                time_end = time.time()
            
            total_time += (time_end - time_start)
            num_images += 1

            pred = to_numpy(out['pred'], sample['shape'])
            Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(save_path, sample['name'][0] + '.png'))

    if num_images > 0:
        avg_fps = num_images / total_time
        print(f"Total images processed: {num_images}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
    else:
        print("No images processed.")

if __name__ == "__main__":
    # torch.cuda.set_device(1)
    args = parse_args()
    opt = load_config(args.config)
    test(opt, args)
