import os
import cv2
import sys
import tqdm
# import torch
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda as cuda
import random # Added for random seed
import numpy as np # Added for random seed

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from lib.optim import *
from data.dataloader import *
from utils.misc import *

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For all GPUs
    
    # For full determinism, uncomment these lines. Note that this might slightly impact performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# #initial
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# #initial
# torch.backends.cudnn.deterministic = True


print("torch.cuda.is_available()",torch.cuda.is_available())
print("torch.cuda.get_device_capability()",torch.cuda.get_device_capability())
print("torch.backends.cudnn.enabled",torch.backends.cudnn.enabled)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cudnn.benchmark = False  # the default is False
# torch.backends.cudnn.deterministic = True
def train(opt, args):
    train_dataset = eval(opt.Train.Dataset.type)(
        root=opt.Train.Dataset.root, 
        sets=opt.Train.Dataset.sets,
        tfs=opt.Train.Dataset.transforms)
    
    # print('train_dataset',train_dataset)
    # print("args.local_rank",args.local_rank)
    if args.device_num > 1:
        cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num, timeout=datetime.timedelta(seconds=3600))
        # train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
        # process the line
        # print("train_sampler = None")

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=opt.Train.Dataloader.batch_size,
                            shuffle=False,
                            sampler=train_sampler,
                            num_workers=opt.Train.Dataloader.num_workers,
                            pin_memory=opt.Train.Dataloader.pin_memory,
                            drop_last=False)

    model_ckpt = None
    state_ckpt = None
    
    if args.resume is True:
        if os.path.isfile(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth')):
            model_ckpt = torch.load(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'), map_location='cpu')
            if args.local_rank <= 0:
                print('Resume from checkpoint')
        if os.path.isfile(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'state.pth')):
            state_ckpt = torch.load(os.path.join(opt.Train.Checkpoint.checkpoint_dir,  'state.pth'), map_location='cpu')
            if args.local_rank <= 0:
                print('Resume from state')
    # print("opt.Model.name",opt.Model.name)
    model = eval(opt.Model.name)(**opt.Model)  #InSPyReNet_SwinB   InSPyReNet_Res2Net50
    if model_ckpt is not None:
        model.load_state_dict(model_ckpt)

    if args.device_num > 1:  #1
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # print("args.device",args.device)
        # model = model.cuda(args.device) args.local_rank
        model = model.cuda(args.local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
    else:
        model = model.cuda(args.device)
        # process the line
        # print("model = model.cuda(args.device)")


    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)
    params_list = [{'params': backbone_params}, {
        'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10}]
    
    optimizer = eval(opt.Train.Optimizer.type)(  #"Adam"
        params_list, opt.Train.Optimizer.lr, weight_decay=opt.Train.Optimizer.weight_decay)
    
    if state_ckpt is not None:
        optimizer.load_state_dict(state_ckpt['optimizer'])
    
    if opt.Train.Optimizer.mixed_precision is True: #False
        scaler = GradScaler()
    else:
        scaler = None
        # process the line
        # print("scaler = None")

    scheduler = eval(opt.Train.Scheduler.type)(optimizer, gamma=opt.Train.Scheduler.gamma,  #"PolyLr"
                                                minimum_lr=opt.Train.Scheduler.minimum_lr,
                                                max_iteration=len(train_loader) * opt.Train.Scheduler.epoch,  # 60
                                                warmup_iteration=opt.Train.Scheduler.warmup_iteration)  #12000
    if state_ckpt is not None:
        scheduler.load_state_dict(state_ckpt['scheduler'])

    model.train()   #ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt

    start = 1
    if state_ckpt is not None:
        start = state_ckpt['epoch']
        
    epoch_iter = range(start, opt.Train.Scheduler.epoch + 1)  #61
    if args.local_rank <= 0 and args.verbose is True:  #-1
        epoch_iter = tqdm.tqdm(epoch_iter, desc='Epoch', total=opt.Train.Scheduler.epoch, initial=start - 1,
                                position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')

    for epoch in epoch_iter:#60
        if args.local_rank <= 0 and args.verbose is True:#-1
            step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
                train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
            if args.device_num > 1 and train_sampler is not None:
                train_sampler.set_epoch(epoch)
        else:
            step_iter = enumerate(train_loader, start=1)

        for i, sample in step_iter:
            optimizer.zero_grad()
            if opt.Train.Optimizer.mixed_precision is True and scaler is not None:
                with autocast():
                    sample = to_cuda_device(sample,args.device)
                    # sample = to_cuda(sample, args.local_rank)
                    out = model(sample)

                scaler.scale(out['loss']).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                sample = to_cuda_device(sample,args.device)
                # sample = to_cuda(sample, args.local_rank)
                out = model(sample) #ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
                out['loss'].backward()
                optimizer.step()
                scheduler.step()

            if args.local_rank <= 0 and args.verbose is True:  #-1   True
                step_iter.set_postfix({'loss': out['loss'].item()})

        if args.local_rank <= 0:  #-1
            os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(
                opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)
            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                if args.device_num > 1:
                    model_ckpt = model.module.state_dict()  
                else:
                    model_ckpt = model.state_dict()
                    
                state_ckpt = {'epoch': epoch + 1,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()}
                
                torch.save(model_ckpt, os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))
                torch.save(state_ckpt, os.path.join(opt.Train.Checkpoint.checkpoint_dir,  'state.pth'))
                
            if args.debug is True:
                debout = debug_tile(sum([out[k] for k in opt.Train.Debug.keys], []), activation=torch.sigmoid)
                cv2.imwrite(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)
    
    if args.local_rank <= 0: #-1
        torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(),
                    os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    args = parse_args()
    opt = load_config(args.config)

    seed = getattr(args, 'seed', 1024) # Use args.seed if it exists, otherwise use 42
    # seed = getattr(args, 'seed', 1028) 
    # seed = getattr(args, 'seed', 1032) 
    # seed = getattr(args, 'seed', 42) 
    set_all_seeds(seed)

    # print("args ", args )
    # print("opt",opt)
    train(opt, args)
