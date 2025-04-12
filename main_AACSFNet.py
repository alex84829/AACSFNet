import os
import time
import scipy.io
import torch
import random
import numpy as np
import torchvision.models as models

from torch.utils.data import DataLoader
from torch import optim, nn
from torchnet import meter
from datetime import datetime
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm
from torchvision.models import ResNet50_Weights

from config import opt, write_arguments_file
from data.dataset_resnet50_online import UNBCdataset
from utils.compute_MAE import *
from utils.pytorchtools import EarlyStopping
from model.AACSFNet import AACSFNet

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2)  # 设置随机数种子, 保证实验可重复性


def train(person=0):
    if opt.gpu == -1:
        opt.device = "cpu"
    else:
        torch.cuda.set_device(opt.gpu)
        torch.cuda.empty_cache()

    model = AACSFNet()
    # print(model)
    criterion = nn.CrossEntropyLoss()
    # model.fc = nn.Linear(64, 1)


    if opt.resume:
        print('Using Last version model...', end=' ')
        checkpoint = torch.load(opt.checkpoint_path)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        opt.init_epoch = checkpoint['epoch'] + 1

    # model = nn.DataParallel(model, device_ids=[0, 1, 2])
    # model = model.cuda(device=opt.device)
    model.to(opt.device)

    print('Model Loaded.')

    # data Loading

    traindata = UNBCdataset(id=person, mode="train")
    valdata = UNBCdataset(id=person, mode="val")

    traindata_loader = DataLoader(traindata,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  # sampler=sampler,
                                  drop_last=True,
                                  num_workers=0,   #默认为4，因为OSError。暂时改成0
                                  pin_memory=True
                                  )
    valdata_loader = DataLoader(valdata,
                                batch_size=opt.test_batch_size,
                                shuffle=False,
                                drop_last=True,
                                num_workers=0,    #默认为4，因为OSError。暂时改成0
                                pin_memory=True)

    print('Train data Loaded.')

    ##################################
    # Optimizer & loss  & loss meter
    ##################################
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': opt.lr}], lr=opt.lr,
                           weight_decay=opt.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1, last_epoch=opt.init_epoch - 1)
    # print("Current learning rate : %g" % (scheduler.get_lr()[0]))

    loss = nn.MSELoss()

    train_loss = meter.AverageValueMeter()
    val_loss = meter.AverageValueMeter()


    print('Start training...')
    best_PCC = 0.0
    for e in range(opt.init_epoch, opt.init_epoch + opt.epochs):

        # Train
        tic1 = time.perf_counter()
        train_loss.reset()
        val_loss.reset()
        model.train()
        for i, (images, labels) in enumerate(tqdm(traindata_loader)):

            images = images.type(torch.FloatTensor)
            images = images.to(opt.device, non_blocking=True)

            labels = labels.type(torch.FloatTensor)
            labels = labels.to(opt.device, non_blocking=True)

            preds = model(images)
            preds = preds.squeeze(-1)

            optimizer.zero_grad()

            running_loss = loss(preds, labels)

            running_loss.backward()

            optimizer.step()

            train_loss.add(running_loss.item())

        scheduler.step()

        # Validation
        model.eval()
        indx = 0
        num_images = len(valdata)
        preds_list = np.zeros(num_images)
        labels_list = np.zeros(num_images)
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(valdata_loader)):
                images = images.type(torch.FloatTensor)
                images = images.to(opt.device, non_blocking=True)

                labels = labels.type(torch.FloatTensor)
                labels = labels.to(opt.device, non_blocking=True)

                preds = model(images)
                preds = preds.squeeze(-1)

                running_loss = loss(preds, labels)

                val_loss.add(running_loss.item())

                preds_list[indx: (indx + preds.shape[0])] = preds.to("cpu").data.numpy()
                labels_list[indx: (indx + labels.shape[0])] = labels.to("cpu").data.numpy()
                indx = indx + preds.shape[0]

            preds_list = preds_list.transpose()
            labels_list = labels_list.transpose()

            MAE = test_MAE(preds_list, labels_list)
            MSE = test_MSE(preds_list, labels_list)
            PCC = test_PCC(preds_list, labels_list)

            ##################################
            # Save Log
            ##################################
            toc1 = time.perf_counter()
            consuming = toc1 - tic1
            clk = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            print('[person / epoch =  {:0>2d} / {:0>3d}] '
                  '[MAE = {:.4f} \t MSE = {:.4f}]'
                  '[{}  elapsed : {:.1f}min] '.format
                  (person, e,
                   train_loss.value()[0], val_loss.value()[0],
                   MAE, MSE, PCC,
                   clk, consuming / 60.0))

            opt.experiment_path = "D:\\pythonproject\\painfull go\\test\\train_resnet50_cbam"
            opt.log_dir_name = "D:\\pythonproject\\painfull go\\test\\train_resnet50_cbam\\checkpoints"
            # opt.experiment_path = "D:\\pythonproject\\painfull go\\test\\train_AACSFNet"
            # opt.log_dir_name = "D:\\pythonproject\\painfull go\\test\\train_AACSFNet\\checkpoints"
            with open(os.path.join(opt.experiment_path, opt.log_dir_name, '{}_log.txt'.format(opt.version)), 'a+') as f:
                f.write(
                    '[person / epoch =  {:0>2d} / {:0>3d}] '
                    '[MAE = {:.4f} \t MSE = {:.4f} ]  '
                    '[{}  elapsed : {:.1f}min]\n'.format
                    (person, e,
                     train_loss.value()[0], val_loss.value()[0],
                     MAE, MSE, PCC,
                     clk, consuming / 60.0))

        # Save model
        if e % 5 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'person': person,
                'epoch': e,
            }

            opt.experiment_path = "D:\\pythonproject\\painfull go\\test\\train_AACSFNet"
            opt.log_dir_name = "D:\\pythonproject\\painfull go\\test\\train_AACSFNet\\checkpoints"
            save_path = os.path.join(opt.experiment_path, opt.model_dir_name, opt.version)
            if os.path.exists(save_path) is False:
                os.mkdir(save_path)

            torch.save(checkpoint,
                       os.path.join(save_path,
                                    'model_pain_person{:0>2d}_epoch_{}_{:.0f}.pth'.format(person, e, 1000 * best_PCC)))

        if e == 0:
            write_arguments_file(
                os.path.join(opt.experiment_path, opt.log_dir_name, '{}_args.json'.format(opt.version)), opt)


if __name__ == '__main__':
    # for i in [24]:
    for i in range(1, 25):
        person = i

        opt.version = 'AACSFNet_person{:0>2d}'.format(person)
        print(opt.version)
        train(person=person)

