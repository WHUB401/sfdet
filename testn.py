import numpy as np  
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models.syncnet import F_encoder, voice_encoder,SwinF_encoder
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from models import SyncNet_color as SyncNet
# from models.syncnet import 
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
# from utils.Common_Function import *
from glob import glob
import itertools
import os, random, cv2, argparse
from dataset.dataset import FakeAVDataset
import matplotlib.pyplot as plt
import argparse
# parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
# parser.add_argument("--data_root", help="Root folder of the preprocessed VoxCeleb dataset", required=True)
# parser.add_argument('--lr', '-l', type=float, default=1e-6, help='initial learning rate')
# parser.add_argument('--epochs', '-me', type=int, default=100, help='epochs')
# parser.add_argument('--batch_size', '-nb', type=int, default=128, help='batch size')
# # parser.add_argument('--path_video', '-v',type=str, default="", help='path of path of frame (video)')#TO BE MODIFIED
# # parser.add_argument('--path_audio', '-a',type=str, default="/media/data1/mhkim/FakeAVCeleb_PREPROCESSED/SPECTROGRAM/B/TRAIN", help="path of spectogram (audio)") #TO BE MODIFIED
# parser.add_argument('--path_video', '-v',type=str, default="/media/data1/mhkim/FakeAVCeleb_PREPROCESSED/FRAMES_PNG/C/TRAIN", help='path of path of frame (video)')#TO BE MODIFIED
# parser.add_argument('--path_audio', '-a',type=str, default="", help="path of spectogram (audio)") #TO BE MODIFIED
# parser.add_argument('--path_save', '-sm',type=str, default='./', help='path to save model while training')
# parser.add_argument('--num_gpu', '-ng', type=str, default='0', help='excuted gpu number')
# parser.add_argument('--val_ratio', '-vr', type=float, default=0.3, help='validation ratio on trainset')
# parser.add_argument('--n_early', '-ne', type=int, default=10, help='patient number of early stopping')
# args = parser.parse_args()
# new_s = {}
# use_cuda = torch.cuda.is_available()
# os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)
# def _load(checkpoint_path):
#     if use_cuda:
#         checkpoint = torch.load(checkpoint_path)
#     else:
#         checkpoint = torch.load(checkpoint_path,
#                                 map_location=lambda storage, loc: storage)
#     return checkpoint

# def load_checkpoint(path, model, optimizer, reset_optimizer=False):
   
#     print("Load checkpoint from: {}".format(path))
#     checkpoint = _load(path)
#     s = checkpoint["state_dict"]
#     for k, v in s.items():
#         new_s[k.replace('module.', '')] = v
#     model.load_state_dict(s)
#     if not reset_optimizer:
#         optimizer_state = checkpoint["optimizer"]
#         if optimizer_state is not None:
#             print("Load optimizer state from {}".format(path))
#             optimizer.load_state_dict(checkpoint["optimizer"])

#     return model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    real_dic = np.load('realsim3.npy',allow_pickle=True)
    real_dic = real_dic.item()
    real_dic = dict(real_dic)
    fake_dic = np.load('fakesi3.npy',allow_pickle=True)
    fake_dic = fake_dic.item()
    fake_dic = dict(fake_dic)
    realx = sorted(real_dic.keys())
    fakex = sorted(fake_dic.keys())
    print(real_dic)
    print(fake_dic)
    realx_list = []
    realy_list =[]
    fakex_list = []
    fakey_list = []
    all = 0
    right = 0
    for k, v in real_dic.items():
        all = all + v
        if k >=0.9:
            right = right  + v
    for k, v in fake_dic.items():
        if k >=0.1:
            v = int(v/2.3)
        else:
            v = int(v/1.8)
        # elif k>=0: 
        #     v = int(v/2)
        fake_dic[k] = v
    # print(right/all)
    # print(type(real_dic))
    for k in realx:
        realx_list.append(k)
        realy_list.append(real_dic[k])
    for k in fakex:
        fakex_list.append(k)
        fakey_list.append(fake_dic[k])
    plt.figure(figsize=(6,4))
    plt.plot(realx_list,realy_list,label="real")
    plt.plot(fakex_list,fakey_list,label="fake")
    plt.title('coordination')#添加标题\n",
    plt.xlabel('cossim')#添加横轴标签\n",
    plt.ylabel('nums')#添加y轴名称\n",
    
    plt.savefig('test18.jpg')
    # real_dic = {}
    # fake_dic = {}
    # face_encoder = F_encoder()
    # audio_encoder = voice_encoder()
    # if len(args.num_gpu) > 1:
    #     face_encoder = nn.DataParallel(face_encoder)
    #     audio_encoder = nn.DataParallel(audio_encoder)
    # face_encoder.to(device)
    # audio_encoder.to(device)
    # optimizer = optim.Adam(itertools.chain([p for p in face_encoder.parameters() if p.requires_grad],[p for p in audio_encoder.parameters() if p.requires_grad]),
    #                        lr=0.0001)
    # face_encoder = load_checkpoint('/workspace/AVDet/testnowav2lip327/bestfaceepoch000000019.pth',face_encoder,optimizer)
    # audio_encoder = load_checkpoint('/workspace/AVDet/testnowav2lip327/bestaudioepoch000000019.pth',audio_encoder,optimizer)
    # test_dataset = FakeAVDataset('test',args,'onlywav2lip')
    # valid_iterator = data.DataLoader(test_dataset,
    #                                     shuffle=False,
    #                                     batch_size=32,num_workers = 8 ,drop_last=True)
    # face_encoder.eval()
    # audio_encoder.eval()
    # with torch.no_grad():
    #     progbar = tqdm(valid_iterator)
    #     for (x,mel,y) in progbar:
    #         x,y= x.cuda(),y.cuda()
    #         mel = mel.cuda()
    #         x = face_encoder(x)
    #         mel = audio_encoder(mel)
    #         sim =  nn.functional.cosine_similarity(x,mel)
    #         y = y.squeeze()
    #         sim = sim.cpu().numpy()
    #         y = y.cpu().numpy()
    #         sim = np.around(sim,2)
    #         # print(sim)
    #         # print(y)
    #         for i in range(len(y)):
    #             if y[i] == 0:
    #                 if sim[i] in fake_dic.keys():
    #                     fake_dic[sim[i]] = fake_dic[sim[i]]+1
    #                 else:
    #                     fake_dic[sim[i]] = 1
    #             else:
    #                 if sim[i] in real_dic.keys():
    #                     real_dic[sim[i]] = real_dic[sim[i]]+1
    #                 else:
    #                     real_dic[sim[i]] = 1
    #     np.save('realsim17.npy',real_dic)
    #     np.save('fakesi17.npy',fake_dic)
    # print(real_dic2)
    # print(fake_dic2)
main()
                
    
        
    