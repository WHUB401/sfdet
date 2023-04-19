from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from torch.optim import SGD
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
from hparams import hparams, get_image_list
from dataset.dataset import VoxDataset,FakeAVDataset2

from models.syncnet import F_encoder, voice_encoder,SwinF_encoder
parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed VoxCeleb dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--lr', '-l', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--epochs', '-me', type=int, default=50, help='epochs')
parser.add_argument('--batch_size', '-nb', type=int, default=400, help='batch size')
parser.add_argument('--num_gpu', '-ng', type=str, default='0', help='excuted gpu number')
args = parser.parse_args()
# set_seeds()
global_step = 0
global_epoch = 0
torch.multiprocessing.set_sharing_strategy('file_system')
print('GPU num is' , args.num_gpu)
os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)

# b = torch.randn(32,1,512)
# sim = nn.functional.cosine_similarity(a,b,dim = 2)
# print(sim.shape)
def con_loss(x,mel,device,num_neg):
    x = x.view(-1,num_neg+1,x.shape[1])
    mel = mel.view(-1,num_neg+1,mel.shape[1])
    logits = nn.functional.cosine_similarity(x,mel,dim=2)
    labels = torch.zeros(len(logits), dtype=torch.long, device=device)
    return nn.functional.cross_entropy(logits,labels)
    return loss
def train(device, face_encoder,voice_encoder, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,num_neg= 4):

    global global_epoch
    resumed_step = global_step
    
    # print('222')
    running_loss = 0.
    prog_bar = tqdm(train_data_loader)
    for (x, mel) in prog_bar:
        face_encoder.train()
        voice_encoder.train()
        optimizer.zero_grad()
        # print(333)
        # Transform data to CUDA device
        x = x.to(device)
        x = x.view(-1,x.shape[2],x.shape[3],x.shape[4])
        # print(x.shape)
        mel = mel.to(device)
        mel = mel.view(-1,mel.shape[2],mel.shape[3],mel.shape[4])
        x = face_encoder(x)
        # print(x.shape)
        mel = voice_encoder(mel)
        loss = con_loss(x,mel,device,num_neg)
        # print('1111')
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


        prog_bar.set_description('Loss: {}'.format(loss))
    train_loss = running_loss/len(train_data_loader)
    return train_loss
def test(device, face_encoder,voice_encoder,test_data_loader, 
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,num_neg = 4):

    global global_epoch
    resumed_step = global_step
    
    
    running_loss = 0.
    prog_bar = tqdm(test_data_loader)
    for (x,mel) in prog_bar:
        face_encoder.eval()
        voice_encoder.eval()
        # Transform data to CUDA device
        x = x.to(device)
        x = x.view(-1,x.shape[2],x.shape[3],x.shape[4])
        # print(x.shape)
        mel = mel.to(device)
        mel = mel.view(-1,mel.shape[2],mel.shape[3],mel.shape[4])
        x = face_encoder(x)
    
        mel = voice_encoder(mel)
        
        loss = con_loss(x,mel,device,num_neg)
        running_loss += loss.item()
        prog_bar.set_description('Loss: {}'.format(loss))
    test_loss = running_loss/len(test_data_loader)
    return test_loss
def save_checkpoint(model, optimizer,checkpoint_dir, epoch,type):

    checkpoint_path = join(
        checkpoint_dir, "{}epoch{:09d}.pth".format(str(type),epoch))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)
use_cuda = torch.cuda.is_available()
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint
# new_s = []
def load_checkpoint(path, model, optimizer, reset_optimizer=False):
   
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    # for k, v in s.items():
    #     new_s[k.replace('module.', '')] = v
    model.load_state_dict(s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])

    return model
if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
    num_neg = 4
    train_dataset = FakeAVDataset2('train',args,'nowav2lip',num_neg)
    test_dataset = FakeAVDataset2('test',args,'nowav2lip',num_neg)
    # train_dataset = VoxDataset('train',args,wrong_id_nums= wrong_id_nums,not_sync_nums=not_sync_nums)
    # test_dataset = VoxDataset('test',args,wrong_id_nums= wrong_id_nums,not_sync_nums=not_sync_nums)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=32,shuffle=True,
        num_workers=8,drop_last=True)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=16,shuffle=True,
        num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_encoder = F_encoder()
    audio_encoder = voice_encoder()
    # # voicecheckpoint = torch.load('/workspace/Wav2Lip-master/wav2lip_gan.pth')
    # s = voicecheckpoint["state_dict"]
    # new_s = {}
    # for k, v in s.items():
    #     # print(k)
    #     # print(k.replace('module.', ''))
    #     new_s[k.replace('module.', '')] = v
    # # audio_encoder.load_state_dict(new_s,strict=False)
    # device2 = torch.device('cpu')
    # for key in audio_encoder.state_dict():
    #     # print(audio_encoder.state_dict()[key] == s[key].to(device2))
    #     try:
    #         print(audio_encoder.state_dict()[key] == new_s[key].to(device2))
    #         # print('success')
    #     except:
    #         None
    #         # print('fffff')
    # print('------------------------------------------')
    modeldict = audio_encoder.state_dict()
    # for key in audio_encoder.state_dict():
    #     # print(mo.state_dict()[key] == s[key])
    #     if key in new_s.keys():
    #     #     print('111')
    #         try:
    #             # audio_encoder.state_dict()[key] = new_s[key].to(device2)
    #             print(audio_encoder.state_dict()[key] == new_s[key].to(device2))
    #             # print(new_s[key].to(device2))
    #             # break
    #             # print('sfsafsaf')
    #         except :
    #             print('hahaha')
    # pretrained_dict = {k: v.to(device2) for k, v in new_s.items() if k in modeldict and modeldict[k].to(device2).shape == new_s[k].shape}
    # modeldict.update(pretrained_dict)
    # audio_encoder.load_state_dict(modeldict)
    # for key in audio_encoder.state_dict():
    #     # print(mo.state_dict()[key] == s[key])
    #     if key in new_s.keys():
    #     #     print('111')
    #         try:
    #             # audio_encoder.state_dict()[key] = new_s[key].to(device2)
    #             print(audio_encoder.state_dict()[key] == new_s[key].to(device2))
    #             # print(new_s[key].to(device2))
    #             # break
    #             # print('sfsafsaf')
    #         except :
    #             print('hahaha')
    # for key in audio_encoder.state_dict():
    #     # print(audio_encoder.state_dict()[key] == s[key].to(device2))
    #     try:
    #         print(audio_encoder.state_dict()[key] == new_s[key].to(device2))
    #     except:
    #         None
    #         # print('fffff')
    # face_encoder = load_checkpoint('/workspace/AVDet/test/faceepoch000000090.pth',face_encoder,optimizer)
    if len(args.num_gpu) > 1:
        face_encoder = nn.DataParallel(face_encoder)
        audio_encoder = nn.DataParallel(audio_encoder)
    face_encoder.to(device)
    audio_encoder.to(device)
    
    optimizer = optim.Adam(itertools.chain([p for p in face_encoder.parameters() if p.requires_grad],[p for p in audio_encoder.parameters() if p.requires_grad]),
                           lr=hparams.syncnet_lr)
    # face_encoder = load_checkpoint('/workspace/AVDet/testpre2023325/bestfaceepoch000000353.pth',face_encoder,optimizer)
    # audio_encoder = load_checkpoint('/workspace/AVDet/testpre2023325/bestaudioepoch000000353.pth',audio_encoder,optimizer)
    # test_loss = test(device,face_encoder,audio_encoder,test_data_loader,checkpoint_dir,230)
    # print('epoch{},val_loss:{}'.format(230,test_loss))
    best_valid_loss = float('inf')
    for epoch in range(0,20):
        train_loss = train(device,face_encoder,audio_encoder,train_data_loader,test_data_loader,optimizer,checkpoint_dir,epoch,num_neg)
        print('epoch{},train_loss:{}'.format(epoch,train_loss))
        if epoch %3 == 0:
            save_checkpoint(face_encoder,optimizer,checkpoint_dir,epoch,'face')
            save_checkpoint(audio_encoder,optimizer,checkpoint_dir,epoch,'audio')
        print('-----------------------------------')
        print('start test')
        test_loss = test(device,face_encoder,audio_encoder,test_data_loader,checkpoint_dir,epoch,num_neg)
        if test_loss < best_valid_loss:
            best_valid_loss = test_loss
            save_checkpoint(face_encoder,optimizer,checkpoint_dir,epoch,'bestface')
        
            save_checkpoint(audio_encoder,optimizer,checkpoint_dir,epoch,'bestaudio')
        print('epoch{},val_loss:{},bestloss'.format(epoch,test_loss,best_valid_loss))
        print('-----------------------------------')
    # prog_bar = tqdm(train_data_loader)
    # for (x,y,z,mel) in prog_bar:
    #     # print('hahahh')
    #     y = 1
    #     prog_bar.set_description('Loss: {}'.format(0.001))
    # prog_bar.set_description('Loss: {}'.format(0.001))
    # print('success')