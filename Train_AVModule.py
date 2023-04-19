import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import time
import torch
from torch.cuda.amp import GradScaler
from utils.EarlyStopping import EarlyStopping
from utils.Common_Function import *
from models.syncnet import F_encoder, voice_encoder,SwinF_encoder,AVModule,VModule,VFTransformer
from dataset.dataset import FakeAVDataset,VoxDatasetv2, Dataset


import argparse
parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed VoxCeleb dataset", required=True)

# parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--lr', '-l', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--epochs', '-me', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', '-nb', type=int, default=128, help='batch size')
# parser.add_argument('--path_video', '-v',type=str, default="", help='path of path of frame (video)')#TO BE MODIFIED
# parser.add_argument('--path_audio', '-a',type=str, default="/media/data1/mhkim/FakeAVCeleb_PREPROCESSED/SPECTROGRAM/B/TRAIN", help="path of spectogram (audio)") #TO BE MODIFIED
parser.add_argument('--path_video', '-v',type=str, default="/media/data1/mhkim/FakeAVCeleb_PREPROCESSED/FRAMES_PNG/C/TRAIN", help='path of path of frame (video)')#TO BE MODIFIED
parser.add_argument('--path_audio', '-a',type=str, default="", help="path of spectogram (audio)") #TO BE MODIFIED
parser.add_argument('--path_save', '-sm',type=str, default='./', help='path to save model while training')
parser.add_argument('--num_gpu', '-ng', type=str, default='0', help='excuted gpu number')
parser.add_argument('--val_ratio', '-vr', type=float, default=0.3, help='validation ratio on trainset')
parser.add_argument('--n_early', '-ne', type=int, default=10, help='patient number of early stopping')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)
use_cuda = torch.cuda.is_available()

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint
# new_s = {}
def load_checkpoint(path, model,optimizer,reset_optimizer =False):
    new_s = {}
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if  reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    return model
def TrainXception(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # LIST_SELECT = ('VIDEO' if os.path.exists(args.path_video) else '',
    #                'AUDIO' if os.path.exists(args.path_audio) else '')
    # assert (LIST_SELECT[0]!='' and LIST_SELECT[1]!='', 'At least one path must be typed')

    # tu_video, tu_audio = None, None
    # if args.path_video:
    #     tu_video = (args.path_video)
    # if args.path_audio:
    #     tu_audio = (args.path_audio)

    # for MODE in LIST_SELECT:
    #     train_dir = None
    #     if MODE == 'VIDEO':
    #         train_dir = tu_video
    #     elif MODE == 'AUDIO':
    #         train_dir = tu_audio
        
    #     if train_dir is None:
    #         continue
    
    EPOCHS = 300
    BATCH_SIZE = args.batch_size
    VALID_RATIO = args.val_ratio
    START_LR = args.lr
    PATIENCE_EARLYSTOP = args.n_early
    SAVE_PATH = args.path_save
    pretrained_size = 224
    pretrained_means = [0.4489, 0.3352, 0.3106]  # [0.485, 0.456, 0.406]
    pretrained_stds = [0.2380, 0.1965, 0.1962]  # [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.Resize((pretrained_size, pretrained_size)),        
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                                std=pretrained_stds)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((pretrained_size, pretrained_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                                std=pretrained_stds)
    ])
    # train_data = datasets.ImageFolder(root=train_dir,
    #                                   transform=train_transforms)

    # n_valid_examples = int(len(train_data) * VALID_RATIO)  
    # n_train_examples = len(train_data) - n_valid_examples

    # train_data, valid_data = data.random_split(train_data,
    #                                            [n_train_examples, n_valid_examples])
    # valid_data = copy.deepcopy(valid_data)
    # valid_data.dataset.transform = test_transforms
    # valid_data = datasets.ImageFolder(root='/workspace/FakeAVCeleb/dataset/test',
    #                                   transform=test_transforms)
    train_dataset = FakeAVDataset('train',args,'full')
    test_dataset = FakeAVDataset('test',args,'full')
    # train_dataset = VoxDatasetv2('train',args)
    # test_dataset = VoxDatasetv2('test',args)
    # train_dataset = Dataset('train',args)
    # test_dataset = Dataset('test',args)
    print(f'Number of training examples: {len(train_dataset)}')
    print(f'Number of validation examples: {len(test_dataset)}')

    train_iterator = data.DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=BATCH_SIZE,num_workers = 8 ,drop_last=True)

    valid_iterator = data.DataLoader(test_dataset,
                                        shuffle=True,
                                        batch_size=BATCH_SIZE,num_workers = 8 ,drop_last=True)

    print(f'number of train/val/test loader : {len(train_iterator), len(valid_iterator)}')
    # model = AVModule()
    model = VModule()
#     model = VFTransformer(
#         image_size = 224,
#         patch_size = 28,
#         num_classes = 2,
#         dim = 1024,
#         depth = 4,
#         heads = 16,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1,
#         channels=15,
#         bottles=2,
#         fusionlayer=4
# )
    voicecheckpoint = torch.load('/workspace/Wav2Lip-master/wav2lip_gan.pth')
    s = voicecheckpoint["state_dict"]
    device2 = torch.device('cpu')
    new_s = {}
    for k, v in s.items():
        # print(k)
        # print(k.replace('module.', ''))
        new_s[k.replace('module.', '')] = v
    # modeldict = model.vencoder.state_dict()
    # pretrained_dict = {k: v.to(device2) for k, v in new_s.items() if k in modeldict and modeldict[k].to(device2).shape == new_s[k].shape}
    # modeldict.update(pretrained_dict)
    # model.vencoder.load_state_dict(modeldict)
    # audio_encoder.load_state_dict(new_s,strict=False)
    # device2 = torch.device('cpu')
    # model.fencoder = load_checkpoint('/workspace/AVDet/swintransformer/faceepoch000000295.pth',model.fencoder)
    # model.vencoder = load_checkpoint('/workspace/AVDet/swintransformer/audioepoch000000295.pth',model.vencoder)
    # for key in model.fencoder.state_dict():
    #     print(model.fencoder.state_dict()[key] == new_s[key].to(torch.device('cpu')))
        
    
    
    criterion = nn.CrossEntropyLoss().to(device)
    if len(args.num_gpu) > 1:
        model = nn.DataParallel(model)
    model.to(device)
    scaler = GradScaler()
    # ckp = torch.load('/workspace/AVDet/test2023113/epoch344best_avmodule_av.pt')
    # model.load_state_dict(ckp['state_dict'])
    best_valid_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    # model.fencoder = load_checkpoint('/workspace/AVDet/test2023321/bestfaceepoch000000026.pth',model.module.fencoder,optimizer)
    # for p in model.fencoder.parameters():
    #     p.requires_grad = False
    
    # model.to(device)
    # sepoch = ckp['epoch']
    for epoch in range(161,300):

        start_time = time.monotonic()

        train_loss, train_acc_1, train_acc_5 = train(model, train_iterator,epoch, optimizer, criterion, scaler, device)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)
        if epoch%10 == 0:
            torch.save({'state_dict': model.state_dict(),
                        'best_acc': valid_acc_1,
                        'val_loss': valid_loss,
                        'epoch': epoch,
                        'lr': START_LR,
                        "optimizer": optimizer.state_dict(),
                        'best_acc': valid_acc_1,
                        }, f'{SAVE_PATH}/epoch{epoch}avmodule_av.pt')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'state_dict': model.state_dict(),
                        'best_acc': valid_acc_1,
                        'val_loss': valid_loss,
                        'epoch': epoch,
                        'lr': START_LR,
                        "optimizer": optimizer.state_dict(),
                        'best_acc': valid_acc_1,
                        }, f'{SAVE_PATH}/epoch{epoch}best_avmodule_av.pt')

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1 * 100:6.2f}% | ' \
                f'Train Acc @5: {train_acc_5 * 100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1 * 100:6.2f}% | ' \
                f'Valid Acc @5: {valid_acc_5 * 100:6.2f}%')

        # if early_stopping:
        #     early_stopping(valid_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
if __name__ == "__main__":
    TrainXception(args)