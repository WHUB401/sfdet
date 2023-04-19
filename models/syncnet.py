import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from .conv import Conv2d
from .SwinTransformer import ViT
import torch
from torch import nn,einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
import traceback
import torchaudio
import cv2
import os
from os.path import dirname, join, basename, isfile
from glob import glob,escape
import numpy as np
def get_frame_id(frame):
        return int(basename(frame).split('.')[0])

def get_window(start_frame):
    start_id = get_frame_id(start_frame)
    vidname = dirname(start_frame)

    window_fnames = []
    for frame_id in range(start_id, start_id + 5):
        frame = join(vidname, '{:08d}.png'.format(frame_id))
        if not isfile(frame):
            return None
        window_fnames.append(frame)
    return window_fnames
def read_img_astensor(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = img/255
    img = img.transpose(2, 0, 1)
    img = torch.FloatTensor(img).unsqueeze(0)
    return img

def crop_audio_window(spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = get_frame_id(start_frame)+1
        start_idx = int(80. * (start_frame_num / float(25)))

        end_idx = start_idx + 16
        maxlen = spec.shape[0]
        if end_idx > maxlen:
            end_idx = maxlen
            start_idx = end_idx - 16
        return spec[start_idx : end_idx, :]
def read_5batchimgs(dir_path):
    img_names = list(glob(join(dir_path.rstrip('\n'), '*.png')))
    while 1:
            img_names.sort()
      
            img_name = img_names[0]
          
            window_fnames = get_window(img_name)
            # print(img_name)
            if window_fnames is None:
                continue
            else:
                break
    window = []
        # all_read = True
    for fname in window_fnames:
        img = cv2.imread(fname)
        if img is None:
            all_read = False
            break
        # try:
        img = cv2.resize(img, (224, 224))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(1)
        # img = trans(image = img)['image']
        # print(2)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # except Exception as e:
        #     all_read = False
        #     break

        window.append(img)
    audio_path = os.path.dirname(dir_path)
    try:
            wavpath = join(audio_path, "audio.wav")
            waveform,sample_rate = torchaudio.load(wavpath)
            if (waveform.shape[0] == 2):
                waveform = waveform[:1,:]
            wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

            orig_mel = wav.squeeze(0).T
            
            print('hahhaha')
    except Exception as e:
        print(traceback.format_exc())
        # print(3)
    mel = crop_audio_window(orig_mel, img_name)
    x = np.concatenate(window, axis=2) / 255.
    x = x.transpose(2, 0, 1)
    x = torch.FloatTensor(x)
    mel = mel.T.unsqueeze(0)
    return x ,mel

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}
def InitXception(model=None, num_class=2, pretrained=True):
    if(not model) :
        assert ("model is empty(None)")
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, num_class)
    return model
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x
class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=2):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        # self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,1024,3,1,1)
        self.bn4 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(1024, 512)
    

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
#         print(input.size())

        x = self.conv1(input) #(32, 299, 299)
#         print(x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x) #(64, 299, 299)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x) #(1024, 299, 299)

        x = self.conv3(x) #(1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x) #(2048, 299, 299)
#         print(x.size())


        x = self.bn4(x)
#         print(x.size())

        return x


    def GetEachFeatures(self, input):
#         print(input.size())

        list_feat = []
        feat1 = self.conv1(input) #(32, 299, 299)
#         print(x.size())
        feat1 = self.bn1(feat1)
        feat1 = self.relu(feat1)
        
        list_feat.append(feat1)
        feat2 = self.conv2(feat1) #(64, 299, 299)
        feat2 = self.bn2(feat2)
        feat3 = self.relu(feat2)
        list_feat.append(feat3)

        feat4 = self.block1(feat3)
        feat5 = self.block2(feat4)
        feat6 = self.block3(feat5)
        feat7 = self.block4(feat6)
        feat8 = self.block5(feat7)
        feat9 = self.block6(feat8)
        feat10 = self.block7(feat9)
        feat11 = self.block8(feat10)
        feat12 = self.block9(feat11)
        feat13 = self.block10(feat12)
        feat14 = self.block11(feat13)
        feat15 = self.block12(feat14) #(1024, 299, 299)

        list_feat.append(feat4)
        list_feat.append(feat5)
        list_feat.append(feat6)
        list_feat.append(feat7)
        list_feat.append(feat8)
        list_feat.append(feat9)
        list_feat.append(feat10)
        list_feat.append(feat11)
        list_feat.append(feat12)
        list_feat.append(feat13)
        list_feat.append(feat14)
        list_feat.append(feat15)

        
        feat16 = self.conv3(feat15) #(1536, 299, 299)
        feat16 = self.bn3(feat16)
        feat16 = self.relu(feat16)
        list_feat.append(feat16)

        feat17 = self.conv4(feat16) #(2048, 299, 299)
#         print(x.size())


        feat17 = self.bn4(feat17)
#         print(x.size())
        list_feat.append(feat17)
        return list_feat

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        feat = self.features(input)
        x = self.logits(feat)
        return x
class Xception3(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=2):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception3, self).__init__()
        self.num_classes = num_classes

        # self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,1024,3,1,1)
        self.bn4 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(1024, 512)
    

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
#         print(input.size())

        x = self.conv1(input) #(32, 299, 299)
#         print(x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x) #(64, 299, 299)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x) #(1024, 299, 299)

        x = self.conv3(x) #(1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x) #(2048, 299, 299)
#         print(x.size())


        x = self.bn4(x)
#         print(x.size())

        return x


    def GetEachFeatures(self, input):
#         print(input.size())

        list_feat = []
        feat1 = self.conv1(input) #(32, 299, 299)
#         print(x.size())
        feat1 = self.bn1(feat1)
        feat1 = self.relu(feat1)
        
        list_feat.append(feat1)
        feat2 = self.conv2(feat1) #(64, 299, 299)
        feat2 = self.bn2(feat2)
        feat3 = self.relu(feat2)
        list_feat.append(feat3)

        feat4 = self.block1(feat3)
        feat5 = self.block2(feat4)
        feat6 = self.block3(feat5)
        feat7 = self.block4(feat6)
        feat8 = self.block5(feat7)
        feat9 = self.block6(feat8)
        feat10 = self.block7(feat9)
        feat11 = self.block8(feat10)
        feat12 = self.block9(feat11)
        feat13 = self.block10(feat12)
        feat14 = self.block11(feat13)
        feat15 = self.block12(feat14) #(1024, 299, 299)

        list_feat.append(feat4)
        list_feat.append(feat5)
        list_feat.append(feat6)
        list_feat.append(feat7)
        list_feat.append(feat8)
        list_feat.append(feat9)
        list_feat.append(feat10)
        list_feat.append(feat11)
        list_feat.append(feat12)
        list_feat.append(feat13)
        list_feat.append(feat14)
        list_feat.append(feat15)

        
        feat16 = self.conv3(feat15) #(1536, 299, 299)
        feat16 = self.bn3(feat16)
        feat16 = self.relu(feat16)
        list_feat.append(feat16)

        feat17 = self.conv4(feat16) #(2048, 299, 299)
#         print(x.size())


        feat17 = self.bn4(feat17)
#         print(x.size())
        list_feat.append(feat17)
        return list_feat

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        feat = self.features(input)
        x = self.logits(feat)
        return x
    
def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    # model.last_linear = model.fc
    # del model.fc
    return model
def xception3(num_classes=1000, pretrained='imagenet'):
    model = Xception3(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    # model.last_linear = model.fc
    # del model.fc
    return model
# class SyncNet_color(nn.Module):
#     def __init__(self):
#         super(SyncNet_color, self).__init__()

#         self.face_encoder = xception(num_classes=2, pretrained='')

#         self.audio_encoder = nn.Sequential(
#             Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
#             Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
#             Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
#             Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

#     def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
#         face_embedding = self.face_encoder(face_sequences)
#         audio_embedding = self.audio_encoder(audio_sequences)

#         audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
#         face_embedding = face_embedding.view(face_embedding.size(0), -1)

#         audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
#         face_embedding = F.normalize(face_embedding, p=2, dim=1)


        # return audio_embedding, face_embedding
def pair(t):
    return t if isinstance(t,tuple) else (t,t)

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super(PreNorm,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)
    

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.):
        super(Attention,self).__init__()
        inner_dim = dim_head*heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim,inner_dim*3,bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()
        
    def forward(self,x):
        b,n,_,h = *x.shape,self.heads
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=h),qkv)
        dots = einsum('b h i d,b h j d -> b h i j',q,k)*self.scale
        attn = self.attend(dots)
        out = einsum('b h i j,b h j d -> b h i d',attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')
        return self.to_out(out)
# class Attention(nn.Module):
#     def __init__(self,dim,heads=16,dim_head=64,dropout=0.):
#         super(Attention,self).__init__()
#         inner_dim = dim_head*heads
#         project_out = not (heads == 1 and dim_head == dim)
#         # self.attlayer = torch.nn.MultiheadAttention(embed_dim=dim,num_heads=16,dropout=0.1)
#         self.heads = heads
#         self.scale = dim_head ** -0.5
        
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(dim,inner_dim*3,bias=False)
        
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim,dim),
#             nn.Dropout(dropout)) if project_out else nn.Identity()
        
#     def forward(self,x):
#         b,n,_,h = *x.shape,self.heads
#         qkv = self.to_qkv(x).chunk(3,dim=-1)
#         q,k,v = map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=h),qkv)
#         print(q.shape)
#         dots = einsum('b h i d,b h j d -> b h i j',q,k)*self.scale
#         attn = self.attend(dots)
#         out = einsum('b h i j,b h j d -> b h i d',attn,v)
#         out = rearrange(out,'b h n d -> b n (h d)')
#         x ,_= self.attlayer(x,x,x)
#         return x
    
    
class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,dim_head,mlp_dim,dropout=0.):
        super(Transformer,self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,Attention(dim,heads=heads,dim_head=dim_head,dropout=dropout)),
                PreNorm(dim,FeedForward(dim,mlp_dim,dropout=dropout))]))
    def forward(self,x):
        for attn,ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class F_encoder(nn.Module):
    def __init__(self):
        super(F_encoder, self).__init__()
        self.face_encoder = xception(num_classes=2, pretrained='')
    def forward(self, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        return face_embedding
class SwinF_encoder(nn.Module):
    def __init__(self):
        super(SwinF_encoder, self).__init__()
        self.face_encoder = ViT(
        image_size = 224,
        patch_size = 28,
        num_classes = 512,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    def forward(self, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        return face_embedding
class voice_encoder(nn.Module):
    def __init__(self):
        super(voice_encoder, self).__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences): # audio_sequences := (B, dim, T)
        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        return audio_embedding
class AVModule(nn.Module):
    def __init__(self):
        super(AVModule, self).__init__()
        self.vencoder = voice_encoder()
        self.fencoder = F_encoder()
        self.fc = nn.Linear(1024,2)
        self.transformer = Transformer(512,6,16,64,2048,dropout=0.)
    def forward(self,f_embedding,vembedding):
        f = self.fencoder(f_embedding)
        v = self.vencoder(vembedding)
        f = f.unsqueeze(dim = 1)
        v = v.unsqueeze(dim = 1)
        x = torch.concat([f,v] ,dim = 1)
        x = self.transformer(x)
        x = x.view(x.shape[0],-1)
        out = self.fc(x)
        return out
class VModule(nn.Module):
    def __init__(self):
        super(VModule, self).__init__()
        self.vencoder = voice_encoder()
        self.fencoder = F_encoder()
        self.fc = nn.Linear(1024,2)
        # self.transformer = Transformer(512,6,16,64,2048,dropout=0.)
    def forward(self,f_embedding,vembedding):
        f = self.fencoder(f_embedding)
        v = self.vencoder(vembedding)
        # print(v.shape)
        f = f.unsqueeze(dim = 1)
        v = v.unsqueeze(dim = 1)
        x = torch.concat([f,v] ,dim = 1)
        # print(x.shape)
        # x = self.transformer(x)
        x = x.view(x.shape[0],-1)
        out = self.fc(x)
        return out
class testVModule(nn.Module):
    def __init__(self,path):
        super(testVModule, self).__init__()
        self.vencoder = voice_encoder()
        self.fencoder = F_encoder()
        self.f_embedding,self.vembedding = read_5batchimgs(path)
        self.fc = nn.Linear(1024,2)
        # self.transformer = Transformer(512,6,16,64,2048,dropout=0.)
    def forward(self,f_embedding):
        f = self.fencoder(self.f_embedding.unsqueeze(0))
        v = self.vencoder(self.vembedding.unsqueeze(0))
        # print(v.shape)
        f = f.unsqueeze(dim = 1)
        v = v.unsqueeze(dim = 1)
        x = torch.concat([f,v] ,dim = 1)
        # print(x.shape)
        # x = self.transformer(x)
        x = x.view(x.shape[0],-1)
        out = self.fc(x)
        return out
class testAVSimModule(nn.Module):
    def __init__(self,path):
        super(testAVSimModule, self).__init__()
        self.vencoder = voice_encoder()
        self.fencoder = F_encoder()
        self.f_embedding,self.vembedding = read_5batchimgs(path)
        self.fc = nn.Linear(1024,2)
        # self.transformer = Transformer(512,6,16,64,2048,dropout=0.)
    def forward(self,f_embedding):
        f = self.fencoder(self.f_embedding.unsqueeze(0))
        v = self.vencoder(self.vembedding.unsqueeze(0))
        # print(v.shape)
        cos = nn.functional.cosine_similarity(f,v,dim=1)
        cos = cos.unsqueeze(0)
        # f = f.unsqueeze(dim = 1)
        # v = v.unsqueeze(dim = 1)
        # x = torch.concat([f,v] ,dim = 1)
        # # print(x.shape)
        # # x = self.transformer(x)
        # x = x.view(x.shape[0],-1)
        # out = self.fc(x)
        return cos

class VFFusion(nn.Module):
    def __init__(self,dim,depth,heads,dim_head,mlp_dim,dropout=0.,bottles = 2):
        super(VFFusion,self).__init__()
        self.n = bottles
        self.fusionblocks = nn.ModuleList([])
        for _ in range(depth):
            self.fusionblocks.append(
                nn.ModuleList(
                    [
                        Transformer(dim,1,heads,dim_head,mlp_dim,dropout),
                        Transformer(dim,1,heads,dim_head,mlp_dim,dropout)
                    ]
                )
            )
    def forward(self,x,y,bottles):
        for vtrans,atrans in self.fusionblocks:
            input_x =  torch.cat((x,bottles),dim=1)
            output_x = vtrans(input_x)
            x = output_x[:,:-self.n]
            bottles = output_x[:,-self.n:]
            # print(bottles.shape)
            # print(y.shape)
            input_y = torch.cat((y,bottles),dim=1)
            output_y = atrans(input_y)
            y = output_y[:,-self.n:]
            bottles = output_y[:,self.n:]
        return x,y,bottles
class VFTransformer(nn.Module):
    def __init__(self,*,image_size,patch_size,num_classes,dim,depth,heads,mlp_dim,pool='mean',fusionlayer = 4,bottles = 2,channels=3,dim_head=64,dropout=0.,emb_dropout=0.):
        super(VFTransformer,self).__init__()
        image_height,image_width = pair(image_size)
        patch_height,patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.melproj = nn.Conv2d(1, dim, kernel_size=16, stride=16)
        # self.melproj = nn.Linear()
        # self.to_patch_embedding = nn.Sequential(
        #     # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 = patch_height,p2 = patch_width),nn.Linear(patch_dim,dim))
        # )
        
        # self.pos_embedding = nn.Parameter(torch.randn(1,num_patches+1,dim))
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.audio_pos_embed = nn.Parameter(torch.zeros(1, 6, dim)) #audiopathces + cls token
        trunc_normal_(self.absolute_pos_embed, std=.02)
        trunc_normal_(self.audio_pos_embed, std=.02)
        self.visualcls_token = nn.Parameter(torch.randn(1,1,dim))
        self.audiocls_token =  nn.Parameter(torch.randn(1,1,dim))
        self.avcls_token =  nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.atransformer = Transformer(dim,2,heads,dim_head,mlp_dim,dropout)
        self.vtransformer = Transformer(dim,2,heads,dim_head,mlp_dim,dropout)
        self.avtrans = Transformer(dim,2,heads,dim_head,mlp_dim,dropout)
        self.fusionlayer = VFFusion(dim,fusionlayer,heads,dim_head,mlp_dim,dropout)
        self.bottleneck = nn.Parameter(torch.randn(1,bottles,dim))
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_classes))
        
    def forward(self,img,mel):
        # x = self.to_patch_embedding(img)
        # print(mel.shape)
        x = self.proj(img).flatten(2).transpose(1, 2)
        b,n,_ = x.shape
        y = self.melproj(mel).flatten(2).transpose(1, 2)
        b1,n1,_ = y.shape
        # print(y.shape)
        visual_cls_tokens = repeat(self.visualcls_token,'() n d -> b n d',b=b)
        x = torch.cat((visual_cls_tokens,x),dim=1)
        audio_cls_token = repeat(self.visualcls_token,'() n d -> b n d',b=b1)
        avtoken = repeat(self.avcls_token,'() n d -> b n d',b=b)
        fs_tokens = repeat(self.bottleneck,'() n d -> b n d',b=b)
        y = torch.cat((audio_cls_token,y),dim=1)
        x += self.absolute_pos_embed[:,:(n+1)]
        y += self.audio_pos_embed[:,:(n1+1)]
        x = self.dropout(x)
        y = self.dropout(y)
        x = self.vtransformer(x)
        y = self.atransformer(y)
        
        x,y,fs_tokens = self.fusionlayer(x,y,fs_tokens)
        v_embedding = x[:,0,:].unsqueeze(1)
        a_embedding = y[:,0,:].unsqueeze(1)
        # print(v_embedding.shape)
        vf = torch.cat((avtoken,v_embedding ),dim = 1)
        vf = torch.cat((vf,a_embedding),dim = 1)
        vf = self.avtrans(vf)
        vftoken = vf[:,0]
        # x = self.to_latent(v_embedding)
        logits = self.mlp_head(vftoken)
        return logits