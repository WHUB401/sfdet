import torch
from torch import nn,einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_




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
            print(bottles.shape)
            print(y.shape)
            input_y = torch.cat((y,bottles),dim=1)
            output_y = atrans(input_y)
            y = output_y[:,-self.n:]
            bottles = output_y[:,self.n:]
        return x,y,bottles

class ViT(nn.Module):
    def __init__(self,*,image_size,patch_size,num_classes,dim,depth,heads,mlp_dim,pool='mean',channels=3,dim_head=64,dropout=0.,emb_dropout=0.):
        super(ViT,self).__init__()
        image_height,image_width = pair(image_size)
        patch_height,patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        # self.to_patch_embedding = nn.Sequential(
        #     # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 = patch_height,p2 = patch_width),nn.Linear(patch_dim,dim))
        # )
        
        # self.pos_embedding = nn.Parameter(torch.randn(1,num_patches+1,dim))
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim,depth,heads,dim_head,mlp_dim,dropout)
        
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_classes))
        
    def forward(self,img):
        # x = self.to_patch_embedding(img)
        x = self.proj(img).flatten(2).transpose(1, 2)
        b,n,_ = x.shape
        
        # cls_tokens = repeat(self.cls_token,'() n d -> b n d',b=b)
        # x = torch.cat((cls_tokens,x),dim=1)
        x += self.absolute_pos_embed[:,:(n)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:,0]
        x = self.to_latent(x)
        return self.mlp_head(x)

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
        self.dropout = nn.Dropout(emb_dropout)
        
        self.atransformer = Transformer(dim,depth,heads,dim_head,mlp_dim,dropout)
        self.vtransformer = Transformer(dim,depth,heads,dim_head,mlp_dim,dropout)
        self.fusionlayer = VFFusion(dim,fusionlayer,heads,dim_head,mlp_dim,dropout)
        self.bottleneck = nn.Parameter(torch.randn(1,bottles,dim))
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_classes))
        
    def forward(self,img,mel):
        # x = self.to_patch_embedding(img)
        print(mel.shape)
        x = self.proj(img).flatten(2).transpose(1, 2)
        b,n,_ = x.shape
        y = self.melproj(mel).flatten(2).transpose(1, 2)
        b1,n1,_ = y.shape
        print(y.shape)
        visual_cls_tokens = repeat(self.visualcls_token,'() n d -> b n d',b=b)
        x = torch.cat((visual_cls_tokens,x),dim=1)
        audio_cls_token = repeat(self.visualcls_token,'() n d -> b n d',b=b1)
        fs_tokens = repeat(self.bottleneck,'() n d -> b n d',b=b)
        y = torch.cat((audio_cls_token,y),dim=1)
        x += self.absolute_pos_embed[:,:(n+1)]
        y += self.audio_pos_embed[:,:(n1+1)]
        # x = self.dropout(x)
        # y = self.dropout(y)
        x = self.vtransformer(x)
        x,y,fs_tokens = self.fusionlayer(x,y,fs_tokens)
        v_embedding = x[:,0]
        a_embedding = y[:,0]
        x = self.to_latent(v_embedding)
        logits = self.mlp_head(x)
        return v_embedding,a_embedding,logits