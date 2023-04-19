import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.io.image import read_image
from glob import glob,escape
# from torchcam.utils import overlay_mask
import copy
import time
from torch.cuda.amp import GradScaler
from utils.EarlyStopping import EarlyStopping
from utils.Common_Function import *
# from models import xception_origin
import traceback
import torchaudio
import cv2
import albumentations as A
from os.path import dirname, join, basename, isfile
# from torchcam.methods import GradCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image
from PIL import Image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from  models.syncnet import F_encoder, voice_encoder,SwinF_encoder,AVModule,VModule,VFTransformer,testAVSimModule,testVModule
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
def loadpth(model,load_dir):
    new_s = {}
    s = torch.load(load_dir)['state_dict']
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
def main():
    img_path = '/workspace/FakeAVLab/FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/men/id00032/00028_0_id00860_wavtolip/00000000/00000000.png'
    img = read_image(img_path)
    dir_path = '/workspace/FakeAVLab/FakeAVCeleb/FakeVideo-FakeAudio/Asian (South)/men/id00032/00028_0_id00860_wavtolip/00000000'
    input,mel = read_5batchimgs(dir_path)
    print(input.shape)
    print(mel.shape)
    model = testVModule(dir_path)
    load_dir = '/workspace/AVDet/test2345concat/epoch163best_avmodule_av.pt'
    f_load_dir = '/workspace/AVDet/test2023319sim/bestfaceepoch000000313.pth'
    v_load_dir = '/workspace/AVDet/test2023319sim/bestaudioepoch000000313.pth'
    new_s = {}
    rgb_img = cv2.imread(img_path)  # 1是读取rgb
    rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img) / 255
    rgb_img = cv2.resize(rgb_img,(224,224))
    # cv2.imwrite(f'cam_dog.jpg', rgb_img)
    s = torch.load(load_dir)['state_dict']
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    # model.load_state_dict(new_s)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loadpth( model.fencoder,f_load_dir)
    loadpth( model.vencoder,v_load_dir)
    # model.to(device)
    # model.eval()
    input_tensor = read_img_astensor(img_path)
    input_tensor = input_tensor.to(device)
    # print(model.vencoder.audio_encoder[-1].conv_block)
    # for index ,(name, param) in enumerate(model.vencoder.named_parameters()):
    #     print( str(index) + " " +name)
    # print(model.vencoder.audio_encoder.)
    cam_extractor =GradCAM(model,target_layers = [model.fencoder.face_encoder.conv4])
    print(model.vencoder.audio_encoder)
  # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    print(out.shape)
    print(out)
    targets = None
    # print(model)
    # print(out)
    # Retrieve the CAM by passing the class index and the model output
    grayscale_cam = cam_extractor(input_tensor=input, targets=targets)
    print(grayscale_cam.shape)
    print(grayscale_cam)
    grayscale_cam = grayscale_cam[0, :]
    heatmap = cv2.applyColorMap(np.uint8(255 *  grayscale_cam), cv2.COLORMAP_JET)
    # heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True,image_weight=0)
    visualization = cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
    plt.imshow(visualization)
    plt.savefig('testcamface.jpg')
main()