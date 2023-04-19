import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import torchaudio
import numpy as np
import numpy as np
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from glob import glob,escape
from random import sample
import albumentations as A
# import glob
import traceback
def path_remake(path):
    return path.replace(' ', '\ ').replace('(','\(').replace(')','\)')
import os, random, cv2, argparse
cv2.setNumThreads(0)
from hparams import hparams, get_image_list,get_image_list2,get_image_list3
syncnet_T = 5
syncnet_mel_step_size = 16
class FakeAVDataset(object):
    def __init__(self, split,args,text):
        self.all_videos = get_image_list3(args.data_root, split,text)
        self.data_root = args.data_root
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{:08d}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)+1
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size
        maxlen = spec.shape[0]
        if end_idx > maxlen:
            end_idx = maxlen
            start_idx = end_idx - syncnet_mel_step_size
        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        trans = A.Compose([A.ImageCompression(quality_lower=60,quality_upper=70,p=1.0)])
        vidname = self.all_videos[idx]
        id = vidname.split('/')[0]
        if 'RealVideo-RealAudio' in vidname:
            y = torch.ones(1).float()
        else:
            y = torch.zeros(1).float()
        vidname = os.path.join(self.data_root, vidname)
        # print(vidname)
        img_names = list(glob(join(vidname.rstrip('\n'), '*.png')))
        # print(len(img_names))
        if not (isinstance(img_names,list)):
            print(img_names)
            print(vidname)
        if (len(img_names) == 0):
            print(vidname)
        # print('hahah')
        # print(img_names)
        # a = list(glob(join('/workspace/FakeAVLab/FakeAVCeleb/FakeVideo-RealAudio/African/men/id00076/00109_1/00000087','*.png')))
        while 1:
            img_names.sort()
      
            img_name = img_names[0]
          
            window_fnames = self.get_window(img_name)
            # print(img_name)
            if window_fnames is None:
                continue
            else:
                break
        # print(1)
       
        # print(2)
        # print('111')
        # if random.choice([True, False]):
        #     y = torch.ones(1).float()
        #     window_fnames = window_fnames
        # else:
        #     y = torch.zeros(1).float()
        #     window_fnames = wrong_window_fnames
        # print('hahhaha')
        window = []
        all_read = True
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                all_read = False
                break
            # try:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(1)
            img = trans(image = img)['image']
            # print(2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # except Exception as e:
            #     all_read = False
            #     break

            window.append(img)
            
        audio_path = os.path.dirname(vidname)
        try:
            wavpath = join(audio_path, "audio.wav")
            waveform,sample_rate = torchaudio.load(wavpath)
            if (waveform.shape[0] == 2):
                waveform = waveform[:1,:]
            wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

            orig_mel = wav.squeeze(0).T
            # print('hahhaha')
        except Exception as e:
            print(traceback.format_exc())
        # print(3)
        mel = self.crop_audio_window(orig_mel, img_name)
        # not_sync_mel = self.crop_audio_window(orig_mel, wrong_img_name)
       
        # try:
        #     wavpath = join(wrong_id_vidname, "audio.wav")
        #     # print(wavpath)
        #     waveform,sample_rate = torchaudio.load(wavpath)
        #     if(waveform==None):
        #         print(wavpath)
        #     wrong_id_wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

        #     wrong_id_mel = wrong_id_wav.squeeze(0).T
        # except Exception as e:
        #     print(traceback.format_exc())
        # H x W x 3 * T
        # print(5)
        # wrong_id_startidx = random.randint(0,wrong_id_mel.shape[0]-16)
        # wrong_id_mel = wrong_id_mel[wrong_id_startidx:wrong_id_startidx+syncnet_mel_step_size,:]
        x = np.concatenate(window, axis=2) / 255.
        x = x.transpose(2, 0, 1)
        x = torch.FloatTensor(x)
        mel = mel.T.unsqueeze(0)
        return x , mel ,y
        # not_sync_mel = not_sync_mel.T.unsqueeze(0)
        # wrong_id_mel = wrong_id_mel.T.unsqueeze(0)
        # print('hahah')
        # return x, mel,not_sync_mel,wrong_id_mel
class FakeAVDataset2(object):
    def __init__(self, split,args,text,num_n = 4):
        self.realall_videos,self.fakeall_videos = get_image_list2(args.data_root, split,text)
        self.data_root = args.data_root
        self.num_n = num_n
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{:08d}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)+1
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size
        maxlen = spec.shape[0]
        if end_idx > maxlen:
            end_idx = maxlen
            start_idx = end_idx - syncnet_mel_step_size
        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.realall_videos)

    def __getitem__(self, idx):
        fake_v_list = sample(self.fakeall_videos,self.num_n)
        x = []
        vidname = self.realall_videos[idx]
        vidname = os.path.join(self.data_root, vidname)
        # print(vidname)
        img_names = list(glob(join(vidname.rstrip('\n'), '*.png')))
        # print(len(img_names))
        # if not (isinstance(img_names,list)):
        #     print(img_names)
        #     print(vidname)
        # if (len(img_names) == 0):
        #     print(vidname)
        # print('hahah')
        # print(img_names)
        # a = list(glob(join('/workspace/FakeAVLab/FakeAVCeleb/FakeVideo-RealAudio/African/men/id00076/00109_1/00000087','*.png')))
        while 1:
            img_names.sort()
      
            img_name = img_names[0]
          
            window_fnames = self.get_window(img_name)
            # print(img_name)
            if window_fnames is None:
                continue
            else:
                break
        # print(1)
       
        # print(2)
        # print('111')
        # if random.choice([True, False]):
        #     y = torch.ones(1).float()
        #     window_fnames = window_fnames
        # else:
        #     y = torch.zeros(1).float()
        #     window_fnames = wrong_window_fnames
        # print('hahhaha')
        all_samples = []
        window = []
        all_read = True
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                all_read = False
                break
            try:
                img = cv2.resize(img, (224, 224))
            except Exception as e:
                all_read = False
                break

            window.append(img)
            
        all_samples.append(window)
        # print(len(fake_v_list))
        for video in fake_v_list:
            # print(video)
            video = os.path.join(self.data_root, video)
            img_names = list(glob(join(video.strip().rstrip('\n'), '*.png')))
        # print(len(img_names))
            if not (isinstance(img_names,list)):
                print(img_names)
                print(video)
            if (len(img_names) == 0):
                print(video)
            while 1:
                img_names.sort()
                img_name = img_names[0]
                window_fnames = self.get_window(img_name)
                # print(img_name)
                if window_fnames is None:
                    continue
                else:
                    break
                
            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (224, 224))
                except Exception as e:
                    all_read = False
                    break
                window.append(img)
                
        
            all_samples.append(window)
        # print(len(all_samples))
        all_mels = []
        audio_path = os.path.dirname(vidname)
        try:
            wavpath = join(audio_path, "audio.wav")
            waveform,sample_rate = torchaudio.load(wavpath)
            if (waveform.shape[0] == 2):
                waveform = waveform[:1,:]
            wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

            orig_mel = wav.squeeze(0).T
            # print('hahhaha')
        except Exception as e:
            print(traceback.format_exc())
        # print(3)
        mel = self.crop_audio_window(orig_mel, img_name)
        all_mels.append(mel)
        for v in fake_v_list:
            v = os.path.join(self.data_root,v)
            audio_path = os.path.dirname(v)
            try:
                wavpath = join(audio_path, "audio.wav")
                waveform,sample_rate = torchaudio.load(wavpath)
                if (waveform.shape[0] == 2):
                    waveform = waveform[:1,:]
                wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

                orig_mel = wav.squeeze(0).T
                # print('hahhaha')
            except Exception as e:
                print(traceback.format_exc())
            # print(3)
            mel = self.crop_audio_window(orig_mel, img_name)
            all_mels.append(mel)
            
        # not_sync_mel = self.crop_audio_window(orig_mel, wrong_img_name)
       
        # try:
        #     wavpath = join(wrong_id_vidname, "audio.wav")
        #     # print(wavpath)
        #     waveform,sample_rate = torchaudio.load(wavpath)
        #     if(waveform==None):
        #         print(wavpath)
        #     wrong_id_wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

        #     wrong_id_mel = wrong_id_wav.squeeze(0).T
        # except Exception as e:
        #     print(traceback.format_exc())
        # H x W x 3 * T
        # print(5)
        # wrong_id_startidx = random.randint(0,wrong_id_mel.shape[0]-16)
        # wrong_id_mel = wrong_id_mel[wrong_id_startidx:wrong_id_startidx+syncnet_mel_step_size,:]
        x = []
        # print(len(all_samples))
        for window in all_samples:
            i = np.concatenate(window, axis=2) / 255.
            i = i.transpose(2, 0, 1)
            i = torch.FloatTensor(i)
            x.append(i)
        x = torch.stack(x,0)
        mels = []
        for mel in all_mels:
            mel = mel.T.unsqueeze(0)
            mels.append(mel)
        mel = torch.stack(mels,0)
        # print(x.shape)
        # print(mel.shape)
        return x , mel 
class VoxDataset(object):
    def __init__(self, split,args,wrong_id_nums = 4,not_sync_nums = 0):
        self.notsyncnums = not_sync_nums
        self.wrongid = wrong_id_nums
        self.all_videos = get_image_list(args.data_root, split)
        self.data_root = args.data_root
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{:08d}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)+1
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size
        maxlen = spec.shape[0]
        if end_idx > maxlen:
            end_idx = maxlen
            start_idx = end_idx - syncnet_mel_step_size
        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        vidname = self.all_videos[idx]
        id = vidname.split('/')[0]
        vidname = os.path.join(self.data_root, vidname)
        img_names = list(glob(join(vidname, '*.png')))
        # print(vidname)
        # print('hahah')
        # print(img_names)
        while 1:
            img_name = random.choice(img_names)
            window_fnames = self.get_window(img_name)
            # print(img_name)
            if window_fnames is None:
                continue
            else:
                break
        # print(1)
        wrong_img_name123 = []
        wrong_img_nums = 0
        while 1:
            if wrong_img_nums >= self.notsyncnums:
                    break
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)
            wrong_window_fnames = self.get_window(wrong_img_name)
            # print('222')
            if wrong_window_fnames is None:
                continue
            else:
                wrong_img_name123.append(wrong_window_fnames)
                wrong_img_nums = wrong_img_nums + 1
                if wrong_img_nums >= self.notsyncnums:
                    break
        wrong_windows = []
        for wrong_window_fnames in wrong_img_name123:
            if len(wrong_img_name123) ==0:
                break
            wrong_window = []
            for fname in wrong_window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (224, 224))
                except Exception as e:
                    all_read = False
                    break

                wrong_window.append(img)
            wrong_windows.append(wrong_window)
        # print(2)
        # print('111')
        # if random.choice([True, False]):
        #     y = torch.ones(1).float()
        #     window_fnames = window_fnames
        # else:
        #     y = torch.zeros(1).float()
        #     window_fnames = wrong_window_fnames
        # print('hahhaha')
        window = []
        all_read = True
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                all_read = False
                break
            try:
                img = cv2.resize(img, (224, 224))
            except Exception as e:
                all_read = False
                break

            window.append(img)
        
        try:
            wavpath = join(vidname, "audio.wav")
            waveform,sample_rate = torchaudio.load(wavpath)
            if (waveform.shape[0] == 2):
                waveform = waveform[:1,:]
            wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

            orig_mel = wav.squeeze(0).T
            # print('hahhaha')
        except Exception as e:
            print(traceback.format_exc())
        # print(3)
        mel = self.crop_audio_window(orig_mel, img_name)
        # not_sync_mel = self.crop_audio_window(orig_mel, wrong_img_name)
        wrong_id = []
        wrong_id_nums = 0
        while 1:
            if wrong_id_nums >= self.wrongid:
                    break
            wrong_id_vidname = random.choice(self.all_videos)
            # print(self.all_videos[0])
            # print(id)
            if wrong_id_vidname.split('/')[0] == id:
                continue
            else:

        # print('hahahah')
        # print(4)
                wrong_id_vidname = os.path.join(self.data_root, wrong_id_vidname)
                wrong_id_img_names = list(glob(join(wrong_id_vidname, '*.png')))
                while 1:
                    wrong_id_img_name = random.choice(wrong_id_img_names)
                    wrong_id_window_fnames = self.get_window(wrong_id_img_name)
                    # print(img_name)
                    if wrong_id_window_fnames is None:
                        continue
                    else:
                        break
                wrong_id.append(wrong_id_window_fnames)
                wrong_id_nums = wrong_id_nums + 1
                if wrong_id_nums >= self.wrongid:
                    break
        wrong_id_windows = []
        for wrong_id_window_fnames in wrong_id:
            if len(wrong_id) ==0:
                break
            wrong_id_window = []
            for fname in wrong_id_window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (224, 224))
                except Exception as e:
                    all_read = False
                    break

                wrong_id_window.append(img)
            wrong_id_windows.append(wrong_id_window)
        # try:
        #     wavpath = join(wrong_id_vidname, "audio.wav")
        #     # print(wavpath)
        #     waveform,sample_rate = torchaudio.load(wavpath)
        #     if(waveform==None):
        #         print(wavpath)
        #     wrong_id_wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

        #     wrong_id_mel = wrong_id_wav.squeeze(0).T
        # except Exception as e:
        #     print(traceback.format_exc())
        # H x W x 3 * T
        # print(5)
        # wrong_id_startidx = random.randint(0,wrong_id_mel.shape[0]-16)
        # wrong_id_mel = wrong_id_mel[wrong_id_startidx:wrong_id_startidx+syncnet_mel_step_size,:]
        x = np.concatenate(window, axis=2) / 255.
        x = x.transpose(2, 0, 1)
        all = []
        x = torch.FloatTensor(x)
        all.append(x)
        for y in wrong_windows:
            if y == None:
                break
            y = np.concatenate(wrong_window, axis=2) / 255.
            y = y.transpose(2, 0, 1)
            y = torch.FloatTensor(y)
            all.append(y)
        for z in wrong_id_windows:
            if z == None:
                break
            z = np.concatenate(wrong_id_window, axis=2) / 255.
            z = z.transpose(2, 0, 1)
            z = torch.FloatTensor(z)
            all.append(z)
        mel = mel.T.unsqueeze(0)
        x = torch.stack(all,0)
        # print(x.shape)
        return x,mel
    
class VoxDatasetv2(object):
    def __init__(self, split,args):
        self.all_videos = get_image_list(args.data_root, split)
        self.data_root = args.data_root
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{:08d}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)+1
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size
        maxlen = spec.shape[0]
        if end_idx > maxlen:
            end_idx = maxlen
            start_idx = end_idx - syncnet_mel_step_size
        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        vidname = self.all_videos[idx]
        id = vidname.split('/')[0]
        vidname = os.path.join(self.data_root, vidname)
        img_names = list(glob(join(vidname, '*.png')))
        img_names.sort()
        if(len(img_names) == 0):
            print(vidname)
        # print('hahah')
        # print(img_names)
        while 1:
            img_name = random.choice(img_names)
            window_fnames = self.get_window(img_name)
            # print(img_name)
            if window_fnames is None:
                # print(1)
                continue
            else:
                break
        while 1:
            wrong_id_vidname = random.choice(self.all_videos)
            # print(self.all_videos[0])
            # print(id)
            if wrong_id_vidname.split('/')[0] == id:
                # print(1)
                continue
            else:
                break
        # print('hahahah')
        # print(4)
        wrong_id_vidname = os.path.join(self.data_root, wrong_id_vidname)
        wrong_id_img_names = list(glob(join(wrong_id_vidname, '*.png')))
        while 1:
            wrong_id_img_name = random.choice(wrong_id_img_names)
            wrong_id_window_fnames = self.get_window(wrong_id_img_name)
            # print(img_name)
            if wrong_id_window_fnames is None:
                # print(wrong_id_img_name)
            
                continue
            else:
                break
        # print(1)
        while 1:
            d = os.path.dirname(vidname)
            f = os.listdir(d)
            v = random.choice(f)
            wrongvidname = os.path.join(d,v)
            wrong_img_names = list(glob(join(wrongvidname, '*.png')))
            wrong_img_name = random.choice(wrong_img_names)
            if wrong_img_name == img_name:
                continue
            wrong_window_fnames = self.get_window(wrong_img_name)
            # print('222')
            if wrong_window_fnames is None:
                # print('222')
                continue
            else:
                break
        # print(2)
        # print('111')
        if random.choice([True, False]):
            y = torch.ones(1).float()
            window_fnames = window_fnames
        else:
            y = torch.zeros(1).float()
            if random.choice([True, False]):
                window_fnames = wrong_window_fnames
            else:
                window_fnames = wrong_id_window_fnames
        # print('hahhaha')
        window = []
        all_read = True
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                all_read = False
                break
            try:
                img = cv2.resize(img, (224, 224))
            except Exception as e:
                all_read = False
                break

            window.append(img)
        try:
            
            wavpath = join(vidname, "audio.wav")
            waveform,sample_rate = torchaudio.load(wavpath)
            if (waveform.shape[0] == 2):
                waveform = waveform[:1,:]
            wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

            orig_mel = wav.squeeze(0).T
            # print('hahhaha')
        except Exception as e:
            print(traceback.format_exc())
        # print(3)
        mel = self.crop_audio_window(orig_mel, img_name)
        # not_sync_mel = self.crop_audio_window(orig_mel, wrong_img_name)
        
        # try:
        #     wavpath = join(wrong_id_vidname, "audio.wav")
        #     # print(wavpath)
        #     waveform,sample_rate = torchaudio.load(wavpath)
        #     if(waveform==None):
        #         print(wavpath)
        #     wrong_id_wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)

        #     wrong_id_mel = wrong_id_wav.squeeze(0).T
        # except Exception as e:
        #     print(traceback.format_exc())
        # H x W x 3 * T
        # print(5)
        # wrong_id_startidx = random.randint(0,wrong_id_mel.shape[0]-16)
        # wrong_id_mel = wrong_id_mel[wrong_id_startidx:wrong_id_startidx+syncnet_mel_step_size,:]
        x = np.concatenate(window, axis=2) / 255.
        x = x.transpose(2, 0, 1)
        x = torch.FloatTensor(x)
        mel = mel.T.unsqueeze(0)
        return x,mel,y

class Dataset(object):
    def __init__(self, split,args):
        self.all_videos = get_image_list(args.data_root, split)
        self.data_root = args.data_root

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames
    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size
        maxlen = spec.shape[0]
        if end_idx > maxlen:
            end_idx = maxlen
            start_idx = end_idx - syncnet_mel_step_size
        return spec[start_idx : end_idx, :]
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            # print(1111)
            vidname = os.path.join(self.data_root, vidname)
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                # print(vidname)
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                # print(vidname)
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    # print('1212')
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (224, 224))
                except Exception as e:
                    # print(123123)
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue
            # print(1)
            try:
                wavpath = join(vidname, "audio.wav")
                waveform,sample_rate = torchaudio.load(wavpath)
                if (waveform.shape[0] == 2):
                    waveform = waveform[:1,:]
                wav = torchaudio.transforms.MelSpectrogram(n_mels = 80,win_length=800,hop_length=200,n_fft=800,power = 1.0,mel_scale='slaney')(waveform)
                orig_mel = wav.squeeze(0).T
            except Exception as e:
                print(wavpath)
                continue

            mel = self.crop_audio_window(orig_mel, img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            # x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            # print('hahaha')
            return x, mel, y