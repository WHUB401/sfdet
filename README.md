# 环境
运行pip install -r requirements.txt

重要的包：

    facenet-pytorch         2.5.2（人脸检测）
    torchaudio              0.13.0+cu116
    librosa                 0.9.2
    opencv-python           4.6.0.66
    torch                   1.13.0
    albumentations          1.3.0 （数据增强库，做压缩和高斯模糊等实验使用）

# 数据集
在这个链接下载FakeAVCeleb:https://github.com/DASH-Lab/FakeAVCeleb

然后在这个连接下载通用视频数据集 ：http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

这个连接对应的地址好像下载不了了=。= 后面他挪到一个韩国人实验室网站上面去了 

# 数据预处理

首先呢，这个数据集存在一个问题，真实视频只有500个，伪造的19500个。

所以为了解决这个问题（顺便避免信息泄露问题），从VoxCeleb中随机挑选了同身份的真实视频，真实视频数量扩充到8000个：
>python adddata.py

adddata.py会生成一个addlist.txt，即选中的用以扩充的真实视频，然后将这些视频与原数据集合并，即为没有问题的数据集。




用人脸检测算法提取视频的人脸帧，然后按五张连续一组分组，格式如下：

      --FakeAVCeleb
         --FakeVideo-FakeAudio
            --African
              --men
               --id00076
                 --0000000
                 --0000001
                 ...
  不想自己实现的话，运行preprocess.py ，其将main()设置的文件夹下的视频按照这个格式进行检测或切片
 > python preprocess.py
 
 对于VoxCeleb数据集，因为其太太太太太太大，没有用上其所有数据进行实验，所以对于每个人的视频只抽取了十秒作为训练样本。
 
 如果资源足够你可以试试把所有的视频都用来训练。
 
 处理方式：
 
 >python prevox2.py
 
 完成后文件夹格式应当是这个样子：
 
    VoxCeleb2
      --dev
       --mp4
        --id00012
         --train
           --0000
           --0001

然后，将这些视频分为训练集和测试集：
>python split.py

会生成train.txt和test.txt

# 预训练
>python pretrain.py --data_root --num_gpu --checkpoint_dir --lr --batch_size

data_root 为数据集地址。num_gpu为指定用哪几块卡 比如'1,2' checkpoint_dir为保存训练参数的文件夹。

PS.可以自行更换不同的人脸和语音编码器尝试训练。更改训练代码中的face_encoder和voice_encoder即可。

# 训练
>python train.py --data_root --num_gpu --checkpoint_dir --lr --batch_size

在train.py中可以更改预加载权重。

# 测试

>python testn.py

会生成相似度分布图
