# create new anaconda env
conda create -n onerestore python=3.7
conda activate onerestore 

# install pytorch (Take cuda 11.7 as an example to install torch 1.13)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# install other packages
pip install -r requirements.txt
pip install gensim
pip install ftfy regex tqdm 
pip install git+https://github.com/openai/CLIP.git

######上边是配置环境的过程，CLIP如果不能直接用git的话可以去github下载后然后setup

#如果需要用这个模型的话，需要改一下数据的加载和测试部分
#我之前的数据是加载16张图为一组如下，并且结合使用了对比学习损失。
types = ["clear", "color","haze","dark","noise",         "haze2dark","dark2haze","haze2noise","dark2noise",         "color2dark","dark2color","color2noise",         "haze2dark2noise","dark2haze2noise",         "color2dark2noise","dark2color2noise"]

需要修改或者写一个dataloader，能够加载成对的数据【"hazy","clear"】
测试部分也是同理，并不需要加载16个文件夹。

prompt-ckpt是我在这个网络之前训练的一个提示Embedding，如果不需要的话可以删掉，或者修改为其他的。
这个Embedding是基于CLIP做的，详细可以看我的论文。

本项目核心的部分就是model/OneRestore.py这个模型。把dataloader写一下然后改成医疗去雾的训练一下试试，prompt-ckpt这块可以先不动。