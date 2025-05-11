import os, time, torch, argparse
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.is_available())

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import numpy as np
from torchvision import transforms
from makedataset import Dataset
from utils.utils import print_args, load_restore_ckpt_with_optim, adjust_learning_rate, data_process, tensor_metric, load_excel, save_checkpoint
from model.loss import Total_loss

from PIL import Image
from model.Prompt import TextEncoder,Prompts
import clip
from utils.clip_score import L_clip_from_feature
from torch.utils.tensorboard import SummaryWriter
import random



device = "cuda" if torch.cuda.is_available() else "cpu"
#load clip
CLIP_model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
model_path = './prompt-ckpt/model_iter_12000_klrate.pth'
CLIP_model.load_state_dict(torch.load(model_path, map_location=device))
CLIP_model.to(device)
for para in CLIP_model.parameters():
    para.requires_grad = False

def set_seed(seed):
    random.seed(seed)             # Python随机数生成器
    np.random.seed(seed)          # NumPy随机数生成器
    torch.manual_seed(seed)       # PyTorch随机数生成器（CPU）
    torch.cuda.manual_seed(seed)  # PyTorch随机数生成器（GPU）
    torch.cuda.manual_seed_all(seed) # 所有GPU的随机数生成器
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getTextFeature():
    # return size = types*512
    length_prompt = 5
    prompt_path = './prompt-ckpt/prompt_iter_12000_klrate.pth'
    learn_prompt = Prompts(CLIP_model,prompt_path).cuda()
    learn_prompt =  torch.nn.DataParallel(learn_prompt)
    for name, param in learn_prompt.named_parameters():
        param.requires_grad_(False)
    text_encoder = TextEncoder(CLIP_model)
    embedding_prompt=learn_prompt.module.embedding_prompt
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*length_prompt)]])
    text_features = text_encoder(embedding_prompt,tokenized_prompts)
    
    return text_features

writer = SummaryWriter('./ckpts/model-train/model-train-tensorboard/')

def main(args):
    print('> Set Random Seed...')
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('> Model Initialization...')

    text_features = getTextFeature()
    print(text_features.size())

    restorer, optimizer, cur_epoch = load_restore_ckpt_with_optim(device, freeze_model=False, ckpt_name=args.restore_model_path, lr=args.lr)
    loss = Total_loss(args,CLIP_model,text_features)

    print('> Loading dataset...')
    data = Dataset(args.train_input)
    print(len(data))
    dataset = DataLoader(dataset=data, num_workers=args.num_works, batch_size=args.bs, shuffle=True, pin_memory = True)
    # print(len(dataset))
    
    print('> Start training...')
    start_all = time.time()
    # test(args,restorer,text_features,device,1)
    train(restorer, text_features, optimizer, loss, cur_epoch, args, dataset, device)
    end_all = time.time()
    print('Whloe Training Time:' +str(end_all-start_all)+'s.')

L_clip_Feature = L_clip_from_feature(CLIP_model)

type_dict = {
    "clear": [1, 0, 0, 0, 0],
    "color": [0, 1, 0, 0, 0],
    "haze": [0, 0, 1, 0, 0],
    "dark": [0, 0, 0, 1, 0],
    "noise": [0, 0, 0, 0, 1],
    "haze2dark": [0, 0, 0.5, 0.5, 0],
    "dark2haze": [0, 0, 0.5, 0.5, 0],
    "haze2noise": [0, 0, 0.5, 0, 0.5],
    "dark2noise": [0, 0, 0, 0.5, 0.5],
    "color2dark": [0, 0.5, 0, 0.5, 0],
    "dark2color": [0, 0.5, 0, 0.5, 0],
    "color2noise": [0, 0.5, 0, 0, 0.5],
    "haze2dark2noise": [0, 0, 0.33, 0.33, 0.33],
    "dark2haze2noise": [0, 0, 0.33, 0.33, 0.33],
    "color2dark2noise": [0, 0.33, 0, 0.33, 0.33],
    "dark2color2noise": [0, 0.33, 0, 0.33, 0.33]
}

def train(restorer, text_features, optimizer, loss, cur_epoch, args, dataset, device):
    cur_iter = 0
    os.makedirs(args.save_model_path,exist_ok=True)
    metric = []
    for epoch in range(cur_epoch, args.epoch):
        epoch_loss = 0.0
        optimizer = adjust_learning_rate(optimizer, epoch, args.adjust_lr)
        learnrate = optimizer.param_groups[-1]['lr']
        restorer.train()

        for i, data in enumerate(dataset,0):
            cur_iter = cur_iter + 1
            pos, inp, neg = data_process(data, args, device)
            batch_features = []

            k1, rate = L_clip_Feature(inp[0], text_features)

            # l, _  = text_features.size()
            # for k in range(len(inp[1])):
            #     f = torch.zeros(512).to(device)
            #     for j in range(l):
            #         f = f + text_features[j,:] * rate[0][j]
            #     batch_features.append(f)

            for k in range(len(inp[1])):
                t_name = inp[1][k]
                rate = type_dict[t_name]
                f = torch.zeros(512).to(device)
                for j in range(len(rate)):
                    f = f + text_features[j,:] * rate[j]
                batch_features.append(f)
            
            batch_features_tensor = torch.stack(batch_features)
            out = restorer(inp[0], batch_features_tensor)

            restorer.zero_grad()
            total_loss = loss(inp, pos, neg, out)
            writer.add_scalars('Loss_iter', {'train':total_loss}, cur_iter)
            epoch_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()

            mse = tensor_metric(pos,out, 'MSE', data_range=1)
            psnr = tensor_metric(pos,out, 'PSNR', data_range=1)
            ssim = tensor_metric(pos,out, 'SSIM', data_range=1)
            print("[epoch %d][%d/%d] lr :%f Floss: %.4f MSE: %.4f PSNR: %.4f SSIM: %.4f"%(epoch+1, i+1, \
                len(dataset), learnrate, total_loss.item(), mse, psnr, ssim))
        
        psnr_t1, ssim_t1, psnr_t2, ssim_t2 = 0,0,0,0
        
        epoch_loss = epoch_loss / len(dataset)
        writer.add_scalars('Loss_epoch', {'train':epoch_loss}, epoch)

        if(epoch % 20 == 0):
            psnr_t1, ssim_t1, psnr_t2, ssim_t2 = test(args, restorer, text_features, device, epoch)
            # metric.append([psnr_t1, ssim_t1, psnr_t2, ssim_t2])
            # print("[epoch %d] Test images PSNR1: %.4f SSIM1: %.4f"%(epoch+1, psnr_t1,ssim_t1))
            save_checkpoint({'epoch': epoch + 1,'state_dict': restorer.state_dict(),'optimizer' : optimizer.state_dict()},\
                            args.save_model_path, epoch+1,epoch_loss ,psnr_t1,ssim_t1,psnr_t2,ssim_t2)

transform_resize = transforms.Compose([         
    transforms.Resize([224,224]),         
    transforms.ToTensor()         
])
def test(args, restorer, text_features, device, epoch=-1):
    combine_type = args.degr_type
    psnr_1, psnr_2, ssim_1, ssim_2 = 0, 0, 0, 0
    os.makedirs(args.output,exist_ok=True)

    for i in range(len(combine_type)-1):
        file_list =  os.listdir(f'{args.test_input}/{combine_type[i+1]}/')
        for j in range(len(file_list)):
            hq = Image.open(f'{args.test_input}/{combine_type[0]}/{file_list[j]}')
            lq = Image.open(f'{args.test_input}/{combine_type[i+1]}/{file_list[j]}')
            restorer.eval()
            with torch.no_grad():
                lq_re = torch.Tensor((np.array(lq)/255).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                # lq_em = transform_resize(lq).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                hq = torch.Tensor((np.array(hq)/255).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

                starttime = time.time()

                # idx1 = types.index(combine_type[i+1])
                # text_feature1 = text_features[idx1,:]
                # text_feature1 = text_feature1.unsqueeze(0)

                
                # _, rate = L_clip_Feature(lq_re, text_features)
                # idx2 = rate[0].argmax()
                # text_feature2 = text_features[idx2,:]
                # text_feature2 = text_feature2.unsqueeze(0)
                
                t_name = combine_type[i+1]
                rate = type_dict[t_name]
                f = torch.zeros(512).to(device)
                for k in range(len(rate)):
                    f = f + text_features[k,:] * rate[k]
                text_feature1 = f.unsqueeze(0)
                # print(idx)
                
                _, rate = L_clip_Feature(lq_re, text_features)
                l, _  = text_features.size()
                f = torch.zeros(512).to(device)
                for t in range(l):
                    f = f + text_features[t,:]*rate[0][t]
                text_feature2 = f.unsqueeze(0)

                out_1 = restorer(lq_re, text_feature1)
                out_2 = restorer(lq_re, text_feature2)

                # out_1 = restorer(lq_re, text_feature1)
                # if idx1 != idx2:
                #     print(idx1,types[idx1],idx2,types[idx2])
                #     out_2 = restorer(lq_re, text_feature2)
                # else:
                #     out_2 = out_1
                
                endtime1 = time.time()

                # imwrite(torch.cat((lq_re, out_1, out_2, hq), dim=3), args.output \
                #     + file_list[j][:-4] + '_' + str(epoch) + '_' + combine_type[i+1] + '.png', range=(0, 1))
                imwrite(torch.cat((lq_re, out_1,out_2,hq), dim=3), args.output \
                    + file_list[j][:-4] + '_' + combine_type[i+1] + '.png', range=(0, 1))
            psnr_1 += tensor_metric(hq, out_1, 'PSNR', data_range=1)
            ssim_1 += tensor_metric(hq, out_1, 'SSIM', data_range=1)
            psnr_2 += tensor_metric(hq, out_2, 'PSNR', data_range=1)
            ssim_2 += tensor_metric(hq, out_2, 'SSIM', data_range=1)
            print('The ' + file_list[j][:-4] + ' ' + combine_type[i+1] + ' Time:' + str(endtime1 - starttime) + 's.')
    # length = len(combine_type) - 1
    # return psnr_1 / (len(file_list)*length), ssim_1 / (len(file_list)*length),\
    #     psnr_2 / (len(file_list)*length), ssim_2 / (len(file_list)*length)
    length = len(combine_type)
    return psnr_1 / (len(file_list)*length), ssim_1 / (len(file_list)*length),\
        psnr_2 / (len(file_list)*length), ssim_2 / (len(file_list)*length)


types = ["clear","color","haze","dark","noise",
        "haze2dark","dark2haze","haze2noise","dark2noise",
        "color2dark","dark2color","color2noise",
        "haze2dark2noise","dark2haze2noise",
        "color2dark2noise","dark2color2noise"]

# types = ["clear","color","haze","dark",
#         "haze2dark","dark2haze",
#         "color2dark","dark2color"]
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "OneRestore Training")

    # load model
    parser.add_argument("--restore-model-path", type=str, default = None, help = 'restore model path')
    parser.add_argument("--save_model_path", type=str, default = "./ckpts/model-train/", help = 'restore model path')

    parser.add_argument("--epoch", type=int, default = 200, help = 'epoch number')
    parser.add_argument("--bs", type=int, default = 1, help = 'batchsize')
    parser.add_argument("--lr", type=float, default = 1e-4, help = 'learning rate')
    parser.add_argument("--adjust-lr", type=int, default = 40, help = 'adjust learning rate')
    parser.add_argument("--num-works", type=int, default = 4, help = 'number works')
    parser.add_argument("--loss-weight", type=tuple, default = (0.60, 0.30 ,0.00, 0.00), help = 'loss weights')
    parser.add_argument("--degr-type", type=list, default = types, help = 'degradation type')
    
    parser.add_argument("--train-input", type=str, default = "./dataset.h5", help = 'train data')
    parser.add_argument("--test-input", type=str, default = "./data/test_set", help = 'test path')
    parser.add_argument("--output", type=str, default = "./result/", help = 'output path')

    argspar = parser.parse_args()

    print_args(argspar)
    main(argspar)
