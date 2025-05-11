import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.utils_word_embedding import initialize_wordembedding_matrix

class Backbone(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(Backbone, self).__init__()

        if backbone == 'resnet18':
            resnet = torchvision.models.resnet.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet50':
            resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet101':
            resnet = torchvision.models.resnet.resnet101(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        self.block0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.block1 = resnet.layer1
        self.block2 = resnet.layer2
        self.block3 = resnet.layer3
        self.block4 = resnet.layer4

    def forward(self, x, returned=[4]):
        blocks = [self.block0(x)]

        blocks.append(self.block1(blocks[-1]))
        blocks.append(self.block2(blocks[-1]))
        blocks.append(self.block3(blocks[-1]))
        blocks.append(self.block4(blocks[-1]))

        out = [blocks[i] for i in returned]
        return out

class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred

class Embedder(nn.Module):
    """
    Text and Visual Embedding Model.
    """
    def __init__(self,
                 type_name,
                 feat_dim = 512,
                 mid_dim = 1024,
                 out_dim = 324,
                 drop_rate = 0.35,
                 cosine_cls_temp = 0.05,
                 wordembs = 'glove',
                 extractor_name = 'resnet18'):
        super(Embedder, self).__init__()

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.type_name = type_name
        self.feat_dim = feat_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.drop_rate = drop_rate
        self.cosine_cls_temp = cosine_cls_temp
        self.wordembs = wordembs
        self.extractor_name = extractor_name
        self.transform = transforms.Normalize(mean, std)
        
        self._setup_word_embedding()
        self._setup_image_embedding()
        
    def _setup_image_embedding(self):
        # image embedding
        self.feat_extractor = Backbone(self.extractor_name)

        img_emb_modules = [
            nn.Conv2d(self.feat_dim, self.mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU()
        ]
        if self.drop_rate > 0:
            img_emb_modules += [nn.Dropout2d(self.drop_rate)]
        self.img_embedder = nn.Sequential(*img_emb_modules)

        self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_final = nn.Linear(self.mid_dim, self.out_dim)

        self.classifier = CosineClassifier(temp=self.cosine_cls_temp)

    def _setup_word_embedding(self):

        self.type2idx = {self.type_name[i]: i for i in range(len(self.type_name))}
        self.num_type = len(self.type_name)
        train_type = [self.type2idx[type_i] for type_i in self.type_name]
        self.train_type = torch.LongTensor(train_type).to("cuda" if torch.cuda.is_available() else "cpu")

        wordemb, self.word_dim = \
            initialize_wordembedding_matrix(self.wordembs, self.type_name)
        
        self.embedder = nn.Embedding(self.num_type, self.word_dim)
        self.embedder.weight.data.copy_(wordemb)

        self.mlp = nn.Sequential(
                nn.Linear(self.word_dim, self.out_dim),
                nn.ReLU(True)
            )

    def train_forward(self, batch):

        scene, img = batch[0], self.transform(batch[1])
        bs = img.shape[0]

        # word embedding
        scene_emb = self.embedder(self.train_type)
        scene_weight = self.mlp(scene_emb)

        #image embedding
        img = self.feat_extractor(img)[0]
        img = self.img_embedder(img)
        img = self.img_avg_pool(img).squeeze(3).squeeze(2)
        img = self.img_final(img)

        pred = self.classifier(img, scene_weight)
        label_loss = F.cross_entropy(pred, scene)
        pred = torch.max(pred, dim=1)[1]
        type_pred = self.train_type[pred]
        correct_type = (type_pred == scene)
        out = {
            'loss_total': label_loss,
            'acc_type': torch.div(correct_type.sum(),float(bs)), 
        }

        return out
    
    def image_encoder_forward(self, batch):
        img = self.transform(batch)

        # word embedding
        scene_emb = self.embedder(self.train_type)
        scene_weight = self.mlp(scene_emb)

        #image embedding
        img = self.feat_extractor(img)[0]
        bs, _, h, w = img.shape
        img = self.img_embedder(img)
        img = self.img_avg_pool(img).squeeze(3).squeeze(2)
        img = self.img_final(img)

        pred = self.classifier(img, scene_weight)
        pred = torch.max(pred, dim=1)[1]

        out_embedding = torch.zeros((bs,self.out_dim)).to("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(bs):
            out_embedding[i,:] = scene_weight[pred[i],:]
        num_type = self.train_type[pred]
        text_type = [self.type_name[num_type[i]] for i in range(bs)]

        return out_embedding, num_type, text_type
    
    def text_encoder_forward(self, text):

        bs = len(text)

        # word embedding
        scene_emb = self.embedder(self.train_type)
        scene_weight = self.mlp(scene_emb)

        num_type = torch.zeros((bs)).to("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(bs):
            num_type[i] = self.type2idx[text[i]]

        out_embedding = torch.zeros((bs,self.out_dim)).to("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(bs):
            out_embedding[i,:] = scene_weight[int(num_type[i]),:]
        text_type = text

        return out_embedding, num_type, text_type
    
    def text_idx_encoder_forward(self, idx):

        bs = idx.shape[0]

        # word embedding
        scene_emb = self.embedder(self.train_type)
        scene_weight = self.mlp(scene_emb)

        num_type = idx

        out_embedding = torch.zeros((bs,self.out_dim)).to("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(bs):
            out_embedding[i,:] = scene_weight[int(num_type[i]),:]

        return out_embedding

    def contrast_loss_forward(self, batch):

        img = self.transform(batch)

        #image embedding
        img = self.feat_extractor(img)[0]
        img = self.img_embedder(img)
        img = self.img_avg_pool(img).squeeze(3).squeeze(2)
        img = self.img_final(img)

        return img
    
    def forward(self, x, type = 'image_encoder'):

        if type == 'train':
            out = self.train_forward(x)

        elif type == 'image_encoder':
            with torch.no_grad():
                out = self.image_encoder_forward(x)

        elif type == 'text_encoder':
            out = self.text_encoder_forward(x)
        
        elif type == 'text_idx_encoder':
            out = self.text_idx_encoder_forward(x)

        elif type == 'visual_embed':
            x = F.interpolate(x,size=(224,224),mode='bilinear')
            out = self.contrast_loss_forward(x)

        return out
    
if __name__ == '__main__':
    import torch

    # 假设有3个类别
    type_name = ['cat', 'dog', 'car']
    embedder = Embedder(type_name, out_dim=512).to("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 文本编码（prompt编码）
    prompt = ['cat']
    text_emb, num_type, text_type = embedder.text_encoder_forward(prompt)
    print("text_encoder_forward:")
    print("  text_emb shape:", text_emb.shape)  # [1, 512]
    print("  num_type:", num_type)
    print("  text_type:", text_type)

    # 2. 文本索引编码
    idx = torch.tensor([1]).to("cuda" if torch.cuda.is_available() else "cpu")
    idx_emb = embedder.text_idx_encoder_forward(idx)
    print("text_idx_encoder_forward:")
    print("  idx_emb shape:", idx_emb.shape)  # [1, 512]

    # 3. 图像编码
    img = torch.randn(1, 3, 224, 224).to("cuda" if torch.cuda.is_available() else "cpu")
    img_emb, num_type_img, text_type_img = embedder.image_encoder_forward(img)
    print("image_encoder_forward:")
    print("  img_emb shape:", img_emb.shape)  # [1, 512]
    print("  num_type_img:", num_type_img)
    print("  text_type_img:", text_type_img)

    # 4. 视觉特征（对比学习用）
    visual_emb = embedder.contrast_loss_forward(img)
    print("contrast_loss_forward:")
    print("  visual_emb shape:", visual_emb.shape)  # [1, 512]

    # 5. 直接用 forward 调用不同类型
    out = embedder(img, type='image_encoder')
    print("forward(image_encoder):", out[0].shape)
    out = embedder(prompt, type='text_encoder')
    print("forward(text_encoder):", out[0].shape)
    out = embedder(idx, type='text_idx_encoder')
    print("forward(text_idx_encoder):", out.shape)
    out = embedder(img, type='visual_embed')
    print("forward(visual_embed):", out.shape)