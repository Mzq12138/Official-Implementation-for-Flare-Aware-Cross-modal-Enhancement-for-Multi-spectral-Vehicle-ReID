import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import pdb
import numpy as np
def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)





def default(val, d):
    return val if val is not None else d

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.key_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.value_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        # pdb.set_trace()
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        context = default(context, x)

        # Project query, key, and value vectors.
        query = self.query_proj(x)
        key = self.key_proj(context)
        value = self.value_proj(context)

        # Split query, key, and value vectors into heads.
        # (b, n, h, dim_head)
        # (b, h, n, dim_head)

        query = query.view(-1,  self.heads, query.shape[1], self.dim_head)
        key = key.view(-1, self.heads, key.shape[1], self.dim_head)
        value = value.view(-1,  self.heads, value.shape[1], self.dim_head)

        attention_scores = torch.einsum('b h i d, b h j d -> b h i j', query, key) * self.scale

        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.einsum('bhij, bhjd -> bhid', attention_weights, value)
        # pdb.set_trace()
        attention_output = attention_output.view(-1, attention_output.shape[2], self.heads * self.dim_head)
        out = self.out_proj(attention_output)

        # Apply dropout to output.
        out = self.dropout(out)

        return out
    
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args,**kwargs):
        # pdb.set_trace()
        x = self.norm(x)
        x = self.fn(x, *args,**kwargs)
        return x
    
# projecting CLS tokens, if the dimensions don't match, use a linear layer to project it in the right dimension
class ProjectInOut(nn.Module):
    """
    Adapter class that embeds a callable (layer) and handles mismatching dimensions
    """
    def __init__(self, dim_outer, dim_inner, fn):
        """Args:
            dim_outer (int): Input (and output) dimension.
            dim_inner (int): Intermediate dimension (expected by fn).
            fn (callable): A callable object (like a layer).
        """
        super().__init__()
        self.fn = fn

        need_projection = dim_outer != dim_inner

        self.project_in = nn.Linear(dim_outer, dim_inner) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_inner, dim_outer) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        """
        Args:
            *args, **kwargs: to be passed on to fn

        Notes:
            - after calling fn, the tensor has to be projected back into it's original shape   
            - fn(W_in) * W_out
        """
        x = self.project_in(x)
        x = self.fn(x, *args,**kwargs) + x
        x = self.project_out(x)
        
        return x

class CrossAttentionModuleViT(nn.Module):
    def __init__(self,dim=768):
        super(CrossAttentionModuleViT, self).__init__()
        att = Attention(dim = dim, heads = 16, dim_head = 768, dropout=0.5)
        norm = PreNorm(dim= dim, fn=att)
        self.cross_att = ProjectInOut(dim_outer= 768, dim_inner=768, fn= norm)
        self.cross_att1 = copy.deepcopy(self.cross_att)


    def forward(self, feat1, feat2):
        return self.cross_att(feat1, feat2),self.cross_att1(feat2, feat1)




class Backbone(nn.Module):

    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(Backbone, self).__init__()

        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)


        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.base1 = copy.deepcopy(self.base)
        self.base2 = copy.deepcopy(self.base)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b1N = copy.deepcopy(self.b1)
        self.b1T = copy.deepcopy(self.b1)

        self.num_classes = num_classes
        

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifierN = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifierN.apply(weights_init_classifier)

        self.classifierT = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifierT.apply(weights_init_classifier)


        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneckN = nn.BatchNorm1d(self.in_planes)
        self.bottleneckN.bias.requires_grad_(False)
        self.bottleneckN.apply(weights_init_kaiming)


        self.bottleneckT = nn.BatchNorm1d(self.in_planes)
        self.bottleneckT.bias.requires_grad_(False)
        self.bottleneckT.apply(weights_init_kaiming)


        
        
        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = cfg.MODEL.RE_ARRANGE


    
    def forward_features(self,x1,x2,x3, label=None, cam_label= None, view_label=None):
        featR = self.base(x1, cam_label=cam_label, view_label=view_label)
        featN = self.base1(x2, cam_label=cam_label, view_label=view_label)
        featT = self.base2(x3, cam_label=cam_label, view_label=view_label)
        return featR, featN, featT
    
    def process_branch_R(self, features):
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return [cls_score
                        ], [global_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat], dim=1)
            else:
                return torch.cat(
                    [global_feat], dim=1)

    def process_branch_N(self, features):
        b1_feat = self.b1N(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0] # global_Feat 只取token做分类
        # pdb.set_trace()
        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        
    
        feat = self.bottleneckN(global_feat)

        if self.training:
            cls_score = self.classifierN(feat)
            return [cls_score
                        ], [global_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat], dim=1)
            else:
                return torch.cat(
                    [global_feat], dim=1)

    def process_branch_T(self, features):
        b1_feat = self.b1T(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        
        feat = self.bottleneckT(global_feat)

        
        if self.training:
            cls_score = self.classifierT(feat)
            return [cls_score
                        ], [global_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat], dim=1)
            else:
                return torch.cat(
                    [global_feat], dim=1)

    def post_process(self, featR, featN, featT):

        resR = self.process_branch_R(featR)
        resN = self.process_branch_N(featN)
        resT = self.process_branch_T(featT)
        
        return resR, resN, resT
        
        
    def forward(self, x1,x2,x3, label, cam_label= None, view_label=None, flare_label = None):
        
        
       
        featR, featN, featT = self.forward_features(x1,x2,x3,label,cam_label,view_label)


        model1, model2, model3 = self.post_process(featR, featN, featT)


        
        if self.training:
            return model1, model2, model3,None,None,None,None
        else:
            return torch.cat([model1,model2,model3],dim=1)
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))




class FACENet(nn.Module):

    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(FACENet, self).__init__()

        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)


        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.base1 = copy.deepcopy(self.base)
        self.base2 = copy.deepcopy(self.base)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        
        self.b1N = copy.deepcopy(self.b1)
        self.b1T = copy.deepcopy(self.b1)
        self.b2N = copy.deepcopy(self.b2)
        self.b2T = copy.deepcopy(self.b2)

        self.num_classes = num_classes
        

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifierN = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifierN.apply(weights_init_classifier)

        self.classifierT = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifierT.apply(weights_init_classifier)


        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneckN = nn.BatchNorm1d(self.in_planes)
        self.bottleneckN.bias.requires_grad_(False)
        self.bottleneckN.apply(weights_init_kaiming)


        self.bottleneckT = nn.BatchNorm1d(self.in_planes)
        self.bottleneckT.bias.requires_grad_(False)
        self.bottleneckT.apply(weights_init_kaiming)


        self.cross_att = CrossAttentionModuleViT()

        ## threshold
        self.threshold = torch.nn.Parameter(torch.zeros(1))
        ##
        
        ## cls for label
        self.bottleneck_R_label = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_R_label.bias.requires_grad_(False)
        self.bottleneck_R_label.apply(weights_init_kaiming)

        self.bottleneck_N_label = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_N_label.bias.requires_grad_(False)
        self.bottleneck_N_label.apply(weights_init_kaiming)

        self.classifier_R_label = nn.Linear(self.in_planes, 2, bias=False)
        # self.classifier_R_label = nn.Linear(self.in_planes, 2, bias=True)
        self.classifier_R_label.apply(weights_init_classifier)

        self.classifier_N_label = nn.Linear(self.in_planes, 2, bias=False)
        # self.classifier_N_label = nn.Linear(self.in_planes, 2, bias=True)
        self.classifier_N_label.apply(weights_init_classifier)

        # Bottleneck for label
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b_R_label = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b_N_label = copy.deepcopy(self.b_R_label)



        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = cfg.MODEL.RE_ARRANGE

        self.use_fce = cfg.MODEL.FCE
        self.use_mcloss = cfg.MODEL.MCLOSS
        self.use_mfmp = cfg.MODEL.MFMP
    

        

    def get_cls_feat(self, featR, featN):
            
        tokenR = featR[:, 0:1]
        tokenN = featN[:, 0:1]
        
        featR = featR[:, 1:]
        featN = featN[:, 1:]
       
        featR, featN= self.cross_att(featR, featN)

        CA_comR = torch.cat([tokenR, featR], dim=1)
        CA_comN = torch.cat([tokenN, featN], dim=1)
        
        return  CA_comR, CA_comN
        

    def cross_modal_enhancement(self, featR, featN, featRforlabel, featNforlabel, flare_label, featT):
        tokenR = featR[:, 0:1]
        tokenN = featN[:, 0:1]
        threshold = self.threshold
        resnet_R, resnet_N, resnet_T, featRforlabel, featNforlabel = featR[:,1:,:], featN[:,1:,:], featT[:,1:,:], featRforlabel[:,1:,:], featNforlabel[:,1:,:]
       
        if flare_label is not None and (1 in flare_label):
            expo = flare_label
            expo = expo.view(expo.shape[0],1,1).cuda()
            # pdb.set_trace()
            Aid_R = (featRforlabel>threshold).float() * resnet_T
            Part_R_remain = resnet_R * (featNforlabel<=threshold).float()
           
            Aid_N = (featNforlabel>threshold).float() * resnet_T
            Part_N_remain = resnet_N * (featNforlabel<=threshold).float()

            resnet_R = Aid_R + Part_R_remain
            resnet_N = Aid_N + Part_N_remain
            
            resnet_R_flare = resnet_R*expo.float() # flare_sample_feat
            resnet_R_noflare = resnet_R*(expo==0).float() # not_flare_sample_feat
            resnet_R = resnet_R_flare + resnet_R_noflare

            resnet_N_flare = resnet_N*expo.float() # flare_sample_feat
            resnet_N_noflare = resnet_N*(expo==0).float() # not_flare_sample_feat
            resnet_N = resnet_N_flare + resnet_N_noflare

        return torch.cat([tokenR,resnet_R],dim=1), torch.cat([tokenN,resnet_N],dim=1)
 
    def forward_features(self,x1,x2,x3, label=None, cam_label= None, view_label=None):
        featR = self.base(x1, cam_label=cam_label, view_label=view_label)
        featN = self.base1(x2, cam_label=cam_label, view_label=view_label)
        featT = self.base2(x3, cam_label=cam_label, view_label=view_label)
        return featR, featN, featT
    
    def process_branch_R(self, features):
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]


        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return [cls_score
                        ], [global_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat], dim=1)
            else:
                return torch.cat(
                    [global_feat], dim=1)

    def process_branch_N(self, features):
        b1_feat = self.b1N(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0] # global_Feat
    
        feat = self.bottleneckN(global_feat)

        if self.training:
            cls_score = self.classifierN(feat)
            return [cls_score
                        ], [global_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat], dim=1)
            else:
                return torch.cat(
                    [global_feat], dim=1)

    def process_branch_T(self, features):
        b1_feat = self.b1T(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]
        
        feat = self.bottleneckT(global_feat)

        if self.training:
            cls_score = self.classifierT(feat)
            return [cls_score
                        ], [global_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat], dim=1)
            else:
                return torch.cat(
                    [global_feat], dim=1)

    def post_process(self, featR, featN, featT):

        resR = self.process_branch_R(featR)
        resN = self.process_branch_N(featN)
        resT = self.process_branch_T(featT)
        
        return resR, resN, resT
        
        
    def forward(self, x1,x2,x3, label, cam_label= None, view_label=None, flare_label = None):
        
        featR, featN, featT = self.forward_features(x1,x2,x3,label,cam_label,view_label)

        feat1_forlabel, feat2_forlabel = featR, featN

        if self.use_mfmp:
            feat1_forlabel, feat2_forlabel  = self.get_cls_feat(featR, featN)
        if self.use_fce:
            featR, featN = self.cross_modal_enhancement(featR, featN, feat1_forlabel, feat2_forlabel, flare_label, featT)

        model1, model2, model3 = self.post_process(featR, featN, featT)

        score_R_label=None
        score_N_label=None

        
        if self.training:
            if self.use_fce:
                # pdb.set_trace()
                feat1_forlabel = self.b_R_label(feat1_forlabel)
                feat1_forlabel_token = feat1_forlabel[:,0]
                feat1_forlabel_token_bn = self.bottleneck_R_label(feat1_forlabel_token)
                score_R_label = self.classifier_R_label(feat1_forlabel_token_bn)
                
                feat2_forlabel = self.b_N_label(feat2_forlabel)
                feat2_forlabel_token = feat2_forlabel[:,0]
                feat2_forlabel_token_bn = self.bottleneck_N_label(feat2_forlabel_token)
                score_N_label = self.classifier_N_label(feat2_forlabel_token_bn)

                return model1, model2, model3, torch.cat([feat1_forlabel_token], dim=1) ,torch.cat([score_R_label], dim=1),torch.cat([feat2_forlabel_token], dim=1), torch.cat([score_N_label], dim=1)
            return model1, model2, model3,None,None,None,None

        else:
  
            return torch.cat([model1,model2,model3],dim=1)
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
  
           
__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        model = FACENet(num_class, camera_num, view_num, cfg, __factory_T_type)
        print('===========FACENet ===========')
        
    else:
        model = Backbone(num_class, cfg)
        print('===========Backbone===========')
    return model