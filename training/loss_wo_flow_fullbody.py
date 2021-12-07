# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

from typing import Union, List, Dict, Any, cast

import torch.nn.functional as F
from util_functions import random_affine_matrix, get_affine_matrix, getRandomAffineParam, feature_normalize

# import PIL.Image as Image
import cv2
import random
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, gen_z, real_c, retain, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, G_const_encoding, G_style_encoding, 
                 D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, 
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=0, l1_weight=50, vgg_weight=50, 
                 contextual_weight=1.0, mask_weight=1.0):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.G_const_encoding = G_const_encoding
        self.G_style_encoding = G_style_encoding
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.l1_weight = l1_weight
        self.vgg_weight = vgg_weight
        self.mask_weight = mask_weight

        class_weight  = torch.FloatTensor([1,2,2,3,3,3]).to(device)
        self.ce_parsing = torch.nn.CrossEntropyLoss(ignore_index=255,weight=class_weight)

        self.contextual_weight = contextual_weight
        if self.vgg_weight > 0:
            self.criterionVGG = VGGLoss(device=device, requires_grad=False)
            print('device: ', device)

        if self.contextual_weight > 0:
            contextual_vgg_path = './checkpoints/vgg19_conv.pth'
            self.contextual_vgg = VGG19_feature_color_torchversion()
            self.contextual_vgg.load_state_dict(torch.load(contextual_vgg_path))
            self.contextual_vgg.eval()
            for param in self.contextual_vgg.parameters():
                param.requires_grad = False
            self.contextual_vgg.to(device)
            self.contextual_layers = ['r12','r22','r32','r42','r52']
            self.contextual_forward_loss = ContextualLoss_forward()

    def run_G(self, z, c, pose, const_feats, denorm_upper_mask, denorm_lower_mask, denorm_upper_input, denorm_lower_input, sync):
        cat_feats = {}
        for _, cat_feat in enumerate(const_feats):
            h, _ = cat_feat.shape[2], cat_feat.shape[3]
            cat_feats[str(h)] = cat_feat

        with misc.ddp_sync(self.G_const_encoding, sync):
            pose_feat = self.G_const_encoding(pose)

        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        with misc.ddp_sync(self.G_synthesis, sync):
            img, finetune_img, pred_parsing = self.G_synthesis(ws, pose_feat, cat_feats, denorm_upper_input, denorm_lower_input, denorm_upper_mask, denorm_lower_mask)

        return img, finetune_img, pred_parsing, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)

        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits


    # def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, pose_heatmap, sync, gain):
    def accumulate_gradients(self, phase, real_img, gen_z, style_input, retain, pose, denorm_upper_input, denorm_lower_input, denorm_upper_mask, denorm_lower_mask, \
                             gt_parsing, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        with misc.ddp_sync(self.G_style_encoding, sync):
            real_c, cat_feats = self.G_style_encoding(style_input, retain)
            gen_c = real_c # 把 real_c 也当做 gen_c作为CGAN的C

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, gen_finetune_img, pred_parsing, _gen_ws = self.run_G(gen_z, gen_c, pose, cat_feats, denorm_upper_mask, denorm_lower_mask, \
                                                                                                denorm_upper_input, denorm_lower_input, sync=(sync and not do_Gpl)) # May get synced by Gpl.

                # 这里的conditioned GAN的 (gen_img, gen_c) 和 (real_img, real_c) 不是严格对应的。
                # 如果加入pose conditioned, 那么应该 gen_img和real_img严格对应，然后 只用一个real pose, 也就是(gen_img, real_pose) 和 (real_img, real_pose)
                # 视情况, 看是否需要加入L1 和 vgg loss

                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                gen_finetune_logits = self.run_D(gen_finetune_img, gen_c, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gmain = loss_Gmain.mean()
                training_stats.report('Loss/G/loss', loss_Gmain)

                training_stats.report('Loss/scores/fake_finetune', gen_finetune_logits)
                training_stats.report('Loss/signs/fake_finetune', gen_finetune_logits.sign())
                loss_Gmain_finetune = torch.nn.functional.softplus(-gen_finetune_logits) # -log(sigmoid(gen_logits))
                loss_Gmain_finetune = loss_Gmain_finetune.mean()
                training_stats.report('Loss/G/loss_finetune', loss_Gmain_finetune)

                # l1 loss
                loss_G_L1 = 0
                loss_G_finetune_L1 = 0
                if self.l1_weight > 0:
                    loss_G_L1 = torch.nn.L1Loss()(gen_img, real_img) * self.l1_weight
                    # loss_G_L1 = loss_G_L1.mean()

                    loss_G_finetune_L1 = torch.nn.L1Loss()(gen_finetune_img, real_img) * self.l1_weight
                    # loss_G_finetune_L1 = loss_G_finetune_L1.mean()
                training_stats.report('Loss/G/L1', loss_G_L1)
                training_stats.report('Loss/G/L1_finetune', loss_G_finetune_L1)

                loss_mask = 0
                if self.mask_weight > 0:
                    loss_mask = torch.mean(self.ce_parsing(pred_parsing, gt_parsing.long()[:,0,...])) * self.mask_weight

                training_stats.report('Loss/G/mask_loss', loss_mask)

                # vgg loss
                loss_G_VGG = 0
                loss_G_finetune_VGG = 0
                if self.vgg_weight > 0:
                    loss_G_VGG = self.criterionVGG(gen_img, real_img) * self.vgg_weight
                    loss_G_VGG = loss_G_VGG.mean()

                    loss_G_finetune_VGG = self.criterionVGG(gen_finetune_img, real_img) * self.vgg_weight
                    loss_G_finetune_VGG = loss_G_finetune_VGG.mean()
                training_stats.report('Loss/G/vgg', loss_G_VGG)
                training_stats.report('Loss/G/vgg_finetune', loss_G_finetune_VGG)
                
                loss_G = (loss_Gmain + loss_Gmain_finetune) / 2 + \
                         (loss_G_L1 + loss_G_finetune_L1) / 2 + \
                         (loss_G_VGG + loss_G_finetune_VGG) / 2 + loss_mask
                # loss_G = loss_Gmain + \
                #          (loss_G_L1 + loss_G_finetune_L1) / 2 + \
                #          (loss_G_VGG + loss_G_finetune_VGG) / 2 + loss_mask
                
            with torch.autograd.profiler.record_function('Gmain_backward'):
                # loss_Gmain.mean().mul(gain).backward()
                loss_G.mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                # with misc.ddp_sync(self.G_flownet, sync):
                #     flow = self.G_flownet(torch.cat((cloth[:batch_size], aff_pose[:batch_size]), dim=1))
                # warp_cloth = F.grid_sample(cloth[:batch_size, :3, :, :], flow)

                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], pose[:batch_size],
                                            [cat_feat[:batch_size] for cat_feat in cat_feats], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()


        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, gen_finetune_img, _, _gen_ws = self.run_G(gen_z, gen_c, pose, cat_feats, denorm_upper_mask, denorm_lower_mask, \
                                                                        denorm_upper_input, denorm_lower_input, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                gen_finetune_logits = self.run_D(gen_finetune_img, gen_c, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                training_stats.report('Loss/scores/fake_finetune', gen_finetune_logits)
                training_stats.report('Loss/signs/fake_finetune', gen_finetune_logits.sign())
                loss_Dgen_finetune = torch.nn.functional.softplus(gen_finetune_logits) # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Dgen_backward'):
                ((loss_Dgen.mean()+loss_Dgen_finetune.mean())/2).mul(gain).backward()
                # loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------

#----- start VGG Loss-----------------------------------------------------------------------
class VGGLoss(nn.Module):
    def __init__(self, device, requires_grad=False, weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19_Feature(device, requires_grad=requires_grad)
        self.criterion = nn.L1Loss()
        self.weights = weights

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGG19_Feature(torch.nn.Module):
    def __init__(self, device, ckpt_path="./checkpoints/vgg19-dcbb9e9d.pth", requires_grad=False):
        super(VGG19_Feature, self).__init__()
        #from torchvision import models
        #vgg_pretrained_features = models.vgg19(pretrained=True, ckpt=ckpt).features  # torchvision already include vgg net
        vgg_pretrained_features = vgg19(pretrained=True, progress=False, ckpt_path=ckpt_path).features  # load from local file
        vgg_pretrained_features = vgg_pretrained_features.eval()
        vgg_pretrained_features = vgg_pretrained_features.to(device)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        print('load vgg19 success!')

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# --------------start copy from https://github.com/pytorch/vision/blob/c991db82abba12e664eeac14c9b643d0f1f1a7df/torchvision/models/vgg.py#L25 ------

class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg19(pretrained: bool = False, progress: bool = True, ckpt_path: str = "./checkpoints/vgg19-dcbb9e9d.pth", **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, ckpt_path, **kwargs)

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, ckpt_path: str = "./checkpoints/vgg19-dcbb9e9d.pth", **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        print("vgg19 ckpt_path from local: ", ckpt_path)
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)
    return model

# --------------end copy from https://github.com/pytorch/vision/blob/c991db82abba12e664eeac14c9b643d0f1f1a7df/torchvision/models/vgg.py#L25 ------

#----- end VGG Loss-----------------------------------------------------------------------

# ----------------------------------- for ContextualLoss ----------------------------------------

def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst

class VGG19_feature_color_torchversion(nn.Module):
    ''' 
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=True, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class ContextualLoss_forward(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self, PONO=True):
        super(ContextualLoss_forward, self).__init__()
        self.PONO = PONO
        return None

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        if feature_centering:
            if self.PONO:
                X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
                Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)
            else:
                X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size

        # X_features = F.unfold(
        #     X_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2
        # Y_features = F.unfold(
        #     Y_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        # d_norm = d
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-3)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        loss = -torch.log(CX)

        # contextual loss per batch
        # loss = torch.mean(loss)
        return loss

# --------------------------------------------- ContextualLoss end ------------------------------------------------------
