import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None, \
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

    def resize(self, x, gt):
        mode = self.resize_config['mode']
        align_corners = self.resize_config['align_corners']
        x = F.interpolate(
            input=x,
            size=gt.shape[2:],
            mode=mode,
            align_corners=align_corners)
        return x

    def shuffle(self, x_student, x_teacher, n_iter):
        interval = self.shuffle_config['interval']
        B, C, W, H = x_student.shape
        if n_iter % interval == 0:
            idx = torch.randperm(C)
            x_student = x_student[:, idx, :, :].contiguous()
            x_teacher = x_teacher[:, idx, :, :].contiguous()
        return x_student, x_teacher

    def transform(self, x):
        B, C, W, H = x.shape
        loss_type = self.transform_config['loss_type']
        if loss_type == 'pixel':
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, W * H, C)
        elif loss_type == 'channel':
            group_size = self.transform_config['group_size']
            if C % group_size == 0:
                x = x.reshape(B, C // group_size, -1)
            else:
                n = group_size - C % group_size
                x_pad = -1e9 * torch.ones(B, n, W, H).cuda()
                x = torch.cat([x, x_pad], dim=1)
                x = x.reshape(B, (C + n) // group_size, -1)
        return x

    def warmup(self, n_iter):
        mode = self.warmup_config['mode']
        warmup_iters = self.warmup_config['warmup_iters']
        if n_iter > warmup_iters:
            return
        elif n_iter == warmup_iters:
            self.alpha = self.alpha_0
            return
        else:
            if mode == 'linear':
                self.alpha = self.alpha_0 * (n_iter / warmup_iters)
            elif mode == 'exp':
                self.alpha = self.alpha_0 ** (n_iter / warmup_iters)
            elif mode == 'jump':
                self.alpha = 0

    def earlydecay(self, n_iter):
        mode = self.earlydecay_config['mode']
        earlydecay_start = self.earlydecay_config['earlydecay_start']
        earlydecay_end = self.earlydecay_config['earlydecay_end']

        if n_iter < earlydecay_start:
            return
        elif n_iter > earlydecay_start and n_iter < earlydecay_end:
            if mode == 'linear':
                self.alpha = self.alpha_0 * ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'exp':
                self.alpha = 0.001 * self.alpha_0 ** ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'jump':
                self.alpha = 0
        elif n_iter >= earlydecay_end:
            self.alpha = 0

    def forward(self, x_student, x_teacher, gt, n_iter):
        if self.warmup_config:
            self.warmup(n_iter)
        if self.earlydecay_config:
            self.earlydecay(n_iter)

        if self.resize_config:
            x_student, x_teacher = self.resize(x_student, gt), self.resize(x_teacher, gt)
        if self.shuffle_config:
            x_student, x_teacher = self.shuffle(x_student, x_teacher, n_iter)
        if self.transform_config:
            x_student, x_teacher = self.transform(x_student), self.transform(x_teacher)

        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)

        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        loss = self.alpha * loss
        return loss


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)
        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - dice_eff.sum() / N
        return loss


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class PDLoss(KLDLoss):
    def __init__(self):
        super().__init__()
        self.alpha_0 = 1
        self.alpha = 1
        self.tau = 1

        self.resize_config = {'mode': 'bilinear', 'align_corners': False}
        self.shuffle_config = None
        self.transform_config = {'loss_type': 'pixel'}
        self.warmup_config = None
        self.earlydecay_config = None

        self.KLD = torch.nn.KLDivLoss(reduction='sum')


class CDLoss(KLDLoss):
    def __init__(self):
        super().__init__()
        self.alpha_0 = 1
        self.alpha = 1
        self.tau = 1

        self.resize_config = {'mode': 'bilinear', 'align_corners': False}
        self.shuffle_config = None
        self.transform_config = {'loss_type': 'channel', 'group_size': 1}
        self.warmup_config = None
        self.earlydecay_config = None

        self.KLD = torch.nn.KLDivLoss(reduction='sum')


class CGDLoss(KLDLoss):
    def __init__(self, group_size=10, alpha=3, tau=2):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = {'mode': 'bilinear', 'align_corners': False}
        self.shuffle_config = {'interval': 1000}
        self.transform_config = {'loss_type': 'channel', 'group_size': group_size}
        self.warmup_config = None
        self.earlydecay_config = None

        self.KLD = torch.nn.KLDivLoss(reduction='sum')


class CGDLossWS(KLDLoss):
    def __init__(self):
        super().__init__()
        self.alpha_0 = 3
        self.alpha = 3
        self.tau = 2

        self.resize_config = {'mode': 'bilinear', 'align_corners': False}
        self.shuffle_config = {'interval': 1000}
        self.transform_config = {'loss_type': 'channel', 'group_size': 10}
        self.warmup_config = {'mode': 'linear', 'warmup_iters': 2000}
        self.earlydecay_config = {'mode': 'linear', 'earlydecay_start': 110000, 'earlydecay_end': 120000}

        self.KLD = torch.nn.KLDivLoss(reduction='sum')


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')

    def _resize(self, x, x_t):
        x = F.interpolate(
            input=x,
            size=x_t.shape[2:],
            mode='bilinear', align_corners=False)
        return x

    def forward(self, x_student, x_teacher, gt, step):
        # x_student = self._resize(x_student,x_teacher)
        loss_AT = self.mse(x_student.mean(dim=1), x_teacher.mean(dim=1))

        x_student = F.log_softmax(x_student, dim=1)
        x_teacher = F.softmax(x_teacher, dim=1)

        loss_PD = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[1])
        loss = loss_AT + loss_PD
        return loss


class IFVDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')

    def resize(self, x, x_t):
        x = F.interpolate(
            input=x,
            size=x_t.shape[2:],
            mode='bilinear', align_corners=False)
        return x

    def forward(self, preds_S, preds_T, target, step):
        feat_S = preds_S
        feat_T = preds_T
        feat_T = self.resize(feat_T, feat_S)
        feat_T.detach()

        C = feat_T.shape[1]
        x_student = F.log_softmax(feat_S, dim=1)
        x_teacher = F.softmax(feat_T, dim=1)
        loss_PD = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[1])

        size_f = (feat_S.shape[2], feat_S.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.float()).expand(feat_T.size())
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for i in range(C):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * (
                        (mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(
                -1).unsqueeze(-1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * (
                        (mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(
                -1).unsqueeze(-1)
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        loss_IFVD = 10 * self.mse(pcsim_feat_S, pcsim_feat_T)

        loss = loss_IFVD + loss_PD
        return loss

class DRKD(nn.Module):
    '''
    Relational Knowledge Distillation
    https://arxiv.org/pdf/1904.05068.pdf
    '''

    def __init__(self, weight, w_dist=25.0, w_angle=50.0, lambd=3.0, device='cuda:0'):
        super(DRKD, self).__init__()

        self.lamdb = lambd
        self.w_dist = w_dist
        self.w_angle = w_angle
        self.loss_base = nn.CrossEntropyLoss(weight=weight).to(device)

    def forward(self, out_s, out_t, target):
        feat_s = out_s['avg_pool']
        feat_t = out_t['avg_pool']
        logits_s = out_s['logits']
        loss = self.loss_base(logits_s, target)
        loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
            self.w_angle * self.rkd_angle(feat_s, feat_t) + self.lamdb * loss
        return loss

    def rkd_dist(self, feat_s, feat_t):
        with torch.no_grad():
            feat_t_dist = self.pdist(feat_t, squared=False)
            mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
            feat_t_dist = feat_t_dist / mean_feat_t_dist

        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist

        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

        return loss

    def rkd_angle(self, feat_s, feat_t):
        # N x C --> N x N x C
        with torch.no_grad():
            feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
            norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
            feat_t_angle = torch.bmm(
                norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

        feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(
            norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) +
                     feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist


def gram_matrix(features, normalize=True):
    N, C, H, W = features.shape
    feats = features.view(N, C, H * W)
    feats_t = feats.transpose(1, 2)
    G = feats.bmm(feats_t)

    if normalize:
        return G.div(C * H * W)
    else:
        return G


class D_KD(nn.Module):
    def __init__(self, weight, _layers, module_list, lambd=0.5, T=5,
                 lambd_rkd=1.0, w_dist=25.0, w_angle=50.0,
                 lambd_crkd=1e4, device='cuda:0'):
        super(D_KD, self).__init__()
        self.lambd = lambd
        self.lambd_rkd = lambd_rkd
        self.lambd_crkd = lambd_crkd
        self.T = T
        self.w_dist = w_dist
        self.w_angle = w_angle

        self.crit = nn.MSELoss()
        self.layers = _layers
        self.module_list = module_list
        self.loss_base = nn.CrossEntropyLoss(weight=weight).to(device)
        self.loss_blkd = nn.KLDivLoss(size_average=False).to(device)

    def forward(self, out_s, out_t, target):
        feat_s = out_s[self.layers['_layer_s']]
        feat_t = out_t[self.layers['_layer_t']]
        feat_s_conv = self.module_list[1](feat_s)

        pool_s = out_s['avg_pool']
        pool_t = out_t['avg_pool']

        logits_s = out_s['logits']
        logits_t = out_t['logits']

        batch_size = target.shape[0]
        s_max = F.log_softmax(logits_s / self.T, dim=1)
        t_max = F.softmax(logits_t / self.T, dim=1)

        loss_cls = self.loss_base(logits_s, target)
        loss_blkd = self.loss_blkd(s_max, t_max) / batch_size
        loss_rkd = self.w_dist * self.rkd_dist(pool_s, pool_t) + \
                   self.w_angle * self.rkd_angle(pool_s, pool_t)
        loss_crkd = self.cr_dist(feat_s_conv, feat_t)

        loss = (1 - self.lambd) * loss_cls + self.lambd * self.T * self.T * loss_blkd + \
               self.lambd_rkd * loss_rkd + self.lambd_crkd * loss_crkd
        return loss

    def cr_dist(self, feat_s, feat_t):
        gram_s = gram_matrix(feat_s)
        gram_t = gram_matrix(feat_t)
        cr_dist_tmp = F.mse_loss(gram_s, gram_t)
        return cr_dist_tmp.sqrt()

    def rkd_dist(self, feat_s, feat_t):
        with torch.no_grad():
            feat_t_dist = self.pdist(feat_t, squared=False)
            mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
            feat_t_dist = feat_t_dist / mean_feat_t_dist

        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist

        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

        return loss

    def rkd_angle(self, feat_s, feat_t):
        with torch.no_grad():
            feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
            norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
            feat_t_angle = torch.bmm(
                norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

        feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(
            norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) +
                     feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist


class DR(D_KD):
    def __init__(self, weight, _layers, module_list, lambd=0.5, T=5,
                 lambd_rkd=1.0, w_dist=25.0, w_angle=50.0,
                 lambd_crkd=1e4, device='cuda:0'):
        super(DR, self).__init__()
        self.lambd = lambd
        self.lambd_rkd = lambd_rkd
        self.lambd_crkd = lambd_crkd
        self.T = T
        self.w_dist = w_dist
        self.w_angle = w_angle

        self.layers = _layers
        self.module_list = module_list

    def forward(self, out_s, out_t, target=None):
        feat_s = out_s[self.layers['_layer_s']]
        feat_t = out_t[self.layers['_layer_t']]
        feat_s_conv = self.module_list[2](feat_s)

        pool_s = out_s['avg_pool']
        pool_t = out_t['avg_pool']

        loss_rkd = self.w_dist * self.rkd_dist(pool_s, pool_t) + \
                   self.w_angle * self.rkd_angle(pool_s, pool_t)
        loss_crkd = self.cr_dist(feat_s_conv, feat_t)

        loss = self.lambd_rkd * loss_rkd + self.lambd_crkd * loss_crkd
        return loss
# class KLDLoss(nn.Module):
#     def __init__(self,weight,tau,\
#         reshape_config=None,resize_config=None,mask_config=None,transform_config=None,ff_config=None,\
#         earlystop_config=None,shift_config=None,shuffle_config=None,warmup_config=0,edt_config=False):
#         super().__init__()
#         self.weight = weight
#         self.tau = tau
#         self.KLDiv = torch.nn.KLDivLoss(reduction='none')

#         self.reshape_config = reshape_config if reshape_config else False
#         self.resize_config = resize_config if resize_config else False
#         self.mask_config = mask_config if mask_config else False
#         self.transform_config = transform_config if transform_config else False

#         self.earlystop_config = earlystop_config if earlystop_config else False
#         self.shift_config = shift_config if shift_config else False
#         self.shuffle_config = shuffle_config if shuffle_config else False

#         self.edt = edt_config if edt_config else False
#         self.ff = nn.Conv2d(**ff_config,kernel_size=1).cuda() if ff_config else False

#         self.warmup_config = warmup_config if warmup_config > 0 else False

#         self.weight_ = weight


#     def _resize(self,x,gt_semantic_seg):
#         x = F.interpolate(
#             input=x,
#             size=gt_semantic_seg.shape[2:],
#             **self.resize_config)
#         return x

#     def _mask(self,x_student,x_teacher,gt_semantic_seg):
#         teacher_pd = torch.argmax(x_teacher,dim=1,keepdim=True)
#         student_pd = torch.argmax(x_student,dim=1,keepdim=True)
#         if not self.resize_config: # 对gt进行下采样，使size一致
#             gt_semantic_seg = F.interpolate(
#                 input=gt_semantic_seg,
#                 size=teacher_pd.shape[2:],
#                 mode='nearest'
#             )

#         Tr = (teacher_pd == gt_semantic_seg).bool() # teacher预测正确的pixel
#         Sr = (student_pd == gt_semantic_seg).bool() # student预测正确的pixel
#         Bg = (gt_semantic_seg == 255).bool() # 标签为background的pixel

#         TrSr = Tr&Sr
#         TrSf = Tr&(~Sr)
#         TfSr = (~Tr)&Sr
#         TfSf = (~Tr)&(~Sr)&(~Bg)
#         masks = {'TrSr':TrSr,'TrSf':TrSf,'TfSr':TfSr,'TfSf':TfSf,'Bg':Bg}

#         masks_to_apply = [masks[mask_name] for mask_name in self.mask_config] # 使用的mask
#         mask = masks_to_apply[0]
#         for m in masks_to_apply:
#             mask = mask | m
#         mask = mask.expand(-1,150,-1,-1)

#         return mask

#     def _reshape(self,x):
#         # 对任意输入，化为[B,C,W,H]的形式
#         if self.reshape_config == 'logits':
#             # [B,C,W,H]
#             x = x # 不用进行操作
#         elif self.reshape_config == 'attention':
#             # [B,num_head,WH,C]
#             B,num_head,WH,C = x.shape
#             W = int(np.sqrt(WH))
#             x = x.reshape(B*num_head,W,-1,C)
#             x = x.permute(0,3,1,2)
#         elif self.reshape_config == 'feature':
#             # [B,WH,C]
#             B,WH,C = x.shape
#             W = int(np.sqrt(WH))
#             x = x.reshape(B,W,-1,C)
#             x = x.permute(0,3,1,2)

#         return x

#     def _shift(self,x,step):
#         B,C,W,H = x.shape
#         shift_size = step
#         shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
#         return x
#     def _shuffle(self,x_student,x_teacher):
#         B,C,W,H = x_student.shape
#         idx = torch.randperm(C)
#         x_student = x_student[:,idx,:,:].contiguous()
#         x_teacher = x_teacher[:,idx,:,:].contiguous()
#         return x_student,x_teacher

#     def _ff(self,x):
#         return self.ff(x)

#     def _transform(self,x):
#         loss_type = self.transform_config['loss_type']
#         if loss_type == 'channel':
#             group_size = self.transform_config['group_size']
#             B,C,W,H = x.shape
#             if C%group_size == 0:
#                 x = x.reshape(B,C//group_size,-1)
#             else:
#                 n = group_size - C % group_size
#                 x_pad =-1e9 * torch.ones(B,n,W,H).cuda()
#                 x = torch.cat([x,x_pad],dim=1)
#                 x = x.reshape(B,(C+n)//group_size,-1)
#         elif loss_type == 'spatial':
#             kernel_size = self.transform_config['kernel_size']
#             stride = self.transform_config['stride']
#             padding = self.transform_config['padding'] if 'padding' in self.transform_config else 0
#             x = F.unfold(x,kernel_size, dilation=1, padding=padding, stride=stride)
#             x = x.transpose(2,1)
#         return x

#     def forward(self,x_student,x_teacher,gt_semantic_seg,step):

#         if self.warmup_config:
#             if step < self.warmup_config:
#                 self.weight = step/self.warmup_config*self.weight_
#             else:
#                 self.weight = self.weight_
#                 self.warmup_config = False
#         if self.earlystop_config:
#             if step > self.earlystop_config and step < self.earlystop_config+10000:
#                 self.weight = (self.earlystop_config+10000-step)/10000*self.weight_
#             elif step > self.earlystop_config+9999:
#                 self.weight = 0

#         if self.edt:
#             self.weight = self.weight_*(0.01)**(step/160000)

#         x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
#         if self.ff :
#             x_student = self._ff(x_student)
#         if self.resize_config:
#             x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
#             # print(x_student.shape,x_teacher.shape)
#         if self.mask_config:
#             mask = self._mask(x_student,x_teacher,gt_semantic_seg)
#         if self.shift_config:
#             x_student,x_teacher = self._shift(x_student,step),self._shift(x_teacher,step)
#         if self.shuffle_config:
#             x_student,x_teacher = self._shuffle(x_student,x_teacher)

#         if self.transform_config:
#             x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
#             if self.mask_config:
#                 mask = self._transform(mask)
#             else:
#                 mask = torch.zeros_like(x_student).cuda()
#         x_student = F.log_softmax(x_student/self.tau,dim=-1)
#         x_teacher = F.softmax(x_teacher/self.tau,dim=-1)

#         loss = self.KLDiv(x_student,x_teacher)*(1-mask.float())
#         # print(self.weight)
#         loss = self.weight*loss.sum()/(x_student.numel()/x_student.shape[-1])
#         return loss

# class ShiftChannelLoss(KLDLoss):
#     def __init__(self,weight,tau,\
#         reshape_config,resize_config,mask_config,transform_config,ff_config,\
#         earlystop_config=None):
#         super().__init__(weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config,earlystop_config)
#     def _transform(self,x):
#         B,C,W,H = x.shape
#         x = x.reshape(B,C,W*H,1).permute(0,2,1,3)
#         x = F.unfold(x,kernel_size=(self.transform_config['group_size'],1)).transpose(1,2)
#         return x
#     def forward(self,x_student,x_teacher,gt_semantic_seg,step):
#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0
#         x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
#         if self.ff :
#             x_student = self._ff(x_student)
#         if self.resize_config:
#             x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
#         if self.mask_config:
#             x_student,x_teacher = self._mask(x_student,x_teacher,gt_semantic_seg)
#         if self.transform_config:
#             x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)

#         x_student = F.log_softmax(x_student/self.tau,dim=-1)
#         x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
#         loss = self.weight*self.KLDiv(x_student,x_teacher)
#         loss = loss/(x_student.numel()/x_student.shape[-1])
#         return loss

# class ShuffleChannelLoss(KLDLoss):
#     def __init__(self,weight,tau,\
#         reshape_config,resize_config,mask_config,transform_config,ff_config,\
#         earlystop_config=None,shuffle_config=None):
#         self.weight_ = weight
#         self.shuffle_config = 1
#         super().__init__(weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config,earlystop_config)
#     def _shuffle(self,x,step):
#         B,N,G = x.shape
#         if self.shuffle_config != 0:
#             if step % self.shuffle_config == 0:
#                 x = x.transpose(1,2)
#                 x = x.reshape(B,-1)
#                 x = x.reshape(B,N,G)
#         else:
#             x = x.transpose(1,2)
#             x = x.reshape(B,-1)
#             x = x.reshape(B,N,G)
#         return x


#     def forward(self,x_student,x_teacher,gt_semantic_seg,step):
#         if step < self.warmup_config:
#             self.weight = step/self.warmup_config
#         else:
#             self.weight = self.weight_

#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0

#         x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
#         if self.ff :
#             x_student = self._ff(x_student)
#         if self.resize_config:
#             x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
#         if self.mask_config:
#             x_student,x_teacher = self._mask(x_student,x_teacher,gt_semantic_seg)
#         if self.transform_config:
#             x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
#         x_student,x_teacher = self._shuffle(x_student,step),self._shuffle(x_teacher,step)

#         x_student = F.log_softmax(x_student/self.tau,dim=-1)
#         x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
#         loss = self.weight*self.KLDiv(x_student,x_teacher)
#         loss = loss/(x_student.numel()/x_student.shape[-1])
#         return loss

# class ShuffleShiftLoss(KLDLoss):
#     def __init__(self,weight,tau,\
#         reshape_config,resize_config,mask_config,transform_config,ff_config,\
#         earlystop_config=None):
#         super().__init__(weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config,earlystop_config)
#     def _transform(self,x):
#         B,C,W,H = x.shape
#         x = x.reshape(B,C,W*H,1).permute(0,2,1,3)
#         x = F.unfold(x,kernel_size=(self.transform_config['group_size'],1)).transpose(1,2)
#         return x
#     def _shuffle(self,x_student,x_teacher):
#         B,C,W,H = x_student.shape
#         idx = torch.randperm(C)
#         x_student = x_student[:,idx,:,:].contiguous()
#         x_teacher = x_teacher[:,idx,:,:].contiguous()
#         return x_student,x_teacher
#     def forward(self,x_student,x_teacher,gt_semantic_seg,step):
#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0
#         x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
#         if self.ff :
#             x_student = self._ff(x_student)
#         if self.resize_config:
#             x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
#         if self.mask_config:
#             x_student,x_teacher = self._mask(x_student,x_teacher,gt_semantic_seg)
#         x_student,x_teacher = self._shuffle(x_student,x_teacher)
#         if self.transform_config:
#             x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)

#         x_student = F.log_softmax(x_student/self.tau,dim=-1)
#         x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
#         loss = self.weight*self.KLDiv(x_student,x_teacher)
#         loss = loss/(x_student.numel()/x_student.shape[-1])
#         return loss


# class AttentionLoss(nn.Module):
#     def __init__(self,weight,tau,transform_config,earlystop_config,warmup_config):
#         super().__init__()
#         self.weight = weight
#         self.weight_ = weight
#         self.tau = tau
#         self.transform_config = transform_config
#         self.earlystop_config = earlystop_config if earlystop_config else False
#         self.warmup_config = warmup_config

#         self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

#     def _transform(self,x):
#         loss_type = self.transform_config['loss_type']
#         B,N,C = x.shape
#         if loss_type == 'channel':
#             group_size = self.transform_config['group_size']
#             x = x.permute(0,2,1)
#             x = x.reshape(B,C//group_size,-1)
#         elif loss_type == 'spatial':
#             pass
#         return x
#     def forward(self,attn_student,v_student,attn_teacher,v_teacher,gt,step):
#         if step < self.warmup_config:
#             self.weight = step/self.warmup_config
#         else:
#             self.weight = self.weight_

#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0

#         B,num_head,_,C = v_student.shape
#         _,_,N,_ = attn_student.shape
#         attn_student  = attn_student.softmax(dim=-1)
#         attn_teacher  = attn_teacher.softmax(-1)
#         C = C * num_head


#         print((attn_student @ v_student).transpose(1, 2).shape)
#         print((teacher @ v_student).transpose(1, 2).shape)
#         x = (attn_student @ v_student).transpose(1, 2).reshape(B, N, C)
#         x_mimic = (attn_teacher @ v_student).transpose(1, 2).reshape(B, N, C)

#         x,x_mimic = self._transform(x),self._transform(x_mimic)
#         x = F.log_softmax(x/self.tau,dim=-1)
#         x_mimic = F.softmax(x_mimic/self.tau,dim=-1)
#         loss = self.weight*self.KLDiv(x,x_mimic)
#         loss = loss/(x.numel()/x.shape[-1])
#         return loss


# class StudentRE(nn.Module):
#     def __init__(self,weight,tau,transform_config,proj,earlystop_config,**kwargs):
#         super().__init__()
#         self.weight = weight
#         self.weight_ = weight
#         self.tau = tau
#         self.transform_config = transform_config
#         self.earlystop_config = earlystop_config if earlystop_config else False
#         self.KLDiv = torch.nn.KLDivLoss(reduction='sum')
#         self.proj_name = proj

#     def _transform(self,x):
#         loss_type = self.transform_config['loss_type']
#         B,N,C = x.shape
#         if loss_type == 'channel':
#             group_size = self.transform_config['group_size']
#             x = x.permute(0,2,1)
#             x = x.reshape(B,C//group_size,-1)
#         elif loss_type == 'spatial':
#             pass
#         return x
#     def forward(self,attn_student,v_student,attn_teacher,v_teacher,student,teacher,gt,step):

#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0
#         v_student = v_student.detach()
#         B,num_head,_,C = v_student.shape
#         _,_,N,_ = attn_student.shape
#         attn_student  = attn_student.softmax(dim=-1)
#         attn_teacher  = attn_teacher.softmax(dim=-1)
#         C = C * num_head

#         x = (attn_student @ v_student).transpose(1, 2).reshape(B, N, C)
#         x_mimic = (attn_teacher @ v_student).transpose(1, 2).reshape(B, N, C)


#         for name,v in student.named_parameters():
#             if self.proj_name in name:
#                 if 'weight' in name:
#                     W = v
#                 else:
#                     b = v

#         # x,x_mimic = self._transform(x),self._transform(x_mimic)
#         x = F.linear(x, W, b)
#         x_mimic = F.linear(x_mimic, W, b)
#         # x = F.log_softmax(x/self.tau,dim=-1)
#         # x_mimic = F.softmax(x_mimic/self.tau,dim=-1)
#         # loss = self.weight*self.KLDiv(x,x_mimic)
#         # loss = loss/(x.numel()/x.shape[-1])
#         loss = self.weight*torch.mean((x-x_mimic)**2)
#         return loss

# class TeacherRE(nn.Module):
#     def __init__(self,weight,tau,transform_config,proj,earlystop_config,**kwargs):
#         super().__init__()
#         self.weight = weight
#         self.weight_ = weight
#         self.tau = tau
#         self.transform_config = transform_config
#         self.earlystop_config = earlystop_config if earlystop_config else False

#         self.proj_name = proj

#         self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

#     def _transform(self,x):
#         loss_type = self.transform_config['loss_type']
#         B,N,C = x.shape
#         if loss_type == 'channel':
#             group_size = self.transform_config['group_size']
#             x = x.permute(0,2,1)
#             x = x.reshape(B,C//group_size,-1)
#         elif loss_type == 'spatial':
#             pass
#         return x
#     def forward(self,attn_student,v_student,attn_teacher,v_teacher,student,teacher,gt,step):

#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0

#         B,num_head,_,C = v_teacher.shape
#         _,_,N,_ = attn_teacher.shape
#         attn_student  = attn_student.softmax(dim=-1)
#         attn_teacher  = attn_teacher.softmax(-1)
#         C = C * num_head

#         for name,v in teacher.named_parameters():
#             if self.proj_name in name:
#                 if 'weight' in name:
#                     W = v
#                 else:
#                     b = v

#         # print((attn_student @ v_student).transpose(1, 2).shape)
#         # print((teacher @ v_student).transpose(1, 2).shape)
#         x = (attn_teacher @ v_teacher).transpose(1, 2).reshape(B, N, C)
#         x_mimic = (attn_student @ v_teacher).transpose(1, 2).reshape(B, N, C)
#         x = F.linear(x, W, b)
#         x_mimic = F.linear(x_mimic, W, b)
#         loss = self.weight * torch.mean((x-x_mimic)**2)
#         # x,x_mimic = self._transform(x),self._transform(x_mimic)
#         # x = F.log_softmax(x/self.tau,dim=-1)
#         # x_mimic = F.softmax(x_mimic/self.tau,dim=-1)
#         # loss = self.weight*self.KLDiv(x,x_mimic)
#         # loss = loss/(x.numel()/x.shape[-1])

#         return loss

# class FlattenLoss(nn.Module):
#     def __init__(self,weight,tau,transform_config,earlystop_config,**kwargs):
#         super().__init__()
#         self.weight = weight
#         self.weight_ = weight
#         self.tau = tau
#         self.transform_config = transform_config
#         self.earlystop_config = earlystop_config if earlystop_config else False

#         self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

#     def forward(self,x_student,x_teacher,gt,step):
#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0

#         B,WH,C_s = x_student.shape
#         B,WH,C_t = x_teacher.shape

#         x_student = F.log_softmax(x_student/self.tau,dim=1)
#         x_teacher = F.softmax(x_teacher/self.tau,dim=1)

#         x_student = x_student.mean(dim=2)
#         x_teacher = x_teacher.mean(dim=2)

#         loss = self.weight*self.KLDiv(x_student,x_teacher)
#         loss = loss/(x_student.numel()/x_student.shape[-1])
#         return loss


# class MTLoss(KLDLoss):
#     def __init__(self,weight,tau,reshape_config,resize_config,transform_config,latestart_config,earlystop_config,rot_config=None,**kwargs):
#         super().__init__(weight,tau,reshape_config=reshape_config,resize_config=resize_config,\
#             transform_config=transform_config)
#         self.weight_ = weight

#         self.latestart_config = latestart_config if latestart_config else False
#         self.earlystop_config = earlystop_config if earlystop_config else False
#         self.rot = rot_config if rot_config else False

#         self.KLDiv = torch.nn.KLDivLoss(reduction='sum')
#     def forward(self,x_student,x_teacher,gt,step):
#         if self.latestart_config:
#             if step < self.latestart_config:
#                 self.weight = 0
#             else:
#                 self.weight = self.weight_

#         if self.rot:
#             i,interval = self.rot
#             if ((step-1) // interval) % 4 == i:
#                 self.weight = self.weight_
#             else:
#                 self.weight = 0

#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0

#         x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
#         if self.resize_config:
#             x_student,x_teacher = self._resize(x_student,gt),self._resize(x_teacher,gt)
#         if self.transform_config:
#             x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)

#         x_student = F.log_softmax(x_student/self.tau,dim=-1)
#         x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
#         loss = self.weight*self.KLDiv(x_student,x_teacher)
#         loss = loss/(x_student.numel()/x_student.shape[-1])
#         return loss


# class MTRandomLoss(KLDLoss):
#     def __init__(self,weight,tau,reshape_config,resize_config,transform_config,latestart_config,earlystop_config,rot_config=None,**kwargs):
#         super().__init__(weight,tau,reshape_config=reshape_config,resize_config=resize_config,\
#             transform_config=transform_config)
#         self.weight_ = weight

#         self.latestart_config = latestart_config if latestart_config else False
#         self.earlystop_config = earlystop_config if earlystop_config else False
#         self.rot = rot_config if rot_config else False

#         self.KLDiv = torch.nn.KLDivLoss(reduction='sum')
#     def forward(self,x_student,x_teacher,gt,step):
#         if self.latestart_config:
#             if step < self.latestart_config:
#                 self.weight = 0
#             else:
#                 self.weight = self.weight_
#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0

#         if self.rot:
#             i,interval = self.rot
#             if ((step-1) // interval) % 4 == i:
#                 self.weight = self.weight_
#             else:
#                 self.weight = 0

#         # N_teacher,B,C,W,H = x_teacher.shape
#         # teacher_mask = torch.randint(0,2,[N_teacher,B,C]).cuda()
#         # cnt = teacher_mask.sum()
#         # teacher_mask = .expand(N_teacher,B,C,W,H).cuda() # [N_teacher,B,C,W,H]
#         # x_teacher = (x_teacher*teacher_mask).sum(dim=0)
#         # x_teacher = x_teacher / (teacher_mask.sum(dim=0))

#         x_student = self._reshape(x_student)
#         x_student = self._resize(x_student,gt)
#         x_student = self._transform(x_student)

#         x_teachers = []
#         for x in x_teacher:
#             x = self._reshape(x)
#             x = self._resize(x,gt)
#             x = self._transform(x)
#             x_teachers.append(x)
#         x_teacher = torch.cat([i.unsqueeze(0) for i in x_teachers],dim=0)
#         num_teachers,B,C,N = x_teacher.shape

#         teacher_mask = torch.randint(0,2,[num_teachers,B,C]).cuda()
#         teacher_cnt = teacher_mask.sum(dim=0).reshape(B,C,1).expand(B,C,N)
#         teacher_mask = teacher_mask.reshape(num_teachers,B,C,1)
#         x_teacher = (x_teacher*teacher_mask).sum(dim=0)
#         x_teacher = x_teacher / teacher_cnt
#         x_teacher[teacher_cnt==0] = x_student[teacher_cnt==0].clone()

#         x_student = F.log_softmax(x_student/self.tau,dim=-1)
#         x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
#         loss = self.weight*self.KLDiv(x_student,x_teacher)
#         loss = loss/(x_student.numel()/x_student.shape[-1])
#         return loss

# class MSE(nn.Module):
#     def __init__(self,weight,\
#         reshape_config=None,resize_config=None,mask_config=None,transform_config=None,ff_config=None,\
#         earlystop_config=None,shift_config=None,warmup_config=0):
#         super().__init__()
#         self.weight = weight
#         self.earlystop_config = earlystop_config
#         self.ff = nn.Conv1d(**ff_config,kernel_size=1).cuda() if ff_config else False
#         self.MSE = nn.MSELoss()
#     def forward(self,x_student,x_teacher,gt_semantic_seg,step):
#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0
#         if self.ff:
#             x_student = x_student.transpose(1,2)
#             x_student = self.ff(x_student)
#             x_student = x_student.transpose(1,2)
#         loss = self.weight*torch.mean((x_student-x_teacher)**2)
#         return loss


# class Mimic(nn.Module):
#     def __init__(self,weight,tau,\
#         reshape_config=None,resize_config=None,mask_config=None,transform_config=None,ff_config=None,\
#         earlystop_config=None,shift_config=None,warmup_config=0):
#         super().__init__()
#         self.weight = weight
#         self.earlystop_config = earlystop_config
#         self.ff = nn.Conv1d(**ff_config,kernel_size=1).cuda() if ff_config else False
#         self.MSE = nn.MSELoss()
#     def forward(self,x_student,x_teacher,gt_semantic_seg,step):
#         if self.earlystop_config:
#             if step > self.earlystop_config:
#                 self.weight = 0
#         x_student = x_student.transpose(1,2)
#         x_student = self.ff(x_student)
#         x_student = x_student.transpose(1,2)
#         loss = self.weight*torch.mean((x_student-x_teacher)**2)
#         return loss

# # class IFVD(nn.Module):
# #     def __init__(self,weight,tau,\
# #         reshape_config=None,resize_config=None,mask_config=None,transform_config=None,ff_config=None,\
# #         earlystop_config=None,shift_config=None,warmup_config=0):
# #         super().__init__()
# #         self.weight = weight
# #         self.earlystop_config = earlystop_config
# #         self.ff = nn.Conv1d(**ff_config,kernel_size=1).cuda() if ff_config else False

# #     def avg_pooling(self,x):


# class IFVD(nn.Module):
#     def __init__(self, classes=150,**kwargs):
#         super().__init__()
#         self.num_classes = classes
#         self.mse = nn.MSELoss(reduce='mean')

#     def _resize(self,x,x_t):
#         x = F.interpolate(
#             input=x,
#             size=x_t.shape[2:],
#             mode='bilinear',align_corners=False)
#         return x

#     def forward(self, preds_S, preds_T, target,step):
#         # print(preds_S.shape, preds_T.shape)
#         feat_S = preds_S
#         feat_T = preds_T
#         feat_S = self._resize(feat_S,feat_T)
#         feat_T.detach()
#         size_f = (feat_S.shape[2], feat_S.shape[3])
#         tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.float()).expand(feat_S.size())
#         tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.float()).expand(feat_T.size())
#         center_feat_S = feat_S.clone()
#         center_feat_T = feat_T.clone()
#         for i in range(self.num_classes):
#           mask_feat_S = (tar_feat_S == i).float()
#           mask_feat_T = (tar_feat_T == i).float()
#           center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
#           center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

#         # cosinesimilarity along C
#         cos = nn.CosineSimilarity(dim=1)
#         pcsim_feat_S = cos(feat_S, center_feat_S)
#         pcsim_feat_T = cos(feat_T, center_feat_T)

#         # mseloss

#         loss = 100*self.mse(pcsim_feat_S, pcsim_feat_T)
#         return loss


# class AT(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss(reduce='mean')

#     def _resize(self,x,x_t):
#         x = F.interpolate(
#             input=x,
#             size=x_t.shape[2:],
#             mode='bilinear',align_corners=False)
#         return x
#     def forward(self,x_student, x_teacher, gt,step):
#         x_student = self._resize(x_student,x_teacher)
#         x_student = x_student.mean(dim=1)
#         x_teacher = x_teacher.mean(dim=1)
#         loss = 0.1*self.mse(x_student, x_teacher)
#         return loss
