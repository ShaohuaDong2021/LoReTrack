"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
import seaborn as sns
from PIL import Image, ImageDraw
from torchvision import transforms
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, build_box_head_teacher
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                 std=[1., 1., 1.]),
                                            ])

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        # x, aux_dict, aux_q, aux_k, aux_v, aux_attn_out = self.backbone(z=template, x=search,
        #                             ce_template_mask=ce_template_mask,
        #                             ce_keep_rate=ce_keep_rate,
        #                             return_last_attn=return_last_attn)

        x, aux_dict, aux_q, aux_k, aux_v, aux_attn_out, attn_out = self.backbone(z=template, x=search,
                                                                                 ce_template_mask=ce_template_mask,
                                                                                 ce_keep_rate=ce_keep_rate,
                                                                                 return_last_attn=return_last_attn)


        # weight = torch.matmul(self.lora_query_matrix_B, self.lora_query_matrix_A)
        # x = x.permute(0, 2, 1) + torch.matmul(final_feature.permute(0, 2, 1), weight)
        # x = x.permute(0, 2, 1)

        ################################# here we get the query attention map ###################################################

        # print(search.shape, "111111111111111111111")
        inv_tensor = self.invTrans(search)
        tensor = inv_tensor.cpu().clone()
        tensor = tensor.squeeze(0)
        tensor = tensor.permute(1, 2, 0)
        image = tensor.numpy()
        image = (image * 255).astype(np.uint8)

        image = Image.fromarray(image)
        import time

        time_3 = time.time()
        image.save('/home/UNT/sd1260/OSTrack_new4A6000_2/assets/yoyo-19-image/image_%s.png' % time_3)

        # print(len(aux_attn_out), "0000000000000000000")
        # print(attn_out.shape, "111111111111")  # torch.Size([1, 12, 256, 64])

        # here we save the last x

        # print(x.shape, "11111111111")  
        # print(attn_out.shape, "22222222222")
        # attn_out = x[:, 64:, :]  
        

        # attn_out = attn_out.permute(0,2,1) 
        # attn_out = torch.mean(attn_out, dim=1)  
        # attn_out = attn_out.squeeze(0).view(16, 16)  
        # aux_attn_out = attn_out.cpu().numpy()
        # pow_mean_att = aux_attn_out
        # plt.figure(figsize=(8, 8))
        # sns.set()  # Set seaborn default style
        # ax = sns.heatmap(pow_mean_att, annot=False, fmt=".2f", cmap="viridis", cbar=False)
        # ax.set_axis_off()
        # plt.tight_layout(pad=0)

        # ax.figure.canvas.draw()
        # heatmap_array = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
        # heatmap_array = heatmap_array.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
        # time_3 = "example"  # Replace with actual timestamp or identifier
        # plt.imsave(f'/home/UNT/sd1260/OSTrack_new4A6000_2/assets/att_map1/pow_mean_att3_{time_3}.png', heatmap_array)



        # here is the attention

        # attn_out = attn_out.permute(0,2,1,3).flatten(2, 3)  
        # attn_out = torch.mean(attn_out, 2)
        # attn_out = attn_out.squeeze(0).view(16, 16)
        # aux_attn_out = attn_out.cpu().numpy()
        # pow_mean_att = aux_attn_out
        # plt.figure(figsize=(8, 8))
        # sns.set()  # Set seaborn default style
        # ax = sns.heatmap(pow_mean_att, annot=False, fmt=".2f", cmap="viridis", cbar=False)
        # ax.set_axis_off()
        # plt.tight_layout(pad=0)
        # ax.figure.canvas.draw()
        # heatmap_array = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
        # heatmap_array = heatmap_array.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
        # plt.imsave('/home/UNT/sd1260/OSTrack_new4A6000_2/assets/att_map1/pow_mean_att3_%s.png' % time_3, heatmap_array)


        #############################################################################################################################

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        # out = self.forward_head(feat_last, None)
        out = self.forward_head(feat_last, uuid=time_3, attn_out=attn_out, gt_score_map=None)

        out.update(aux_dict)
        out.update(aux_q)
        out.update(aux_k)
        out.update(aux_v)
        out.update(aux_attn_out)

        out['backbone_feat'] = x
        return out

    # def forward_head(self, cat_feature, gt_score_map=None):
    def forward_head(self, cat_feature, uuid, attn_out, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            out['opt_feat'] = opt_feat


            # print(outputs_coord_new.shape, "1111111111111")  # [32, 1, 4])
            # print(score_map_ctr.shape, "222222222222222")  # 32, 1, 12, 12
            # print(size_map.shape, "333333333333333")   # 32, 2, 12, 12
            # print(offset_map.shape, "44444444444444")  # 32, 2, 12, 12
            # print(opt_feat.shape, "555555555555555555555")  # 32, 768, 12, 12]
            
            
            
            
            ##################################### here we get the cls map ##########################################

            # score_map_ctr = score_map_ctr.squeeze(0).squeeze(0)
            # score_map_ctr = score_map_ctr.cpu().numpy()
            # pow_mean_att = score_map_ctr
            # plt.figure(figsize=(8, 8))
            # sns.set()  # Set seaborn default style
            # ax = sns.heatmap(pow_mean_att, annot=False, fmt=".2f", cmap="viridis", cbar=False)
            # ax.set_axis_off()
            # plt.tight_layout(pad=0)
            # ax.figure.canvas.draw()
            # heatmap_array = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
            # heatmap_array = heatmap_array.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))

            # plt.imsave('/home/UNT/sd1260/OSTrack_new4A6000_2/assets/att_map2_cls/pow_mean_att3_%s.png' % uuid,heatmap_array)
            
            ###########################################################################################################



            ############################# here we get the att map #############################################

            
            # attn_out = attn_out.permute(0,2,1,3).flatten(2, 3)  
    
            attn_out = attn_out.permute(0,2,1,3).flatten(2, 3)  
            attn_out = torch.mean(attn_out, 2)
            attn_out = attn_out.squeeze(0).view(16, 16)
            aux_attn_out = attn_out.cpu().numpy()
            pow_mean_att = aux_attn_out
            plt.figure(figsize=(8, 8))
            sns.set()  # Set seaborn default style
            ax = sns.heatmap(pow_mean_att, annot=False, fmt=".2f", cmap="viridis", cbar=False)
            ax.set_axis_off()
            plt.tight_layout(pad=0)
            ax.figure.canvas.draw()
            heatmap_array = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
            heatmap_array = heatmap_array.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
            plt.imsave('/home/UNT/sd1260/OSTrack_new4A6000_2/assets/yoyo-19-att/pow_mean_att3_%s.png' % uuid, heatmap_array)


            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    # model_dict = model.state_dict()
    #
    # pretrained_dict = torch.load(
    #     "/home/UNT/sd1260/OSTrack/output_pretrained384/checkpoints/train/ostrack_previous.py/vitb_384_mae_32x4_ep300/OSTrack_ep0300.pth.tar",
    #     map_location="cpu")
    #
    # pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if (k in model_dict and 'backbone.pos_embed_z' not in k and
    #                                                                        'backbone.pos_embed_x' not in k)}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)
    # print('Load pretrained model from 384 pretrained model')
    #
    # for n, p in model.named_parameters():
    #     p.requires_grad = True

    # for name, param in model_teacher.named_parameters():
    #     print("{} {}".format(name, param.requires_grad))

    return model


def build_ostrack_teacher(cfg, training=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        # here, how to load the 384 weight
        backbone_teacher = vit_base_patch16_224(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone_teacher.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone_teacher.finetune_track_teacher(cfg=cfg, patch_start_index=patch_start_index)

    box_head_teacher = build_box_head_teacher(cfg, hidden_dim)

    model_teacher = OSTrack(
        backbone_teacher,
        box_head_teacher,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    model_dict = model_teacher.state_dict()
    pretrained_dict = torch.load(
        "/home/UNT/sd1260/OSTrack/output_pretrained384/checkpoints/train/ostrack/vitb_384_mae_32x4_ep300/OSTrack_ep0300.pth.tar", map_location="cpu")

    model_teacher.load_state_dict(pretrained_dict["net"], strict=True)
    print('Load pretrained model from 384 pretrained model')

    for n, p in model_teacher.named_parameters():
        p.requires_grad = False

    # for name, param in model_teacher.named_parameters():
    #     print("{} {}".format(name, param.requires_grad))

    return model_teacher