from .base_actor_diss import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from torchvision import transforms
from lib.train.actors.kdloss import DistillKL
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.utils import save_image

class AttnDiV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reize = transforms.Resize([12, 12])
        self.C = 768
        self.kdloss = DistillKL(1)

    def forward(self, student, teacher):
        loss = 0
        for i in range(12):
            teacher["attn"][i] = self.reize(teacher["attn"][i][:, 144:].unsqueeze(-1).view(-1, self.C, 24, 24))

            student["attn"][i] = student["attn"][i][:, 36:].unsqueeze(-1).view(-1, self.C, 12, 12)

            loss = loss + self.kdloss(student["attn"][i], teacher["attn"][i])

        return loss

class FeatureLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_feature = transforms.Resize([12, 12])
        self.kdloss = DistillKL(1)

    def forward(self, student, teacher):
        # backbone_student1 = student['opt_feat']
        # backbone_teacher1 = teacher['opt_feat']
        # backbone_student = backbone_student1 +
        backbone_teacher = self.resize_feature(teacher['opt_feat'])
        backbone_student = student['opt_feat']
        kd_loss = self.kdloss(backbone_student, backbone_teacher)

        return kd_loss
class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, net_teacher, objective, loss_weight, settings, cfg=None):
        super().__init__(net, net_teacher, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.resize_template = transforms.Resize([96, 96])
        self.resize_search = transforms.Resize([192, 192])
        self.kdloss = FeatureLoss()
        self.attloss = AttnDiV()


    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        out_dict_high = self.forward_pass_teacher(data)

        # torch.Size([1, 32, 4]) 11111111111111111111111111111111
        # torch.Size([1, 32, 4]) 22222222222222222222222222
        # print(data['search_anno'].shape, "11111111111111111111111111111111")
        # print(data['template_anno'].shape, "22222222222222222222222222")
        # print(data['search_images'].shape, "333333333333333333")
        # print(data['template_images'].shape, "44444444444444444444")

        data['template_images'] = self.resize_template(data['template_images'].squeeze(0)).unsqueeze(0)
        data['search_images'] = self.resize_search(data['search_images'].squeeze(0)).unsqueeze(0)
        # images = self.trans(data['search_images'].squeeze(0).squeeze(0))
        # images.save('/home/UNT/sd1260/OSTrack/lib/train/actors/saveimage/image1.png')

        # print(data['search_images'].shape, "2222222222222222222222")
        # save_image(data['search_images'], '/home/UNT/sd1260/OSTrack/lib/train/actors/saveimage/img1.png')
        # print(data['template_images'].shape, "11111111111111111")

        out_dict = self.forward_pass(data)
        # print(out_dict['attn'].shape, "111111111111111111111111111111")
        # print(out_dict_high['attn'].shape, "222222222222222222222222222")

        # out_dict['pred_boxes'] 'score_map' 'size_map' 'offset_map' 'attn'  'backbone_feat' 'opt_feat'
        # data['template_images'] 'template_anno' 'template_masks' 'search_images' 'search_anno' 'search_masks' 'dataset',
        # 'test_class' 'template_att' 'search_att' 'valid' 'epoch' 'settings'
        # print(out_dict['score_map'].shape, "1111111111111111111111111")  # ([32, 1, 12, 12]) (192)
        # # ([32, 1, 24, 24]) (284)
        # compute losses

        loss, status = self.compute_losses(out_dict, out_dict_high, data)
        # loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict
    def forward_pass_teacher(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net_teacher(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, out_dict_high, gt_dict, return_status=True):
        # gt gaussian map
        # gt_dict['search_anno'][-1][1] = gt_dict['search_anno'][-1][1] / 4.0
        # print(gt_dict['search_anno'][-1][1], "1111111111111111111111111111111")
        # print(gt_dict['search_anno'][-1], "1111111111111111")
        # print(gt_dict['search_anno'], "000000000000000000000")

        gt_dict['search_anno'] = gt_dict['search_anno'] / 2.0
        # print(gt_dict['search_anno'], "1111111111111111111")
        # print(gt_dict['search_anno'][-1], "222222222222222222")

        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)

        # print(gt_bbox, "11111111111111111111111111111111111111111111")
        # change the size (384->192)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE // 2, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)


        # kd_loss = self.kdloss(pred_dict, out_dict_high)
        loss_att = self.attloss(pred_dict, out_dict_high)

        # weighted sum
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + 0.01 * kd_loss
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + 0.01 * loss_att
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      # "Loss/kd": kd_loss.item(),
                      "Loss/kd": loss_att.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
