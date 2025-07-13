from .base_actor_lora import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from torchvision import transforms
from lib.train.actors.kdloss import DistillKL
import matplotlib.pyplot as plt
# from ourtatloss import Distiller
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn

# class attloss_map(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.C = 768
#         self.distiller = Distiller()
#
#
#     def forward(self, student, teacher):
#         loss = 0
#         for i in range(12):
#             p_student = student["attn"][i][:, 36:].unsqueeze(-1).view(-1, self.C, 12, 12)
#             p_student = torch.nn.functional.interpolate(p_student, scale_factor=2,
#                                                         mode='bilinear')
#             p_teacher = teacher["attn"][i][:, 144:].unsqueeze(-1).view(-1, self.C, 24, 24)
#
#             loss = loss + self.distiller(p_student, p_teacher)
#             # loss = loss + self.kdloss_mse(teacher["attn"][i], student["attn"][i])
#
#         return loss
class AttnDiV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 768
        self.kdloss = DistillKL(0.001)
        self.kdloss_mse = torch.nn.MSELoss()

    def forward(self, student, teacher):
        loss = 0
        for i in range(12):
            p_student = student["attn"][i][:, 36:].unsqueeze(-1).view(-1, self.C, 12, 12)
            p_student = torch.nn.functional.interpolate(p_student, scale_factor=2,
                                                        mode='bilinear')
            p_teacher = teacher["attn"][i][:, 144:].unsqueeze(-1).view(-1, self.C, 24, 24)

            loss = loss + self.kdloss(p_student, p_teacher)
            # loss = loss + self.kdloss_mse(teacher["attn"][i], student["attn"][i])

        return loss


class AttnDiV_qkv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 768
        self.kdloss = DistillKL(0.001)
        self.kdloss_mse = torch.nn.MSELoss()

    def forward(self, student, teacher):
        loss = 0
        for i in range(12):
            q_student = student["attn_q"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1, self.C,12, 12)
            q_student = torch.nn.functional.interpolate(q_student, scale_factor=2, mode='bilinear')

            k_student = student["attn_k"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1, self.C,12, 12)
            k_student = torch.nn.functional.interpolate(k_student, scale_factor=2, mode='bilinear')

            v_student = student["attn_v"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1, self.C,12, 12)
            v_student = torch.nn.functional.interpolate(v_student, scale_factor=2, mode='bilinear')


            q_teacher = teacher["attn_q"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C, 24, 24)
            k_teacher = teacher["attn_k"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C, 24, 24)
            v_teacher = teacher["attn_v"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C, 24, 24)


            loss = loss + self.kdloss(q_student, q_teacher) + self.kdloss(k_student, k_teacher) + self.kdloss(v_student, v_teacher)
            # loss = loss + self.kdloss_mse(teacher["attn"][i], student["attn"][i])

        return loss



# class FeatureLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.kdloss = DistillKL(1)
#         self.kdloss = DistillKL(0.001)
#         self.C = 768
#         # self.kdloss_mse = torch.nn.MSELoss()
#
#     def forward(self, student, teacher):
#         backbone_teacher = teacher['opt_feat']  # teacher 1
#         p_teacher = teacher["attn"][11][:, 144:].unsqueeze(-1).view(-1, self.C, 24, 24)  # teacher 2
#
#         # student
#         backbone_student = torch.nn.functional.interpolate(student['opt_feat'], scale_factor=2, mode='bilinear')
#
#         kd_loss1 = self.kdloss(backbone_student, backbone_teacher)
#         kd_loss2 = self.kdloss(backbone_student, p_teacher)
#
#         # kd_loss = self.kdloss_mse(backbone_teacher, backbone_student)
#
#         kd_loss = kd_loss1 + kd_loss2
#
#         return kd_loss


# class FeatureLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.kdloss = DistillKL(1)
#         self.kdloss = DistillKL(0.001)
#         self.C = 768
#         # self.kdloss_mse = torch.nn.MSELoss()
#
#     def forward(self, student, teacher):
#         backbone_teacher = teacher['opt_feat']  # teacher 1
#         backbone_teacher_1 = backbone_teacher.flatten(2, 3)
#
#         # here get the teacher topk number
#         mean_value = torch.mean(backbone_teacher_1, 1)
#         number, index_number = mean_value.topk(100, 1)
#         number_no, index_number_no = mean_value.topk(576-100, 1, largest=False)
#         output = []
#         output_no = []
#         for i in range(32):
#             result = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
#             result_no = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number_no[i]))[i, :, :]
#             output.append(result)
#             output_no.append(result_no)
#
#         stacked_tensor_saliency = torch.stack(output)
#         stacked_tensor_no_saliency = torch.stack(output_no)
#
#
#         # student
#         backbone_student = torch.nn.functional.interpolate(student['opt_feat'], scale_factor=2, mode='bilinear')
#         backbone_student_1 = backbone_student.flatten(2, 3)
#
#         mean_value_student = torch.mean(backbone_student_1, 1)
#         number_student, index_number_student = mean_value_student.topk(100, 1)
#         number_no_student, index_number_no_student = mean_value_student.topk(576 - 100, 1, largest=False)
#         output_student = []
#         output_no_student = []
#         for i in range(32):
#             result_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number_student[i]))[i, :, :]
#             result_no_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number_no_student[i]))[i, :, :]
#             output_student.append(result_student)
#             output_no_student.append(result_no_student)
#
#         stacked_tensor_saliency_student = torch.stack(output_student)
#         stacked_tensor_no_saliency_student = torch.stack(output_no_student)
#
#
#         kd_loss1 = self.kdloss(stacked_tensor_saliency_student, stacked_tensor_saliency)
#         kd_loss2 = self.kdloss(stacked_tensor_no_saliency_student, stacked_tensor_no_saliency)
#
#         kd_loss = 10 * kd_loss1 + kd_loss2
#         # kd_loss = self.kdloss_mse(backbone_teacher, backbone_student)
#
#         return kd_loss


# class FeatureLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.kdloss = DistillKL(1)
#         self.kdloss = DistillKL(0.001)
#         self.C = 768
#         # self.kdloss_mse = torch.nn.MSELoss()
# 
#     def forward(self, student, teacher):
#         backbone_teacher = teacher['opt_feat']  # teacher 1
#         backbone_teacher_1 = backbone_teacher.flatten(2, 3)
#         backbone_student = torch.nn.functional.interpolate(student['opt_feat'], scale_factor=2, mode='bilinear')
#         backbone_student_1 = backbone_student.flatten(2, 3)
# 
#         # here get the teacher topk number
#         mean_value = torch.mean(backbone_teacher_1, 1)
#         number, index_number = mean_value.topk(100, 1)
#         number_no, index_number_no = mean_value.topk(576-100, 1, largest=False)
#         output = []
#         output_no = []
#         output_student = []
#         output_no_student = []
#         for i in range(32):
#             result = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
#             result_no = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number_no[i]))[i, :, :]
# 
#             result_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
#             result_no_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number_no[i]))[i,:, :]
# 
#             output.append(result)
#             output_no.append(result_no)
# 
#             output_student.append(result_student)
#             output_no_student.append(result_no_student)
# 
#         stacked_tensor_saliency = torch.stack(output)
#         stacked_tensor_no_saliency = torch.stack(output_no)
#         stacked_tensor_saliency_student = torch.stack(output_student)
#         stacked_tensor_no_saliency_student = torch.stack(output_no_student)
# 
# 
#         kd_loss1 = self.kdloss(stacked_tensor_saliency_student, stacked_tensor_saliency)
#         kd_loss2 = self.kdloss(stacked_tensor_no_saliency_student, stacked_tensor_no_saliency)
# 
#         kd_loss = 10 * kd_loss1 + kd_loss2
#         # kd_loss = self.kdloss_mse(backbone_teacher, backbone_student)
# 
#         return kd_loss


# fan's saliency
# class FeatureLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.kdloss = DistillKL(1)
#         self.kdloss = DistillKL(0.001)
#         self.C = 768
#         # self.kdloss_mse = torch.nn.MSELoss()
#
#     def forward(self, student, teacher):
#         # student
#         p_student = student["attn"][11][:, 36:].unsqueeze(-1).view(-1, self.C, 12, 12)
#         backbone_student_1 = torch.nn.functional.interpolate(p_student, scale_factor=2,
#                                                     mode='bilinear')
#         backbone_student_1 = backbone_student_1.flatten(2, 3)
#
#         # teacher
#         backbone_teacher_1 = teacher["attn"][11][:, 144:].unsqueeze(-1).view(-1, self.C, 24, 24)
#         backbone_teacher_1 = backbone_teacher_1.flatten(2, 3)
#
#
#         # a little bit different
#         # backbone_teacher = teacher['opt_feat']  # teacher 1
#         # backbone_teacher_1 = backbone_teacher.flatten(2, 3)
#         #
#         # backbone_student = torch.nn.functional.interpolate(student['opt_feat'], scale_factor=2, mode='bilinear')
#         # backbone_student_1 = backbone_student.flatten(2, 3)
#
#         attn_out = teacher['aux_attn_out'][11]
#
#         # attn_out_mean = torch.sum(attn_out, dim=3)
#         # attn_out_mean = torch.sum(attn_out_mean, dim=1)
#
#         # here get the sliency and non-saliency index
#         number, index_number = attn_out.topk(100, 1)
#         number_no, index_number_no = attn_out.topk(576-100, 1, largest=False)
#
#         output = []
#         output_no = []
#         output_student = []
#         output_no_student = []
#
#         for i in range(32):
#             result = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
#             result_no = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number_no[i]))[i, :, :]
#
#             result_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
#             result_no_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number_no[i]))[i,:, :]
#
#             output.append(result)
#             output_no.append(result_no)
#
#             output_student.append(result_student)
#             output_no_student.append(result_no_student)
#
#         stacked_tensor_saliency = torch.stack(output)
#         stacked_tensor_no_saliency = torch.stack(output_no)
#         stacked_tensor_saliency_student = torch.stack(output_student)
#         stacked_tensor_no_saliency_student = torch.stack(output_no_student)
#
#
#         kd_loss1 = self.kdloss(stacked_tensor_saliency_student, stacked_tensor_saliency)
#         kd_loss2 = self.kdloss(stacked_tensor_no_saliency_student, stacked_tensor_no_saliency)
#
#         kd_loss = 10 * kd_loss1 + kd_loss2
#         # kd_loss = self.kdloss_mse(backbone_teacher, backbone_student)
#
#         return kd_loss


# class FeatureLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.kdloss = DistillKL(1)
#         self.kdloss = DistillKL(0.001)
#         self.C = 768
#         # self.kdloss_mse = torch.nn.MSELoss()
#         self.alpha = nn.Parameter(torch.zeros(1)).cuda()
#         self.gamma = nn.Parameter(torch.zeros(1)).cuda()
#
#     def forward(self, student, teacher):
#
#         q_student = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1, self.C, 12,
#                                                                                                          12)
#         q_student = torch.nn.functional.interpolate(q_student, scale_factor=2, mode='bilinear')
#
#         k_student = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1, self.C, 12,
#                                                                                                          12)
#         k_student = torch.nn.functional.interpolate(k_student, scale_factor=2, mode='bilinear')
#
#         v_student = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1, self.C, 12,
#                                                                                                          12)
#         v_student = torch.nn.functional.interpolate(v_student, scale_factor=2, mode='bilinear')
#
#         q_teacher = teacher["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
#                                                                                                           24, 24)
#         k_teacher = teacher["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
#                                                                                                           24, 24)
#         v_teacher = teacher["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
#                                                                                                           24, 24)
#
#         # kd_loss = self.kdloss(q_student, q_teacher) + self.kdloss(k_student, k_teacher) + self.kdloss(v_student, v_teacher)
#         kd_loss = self.kdloss(q_student, q_teacher) + self.alpha * self.kdloss(k_student, k_teacher) + self.gamma * self.kdloss(v_student, v_teacher)
#
#
#         return kd_loss


# class qkv11_target(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.kdloss_teacher = DistillKL(4)
#         self.kdloss = DistillKL(4)
#         # self.kdloss = DistillKL(0.001)  # better than before, but what happened if t=4
#         self.C = 768
#         self.gamma = nn.Parameter(torch.zeros(1)).cuda()
#         self.kdloss_mse = torch.nn.MSELoss()
#
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(768, 64, kernel_size=1, padding=0),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #     nn.Conv2d(64, 768, kernel_size=1, padding=0),
#         #     nn.BatchNorm2d(768),
#         #     nn.ReLU()
#         # ).cuda()
#         #
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(768, 64, kernel_size=1, padding=0),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #     nn.Conv2d(64, 768, kernel_size=1, padding=0),
#         #     nn.BatchNorm2d(768),
#         #     nn.ReLU()
#         # ).cuda()
#         #
#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(768, 64, kernel_size=1, padding=0),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #     nn.Conv2d(64, 768, kernel_size=1, padding=0),
#         #     nn.BatchNorm2d(768),
#         #     nn.ReLU()
#         # ).cuda()
#
#     def forward(self, student, teacher):
#         loss = 0
#         for i in range(12):
#             q_student = student["attn_q"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1,
#                                                                                                               self.C,
#                                                                                                               12,
#                                                                                                               12)
#             q_student = torch.nn.functional.interpolate(q_student, scale_factor=2, mode='bilinear')
#             q_student_conv = self.conv1(q_student)
#
#             k_student = student["attn_k"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1,
#                                                                                                               self.C,
#                                                                                                               12,
#                                                                                                               12)
#             k_student = torch.nn.functional.interpolate(k_student, scale_factor=2, mode='bilinear')
#             k_student_conv = self.conv2(k_student)
#
#             v_student = student["attn_v"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 36:].unsqueeze(-1).view(-1,
#                                                                                                               self.C,
#                                                                                                               12,
#                                                                                                               12)
#             v_student = torch.nn.functional.interpolate(v_student, scale_factor=2, mode='bilinear')
#             v_student_conv = self.conv3(v_student)
#
#             q_teacher = teacher["attn_q"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1,
#                                                                                                                self.C,
#                                                                                                                24, 24)
#             k_teacher = teacher["attn_k"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1,
#                                                                                                                self.C,
#                                                                                                                24, 24)
#             v_teacher = teacher["attn_v"][i].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1,
#                                                                                                                self.C,
#                                                                                                                24, 24)
#
#             # kd search loss
#             kd_loss_search = self.kdloss(q_student, q_teacher) + self.kdloss(k_student, k_teacher) + self.kdloss(v_student,v_teacher)
#             kd_loss_search_conv = self.kdloss(q_student_conv, q_teacher) + self.kdloss(k_student_conv, k_teacher) + self.kdloss(v_student_conv,v_teacher)
#
#
#
#
#             # loss = loss + kd_loss_search + kd_loss_search_conv
#             loss = loss + kd_loss_search
#
#         return loss

class qkv11_target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kdloss_teacher = DistillKL(4)
        self.kdloss = DistillKL(4)
        # self.kdloss = DistillKL(0.001)  # better than before, but what happened if t=4
        self.C = 768
        # self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.kdloss_mse = torch.nn.MSELoss()

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(768, 64, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 768, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(768),
        #     nn.ReLU()
        # ).cuda()
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(768, 64, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 768, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(768),
        #     nn.ReLU()
        # ).cuda()
        #
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(768, 64, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 768, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(768),
        #     nn.ReLU()
        # ).cuda()

    def forward(self, student, teacher):

        q_student = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16,16)

        q_student = torch.nn.functional.interpolate(q_student, size=(24, 24), mode='bilinear')

        k_student = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16, 16)
        k_student = torch.nn.functional.interpolate(k_student, size=(24, 24), mode='bilinear')

        v_student = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16, 16)
        v_student = torch.nn.functional.interpolate(v_student, size=(24, 24), mode='bilinear')

        q_teacher = teacher["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)
        k_teacher = teacher["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)
        v_teacher = teacher["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)


        # target loss
        # q_student_target = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :36].unsqueeze(-1).view(-1, self.C,
        #                                                                                                   6,
        #                                                                                                   6)
        # q_student_target = torch.nn.functional.interpolate(q_student_target, scale_factor=2, mode='bilinear')
        #
        # k_student_target = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :36].unsqueeze(-1).view(-1, self.C,
        #                                                                                                   6,
        #                                                                                                   6)
        # k_student_target = torch.nn.functional.interpolate(k_student_target, scale_factor=2, mode='bilinear')
        #
        # v_student_target = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :36].unsqueeze(-1).view(-1, self.C,
        #                                                                                                   6,
        #                                                                                                   6)
        # v_student_target = torch.nn.functional.interpolate(v_student_target, scale_factor=2, mode='bilinear')
        #
        # q_teacher_target = teacher["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)
        # k_teacher_target = teacher["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)
        # v_teacher_target = teacher["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)

        # kd search loss
        kd_loss_search = self.kdloss(q_student, q_teacher) + self.kdloss(k_student, k_teacher) + self.kdloss(v_student,
                                                                                                          v_teacher)

        # kd_loss_search_target = self.kdloss(q_student_target, q_teacher_target) + self.kdloss(k_student_target, k_teacher_target) + self.kdloss(v_student_target,
        #                                                                                                      v_teacher_target)

        kd_loss = kd_loss_search
        # kd_loss = kd_loss_search + self.gamma * kd_loss_search_target

        return kd_loss

class qkv11_target_target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kdloss_teacher = DistillKL(4)
        self.kdloss = DistillKL(4)
        # self.kdloss = DistillKL(0.001)  # better than before, but what happened if t=4
        self.C = 768
        # self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.kdloss_mse = torch.nn.MSELoss()


    def forward(self, student, teacher):

        # 256
        # q_student = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16,16)
        #
        # q_student = torch.nn.functional.interpolate(q_student, size=(24, 24), mode='bilinear')
        #
        # k_student = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16, 16)
        # k_student = torch.nn.functional.interpolate(k_student, size=(24, 24), mode='bilinear')
        #
        # v_student = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16, 16)
        # v_student = torch.nn.functional.interpolate(v_student, size=(24, 24), mode='bilinear')

        # 96
        q_student = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 9:].unsqueeze(-1).view(-1, self.C,
                                                                                                          6, 6)

        q_student = torch.nn.functional.interpolate(q_student, size=(24, 24), mode='bilinear')

        k_student = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 9:].unsqueeze(-1).view(-1, self.C,
                                                                                                          6, 6)
        k_student = torch.nn.functional.interpolate(k_student, size=(24, 24), mode='bilinear')

        v_student = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 9:].unsqueeze(-1).view(-1, self.C,
                                                                                                          6, 6)
        v_student = torch.nn.functional.interpolate(v_student, size=(24, 24), mode='bilinear')


        # 160
        # q_student = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 25:].unsqueeze(-1).view(-1, self.C,
        #                                                                                                   10, 10)
        # q_student = torch.nn.functional.interpolate(q_student, size=(24, 24), mode='bilinear')
        #
        # k_student = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 25:].unsqueeze(-1).view(-1, self.C,
        #                                                                                                   10, 10)
        # k_student = torch.nn.functional.interpolate(k_student, size=(24, 24), mode='bilinear')
        #
        # v_student = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 25:].unsqueeze(-1).view(-1, self.C,
        #                                                                                                   10, 10)
        # v_student = torch.nn.functional.interpolate(v_student, size=(24, 24), mode='bilinear')



        q_teacher = teacher["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)
        k_teacher = teacher["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                           24, 24)
        v_teacher = teacher["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)


        # target loss
        # q_student_target = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C,8,
        #                                                                                                   8)
        #
        # q_student_target = torch.nn.functional.interpolate(q_student_target, size=(12, 12), mode='bilinear')
        #
        # k_student_target = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C,8,
        #                                                                                                   8)
        #
        # k_student_target = torch.nn.functional.interpolate(k_student_target, size=(12, 12), mode='bilinear')
        #
        # v_student_target = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C, 8,
        #                                                                                                   8)
        #
        # v_student_target = torch.nn.functional.interpolate(v_student_target, size=(12, 12), mode='bilinear')
        #
        # q_teacher_target = teacher["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)
        # k_teacher_target = teacher["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)
        # v_teacher_target = teacher["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)

        # kd search loss
        kd_loss_search = self.kdloss(q_student, q_teacher) + self.kdloss(k_student, k_teacher) + self.kdloss(v_student,
                                                                                                          v_teacher)

        # kd_loss_search_target = self.kdloss(q_student_target, q_teacher_target) + self.kdloss(k_student_target, k_teacher_target) + self.kdloss(v_student_target,
        #                                                                                                      v_teacher_target)

        kd_loss = kd_loss_search
        # kd_loss = kd_loss_search + self.gamma * kd_loss_search_target

        return kd_loss



class qkv6_target_target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kdloss_teacher = DistillKL(4)
        self.kdloss = DistillKL(4)
        # self.kdloss = DistillKL(0.001)  # better than before, but what happened if t=4
        self.C = 768
        # self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.kdloss_mse = torch.nn.MSELoss()


    def forward(self, student, teacher):

        q_student = student["attn_q"][6].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16,16)

        q_student = torch.nn.functional.interpolate(q_student, size=(24, 24), mode='bilinear')

        k_student = student["attn_k"][6].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16, 16)
        k_student = torch.nn.functional.interpolate(k_student, size=(24, 24), mode='bilinear')

        v_student = student["attn_v"][6].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16, 16)
        v_student = torch.nn.functional.interpolate(v_student, size=(24, 24), mode='bilinear')

        q_teacher = teacher["attn_q"][6].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)
        k_teacher = teacher["attn_k"][6].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                           24, 24)
        v_teacher = teacher["attn_v"][6].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)


        # target loss
        # q_student_target = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C,8,
        #                                                                                                   8)
        #
        # q_student_target = torch.nn.functional.interpolate(q_student_target, size=(12, 12), mode='bilinear')
        #
        # k_student_target = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C,8,
        #                                                                                                   8)
        #
        # k_student_target = torch.nn.functional.interpolate(k_student_target, size=(12, 12), mode='bilinear')
        #
        # v_student_target = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C, 8,
        #                                                                                                   8)
        #
        # v_student_target = torch.nn.functional.interpolate(v_student_target, size=(12, 12), mode='bilinear')
        #
        # q_teacher_target = teacher["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)
        # k_teacher_target = teacher["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)
        # v_teacher_target = teacher["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)

        # kd search loss
        kd_loss_search = self.kdloss(q_student, q_teacher) + self.kdloss(k_student, k_teacher) + self.kdloss(v_student,
                                                                                                          v_teacher)

        # kd_loss_search_target = self.kdloss(q_student_target, q_teacher_target) + self.kdloss(k_student_target, k_teacher_target) + self.kdloss(v_student_target,
        #                                                                                                      v_teacher_target)

        kd_loss = kd_loss_search
        # kd_loss = kd_loss_search + self.gamma * kd_loss_search_target

        return kd_loss
class qkv10_target_target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kdloss_teacher = DistillKL(4)
        self.kdloss = DistillKL(4)
        # self.kdloss = DistillKL(0.001)  # better than before, but what happened if t=4
        self.C = 768
        # self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.gamma = nn.Parameter(torch.zeros(1)).cuda()
        self.kdloss_mse = torch.nn.MSELoss()


    def forward(self, student, teacher):

        q_student = student["attn_q"][10].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16,16)

        q_student = torch.nn.functional.interpolate(q_student, size=(24, 24), mode='bilinear')

        k_student = student["attn_k"][10].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16, 16)
        k_student = torch.nn.functional.interpolate(k_student, size=(24, 24), mode='bilinear')

        v_student = student["attn_v"][10].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 64:].unsqueeze(-1).view(-1, self.C, 16, 16)
        v_student = torch.nn.functional.interpolate(v_student, size=(24, 24), mode='bilinear')

        q_teacher = teacher["attn_q"][10].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)
        k_teacher = teacher["attn_k"][10].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                           24, 24)
        v_teacher = teacher["attn_v"][10].permute(0, 1, 3, 2).flatten(1, 2)[:, :, 144:].unsqueeze(-1).view(-1, self.C,
                                                                                                          24, 24)


        # target loss
        # q_student_target = student["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C,8,
        #                                                                                                   8)
        #
        # q_student_target = torch.nn.functional.interpolate(q_student_target, size=(12, 12), mode='bilinear')
        #
        # k_student_target = student["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C,8,
        #                                                                                                   8)
        #
        # k_student_target = torch.nn.functional.interpolate(k_student_target, size=(12, 12), mode='bilinear')
        #
        # v_student_target = student["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :64].unsqueeze(-1).view(-1, self.C, 8,
        #                                                                                                   8)
        #
        # v_student_target = torch.nn.functional.interpolate(v_student_target, size=(12, 12), mode='bilinear')
        #
        # q_teacher_target = teacher["attn_q"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)
        # k_teacher_target = teacher["attn_k"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)
        # v_teacher_target = teacher["attn_v"][11].permute(0, 1, 3, 2).flatten(1, 2)[:, :, :144].unsqueeze(-1).view(-1, self.C,
        #                                                                                                    12, 12)

        # kd search loss
        kd_loss_search = self.kdloss(q_student, q_teacher) + self.kdloss(k_student, k_teacher) + self.kdloss(v_student,
                                                                                                          v_teacher)

        # kd_loss_search_target = self.kdloss(q_student_target, q_teacher_target) + self.kdloss(k_student_target, k_teacher_target) + self.kdloss(v_student_target,
        #                                                                                                      v_teacher_target)

        kd_loss = kd_loss_search
        # kd_loss = kd_loss_search + self.gamma * kd_loss_search_target

        return kd_loss


# class FeatureLoss_target(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.kdloss = DistillKL(1)
#         # self.kdloss = DistillKL(0.001)
#         self.kdloss = DistillKL(4)
#         self.C = 768
#         self.alpha = nn.Parameter(torch.zeros(1)).cuda()
#         # self.kdloss_mse = torch.nn.MSELoss()
#
#     def forward(self, student, teacher):
#         backbone_teacher = teacher['opt_feat']  # teacher 1
#         backbone_teacher_1 = backbone_teacher.flatten(2, 3)
#         backbone_student = torch.nn.functional.interpolate(student['opt_feat'], scale_factor=2, mode='bilinear')
#         backbone_student_1 = backbone_student.flatten(2, 3)
#
#         # here get the teacher topk number
#         # mean_value = torch.mean(backbone_teacher_1, 1)
#
#         # max saliency
#         mean_value = backbone_teacher_1.max(1)[0]
#         # number, index_number = mean_value.topk(100, 1)
#         # number_no, index_number_no = mean_value.topk(576-100, 1, largest=False)
#         number, index_number = mean_value.topk(10, 1)
#         number_no, index_number_no = mean_value.topk(576 - 10, 1, largest=False)
#         output = []
#         output_no = []
#         output_student = []
#         output_no_student = []
#         for i in range(32):
#             result = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
#             result_no = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number_no[i]))[i, :, :]
#
#             result_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
#             result_no_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number_no[i]))[i,:, :]
#
#             output.append(result)
#             output_no.append(result_no)
#
#             output_student.append(result_student)
#             output_no_student.append(result_no_student)
#
#         stacked_tensor_saliency = torch.stack(output)
#         stacked_tensor_no_saliency = torch.stack(output_no)
#         stacked_tensor_saliency_student = torch.stack(output_student)
#         stacked_tensor_no_saliency_student = torch.stack(output_no_student)
#
#
#         kd_loss1 = self.kdloss(stacked_tensor_saliency_student, stacked_tensor_saliency)
#         kd_loss2 = self.kdloss(stacked_tensor_no_saliency_student, stacked_tensor_no_saliency)
#
#         # kd_loss = 2 * kd_loss1 + self.alpha * kd_loss2
#         kd_loss = kd_loss1 + self.alpha * kd_loss2
#         # kd_loss = self.kdloss_mse(backbone_teacher, backbone_student)
#
#         return kd_loss

class FeatureLoss_target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.kdloss = DistillKL(1)
        # self.kdloss = DistillKL(0.001)
        self.kdloss = DistillKL(4)
        self.C = 768
        self.alpha = nn.Parameter(torch.zeros(1)).cuda()
        # self.kdloss_mse = torch.nn.MSELoss()

    def forward(self, student, teacher):
        backbone_teacher = teacher['opt_feat']  # teacher 1
        backbone_teacher_1 = backbone_teacher.flatten(2, 3)
        # backbone_student = torch.nn.functional.interpolate(student['opt_feat'], scale_factor=2, mode='bilinear')
        backbone_student = torch.nn.functional.interpolate(student['opt_feat'], size=(24, 24), mode='bilinear')
        backbone_student_1 = backbone_student.flatten(2, 3)

        # max saliency
        # value = backbone_teacher_1.pow(2)
        # # mean_value = value.mean(1)[0]
        # mean_value = value.mean(1)
        # number, index_number = mean_value.topk(100, 1)
        # number_no, index_number_no = mean_value.topk(576 - 100, 1, largest=False)
        # output = []
        # output_no = []
        # output_student = []
        # output_no_student = []
        # for i in range(32):
        #     result = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
        #     result_no = torch.index_select(backbone_teacher_1, dim=2, index=torch.tensor(index_number_no[i]))[i, :, :]
        #
        #     result_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number[i]))[i, :, :]
        #     result_no_student = torch.index_select(backbone_student_1, dim=2, index=torch.tensor(index_number_no[i]))[i,:, :]
        #
        #     output.append(result)
        #     output_no.append(result_no)
        #
        #     output_student.append(result_student)
        #     output_no_student.append(result_no_student)
        #
        # stacked_tensor_saliency = torch.stack(output)
        # stacked_tensor_no_saliency = torch.stack(output_no)
        # stacked_tensor_saliency_student = torch.stack(output_student)
        # stacked_tensor_no_saliency_student = torch.stack(output_no_student)

        value = backbone_teacher_1.pow(2)
        # mean_value = value.mean(1)[0]
        mean_value = value.mean(1)
        mask = mean_value.ge(0.20)  # 32, 576
        mask = mask.unsqueeze(1)
        no_mask = ~mask

        stacked_tensor_saliency_student = backbone_student_1 * mask
        stacked_tensor_no_saliency_student = backbone_student_1 * no_mask

        stacked_tensor_saliency = backbone_teacher_1 * mask
        stacked_tensor_no_saliency = backbone_teacher_1 * no_mask


        kd_loss1 = self.kdloss(stacked_tensor_saliency_student, stacked_tensor_saliency)
        kd_loss2 = self.kdloss(stacked_tensor_no_saliency_student, stacked_tensor_no_saliency)

        # kd_loss = 2 * kd_loss1 + self.alpha * kd_loss2
        # kd_loss = kd_loss1 + self.alpha * kd_loss2

        # this is lasot
        kd_loss = 0.6 * kd_loss1 + 0.4 * kd_loss2

        # this is got-10k
        # kd_loss = 0.75 * kd_loss1 + 0.25 * kd_loss2

        # kd_loss = 0.7 * kd_loss1 + 0.3 * kd_loss2
        # kd_loss = 0.68 * kd_loss1 + 0.32 * kd_loss2
        # kd_loss = self.kdloss_mse(backbone_teacher, backbone_student)

        return kd_loss
class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    # def __init__(self, net_backbone, net_head, net_teacher_backbone, net_teacher_head, objective, loss_weight, settings, cfg=None):
    def __init__(self, net, net_teacher, objective, loss_weight, settings, cfg=None):
        # super().__init__(net_backbone, net_head, net_teacher_backbone, net_teacher_head, objective)
        super().__init__(net, net_teacher, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        # 192
        # self.resize_template = transforms.Resize([96, 96])
        # self.resize_search = transforms.Resize([192, 192])

        # 256
        # self.resize_template = transforms.Resize([128, 128])
        # self.resize_search = transforms.Resize([256, 256])

        # 96
        self.resize_template = transforms.Resize([48, 48])
        self.resize_search = transforms.Resize([96, 96])


        # 160
        # self.resize_template = transforms.Resize([80, 80])
        # self.resize_search = transforms.Resize([160, 160])

        # self.kdloss = FeatureLoss()

        self.qkv_11 = qkv11_target_target()

        # self.qkv_6 = qkv6_target_target()

        # self.qkv_10 = qkv10_target_target()
        # self.attloss = FeatureLoss_target()
        self.kd_loss = FeatureLoss_target()

        # self.attloss = AttnDiV_qkv()


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


        data['template_images'] = self.resize_template(data['template_images'].squeeze(0)).unsqueeze(0)
        data['search_images'] = self.resize_search(data['search_images'].squeeze(0)).unsqueeze(0)

        out_dict = self.forward_pass(data)
        # out_dict = self.forward_pass(data, out_dict_high['backbone_feat'])

        loss, status = self.compute_losses(out_dict, out_dict_high, data)
        # loss, status = self.compute_losses(out_dict, data)

        return loss, status

    # def forward_pass(self, data, final_feature):
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

        # out_dict = self.net(template=template_list,
        #                             search=search_img,
        #                             ce_template_mask=box_mask_z,
        #                             ce_keep_rate=ce_keep_rate,
        #                             return_last_attn=False,
        #                             final_feature=final_feature)
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

        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)

        # middle_gt_bbox = gt_bbox[-1]
        # gt_bbox = middle_gt_bbox * 2.0
        # gt_bbox = gt_bbox.unsqueeze(0)

        # # change the size (384->192)
        # middle_gt = gt_dict['search_anno'][-1][-1]
        # gt_dict['search_anno'][-1][-1] = middle_gt / 2.0

        # gt_gaussian_maps = generate_heatmap(bboxes=gt_dict['search_anno'], patch_size=192, stride=self.cfg.MODEL.BACKBONE.STRIDE)
        # gt_gaussian_maps = generate_heatmap(bboxes=gt_dict['search_anno'], patch_size=256, stride=self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = generate_heatmap(bboxes=gt_dict['search_anno'], patch_size=96, stride=self.cfg.MODEL.BACKBONE.STRIDE)
        # gt_gaussian_maps = generate_heatmap(bboxes=gt_dict['search_anno'], patch_size=160, stride=self.cfg.MODEL.BACKBONE.STRIDE)
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


        kd_loss = self.kd_loss(pred_dict, out_dict_high)
        loss_qkv_11 = self.qkv_11(pred_dict, out_dict_high)




        # loss_qkv_6 = self.qkv_6(pred_dict, out_dict_high)
        # loss_qkv_10 = self.qkv_10(pred_dict, out_dict_high)



        # weighted sum
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + 0.01 * kd_loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + 0.1 * kd_loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + 0.1 * kd_loss + 0.01 * loss_att1

        #******************************************************************#
        # output_0.01p4m4l4kd
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + 0.01 * loss_att1

        # div2 nolocation
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + 0.01 * loss_att1

        # divlocation loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        # 67.80
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * loss_att1

        # ostrack_2
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * loss_att1 + 0.01 * kd_loss

        # ostrack_3, kd loss + att[12] loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss

        # saliency feature  10 * kd_loss1 + kd_loss2 67.04
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss

        # new implement saliency
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss

        # fan's saliency + last feature q,k,v distillation: 67.41
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss

        # qkv distilaltion  + conv: 66.27
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss

        # qkv distilaltion, 256: 69.92
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss

        # qkv11 distillation + template + target(alpha) mean saliency = 69.82
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_att1

        # qkv11 distillation + template + target(alpha) max saliency + number pixel=100, = 70.17
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_att1

        # qkv11 distillation + template + target(alpha) max saliency + number pixel = 80 = 69.88
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_att1

        # qkv11 distillation + target(alpha) max saliency + pixel 200 = 69.66
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_att1

        # qkv11 distillation + target(alpha) max saliency + pixel 50, 69.81
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_att1

        # qkv11 distillation + target(alpha) max saliency + pixel 25, but = 69.91
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_att1

        # qkv11 distillation + target(alpha) max saliency + pixel 10, but = 69.75
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_att1

        # qkv11 distillation + target(alpha) max saliency + pixel 10,  1s + a * non-s but = 69.84


        # from here, this is the ablation study
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_att1

        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss


        # this is qkv11 + kd loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_qkv_11

        # here is our ablation studies. (without qkv-11)
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss

        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
        #     'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_qkv_11 + 0.1 * loss_qkv_6

        # this is got-10k 96, 100 epoch
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * location_loss + 0.01 * kd_loss + 0.01 * loss_qkv_11


        #******************************************************************#

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/kd": kd_loss.item(),
                      "Loss/qkv_11": loss_qkv_11.item(),
                      # "Loss/qkv_6": loss_qkv_6.item(),
                      # "Loss/qkv_10": loss_qkv_10.item(),
                      # "Loss/att1": loss_att1.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
