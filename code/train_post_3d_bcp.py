import shutil

import argparse
import logging
import os
import os.path as osp
import random
import sys
import yaml

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.mixaugs import cut_mix_3d_t,cut_mix_bcp
from dataloaders.dataset_3d import (LAHeart, Pancreas, WeakStrongAugment, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, ramps, metrics
from utils.util import update_values, time_str, AverageMeter
from train_utils import AlternateUpdate, get_compromise_pseudo_btw_tea_stu
from val_3D import var_all_case_LA, var_all_case_Pancrease
from torch.nn.parameter import Parameter

my_DICE = losses.my_DiceLoss(nclass=2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        I. helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


def update_ema_variables(model, ema_model, alpha, global_step, args):
    # adjust the momentum param
    if global_step < args["consistency_rampup"]:
        alpha = 0.0
    else:
        alpha = min(1 - 1 / (global_step - args["consistency_rampup"] + 1), alpha)

    # update weights
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    # update buffers
    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.data = buffer_eval.data * alpha + buffer_train.data * (1 - alpha)


class Linear_vector(nn.Module):
    def __init__(self, n_dim):
        super(Linear_vector, self).__init__()
        self.n_dim = n_dim
        self.paras = Parameter(torch.Tensor(self.n_dim, self.n_dim))  # linear coefficients
        self.init_ratio = 1e-3
        self.initialize()

    def initialize(self):
        for param in self.paras:
            param.data.normal_(0, self.init_ratio)

    def forward(self, x):
        # result = paras*x   (16*9)=(16*16)*(16*9)
        result = torch.mm(self.paras, x)
        return result


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        II. trainer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def worker_init_fn(worker_id):
    random.seed(2023 + worker_id)


def train(args, snapshot_path):
    base_lr = args["base_lr"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    num_classes = args["num_classes"]
    cur_time = time_str()
    writer = SummaryWriter(snapshot_path + '/log')
    csv_train = os.path.join(snapshot_path, "log", "seg_{}_train_iter.csv".format(cur_time))
    csv_test = os.path.join(snapshot_path, "log", "seg_{}_validate_ep.csv".format(cur_time))
    dice_txt = os.path.join(snapshot_path, "log", "seg_{}_pseudo_dice.txt".format(cur_time))

    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args["model"], in_chns=1,
                          class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        if len(args['gpu_id']) > 1:
            model = nn.DataParallel(model, device_ids=list(range(len(args['gpu_id']))))
        return model

    # + + + + + + + + + + + #
    # 1. create model
    # + + + + + + + + + + + #
    model = create_model()
    ema_model = create_model(ema=True)
    teacher1 = create_model()
    teacher2 = create_model()
    model.train()
    teacher1.train()
    teacher2.train()
    ema_model.train()

    # + + + + + + + + + + + #
    # 2. dataset
    # + + + + + + + + + + + #
    fdloader = LAHeart
    flag_pancreas = True if "pancreas" in args["root_path"].lower() else False
    if flag_pancreas:
        fdloader = Pancreas
    db_train = fdloader(base_dir=args["root_path"],
                        data_dir=args['data_path'],
                        split='train',
                        num=None,
                        transform=transforms.Compose([
                            WeakStrongAugment(args["patch_size"],
                                              flag_rot=not flag_pancreas)
                        ]))

    labeled_idxs = list(range(0, args["labeled_num"]))
    unlabeled_idxs = list(range(args["labeled_num"], args["max_samples"]))

    if args.get("flag_sampling_based_on_lb", False):
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, batch_size, batch_size - args["labeled_bs"])
    else:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"])

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=args["workers"], pin_memory=True, worker_init_fn=worker_init_fn,
                             generator=torch.Generator().manual_seed(args["seed"]))

    logging.info("{} iterations per epoch".format(len(trainloader)))

    # + + + + + + + + + + + #
    # 4. optim, scheduler
    # + + + + + + + + + + + #
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer1 = optim.SGD(teacher1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(teacher2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # linear_paras1 = []
    # linear_paras2 = []
    # count = 0
    # for name, parameters in teacher1.named_parameters():
    #     if 'conv' in name and 'weight' in name:
    #         if len(parameters.shape) == 5:
    #             count += 1
    #             outdim = parameters.shape[0]  # output dimension
    #             linear_paras1.append(Linear_vector(outdim))
    #             linear_paras2.append(Linear_vector(outdim))
    #
    # linear_paras1 = nn.ModuleList(linear_paras1)
    # linear_paras2 = nn.ModuleList(linear_paras2)
    # linear_paras1 = linear_paras1.cuda()
    # linear_paras2 = linear_paras2.cuda()
    # linear_optimizer1 = torch.optim.Adam(linear_paras1.parameters(), 2e-2)
    # linear_optimizer2 = torch.optim.Adam(linear_paras2.parameters(), 2e-2)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # + + + + + + + + + + + #
    # 5. training loop
    # + + + + + + + + + + + #
    iter_num = 0
    iter_num_max = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_dice_t1 = 0.0
    best_dice_t2 = 0.0
    best_dice_stu = 0.0
    best_dice_ema = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    l1_iter_num = 0
    # alternate params
    alt_flag_epoch_shuffle_teachers = args["alt_flag_epoch_shuffle_teachers"]
    alt_flag_conflict_mode = args["alt_flag_conflict_mode"]
    alt_flag_conflict_stu_use_more = args["alt_flag_conflict_stu_use_more"]
    alt_param_ensemble_temp = args["alt_param_ensemble_temp"]

    alt_param_conflict_weight = args["alt_param_conflict_weight"]
    alt_flag_updating_period_random = args["alt_flag_updating_period_random"]
    alt_param_updating_period_iters = args["alt_param_updating_period_iters"]

    alternate_indicator = AlternateUpdate(alt_param_updating_period_iters,
                                          initial_flag=True,
                                          flag_random=alt_flag_updating_period_random)

    for epoch_num in iterator:
        # update alt_params
        # a) alternate flag
        if alt_flag_epoch_shuffle_teachers:
            alternate_indicator.reset(alt_param_updating_period_iters,
                                      initial_flag=(epoch_num % 2 == 0),
                                      flag_random=alt_flag_updating_period_random)
        # b) conflict weight
        var_param_conflict_weight = alt_param_conflict_weight
        # c) flag of starting self training
        flag_start_self_learning = False

        # metric indicators
        meter_sup_losses = AverageMeter()
        meter_uns_losses = AverageMeter(20)
        meter_uns_losses_t1 = AverageMeter(20)
        meter_uns_losses_t2 = AverageMeter(20)
        meter_train_losses = AverageMeter(20)
        meter_learning_rates = AverageMeter()
        meter_highc_ratio = AverageMeter()
        meter_highc_ratio_t = AverageMeter()
        meter_mask_ratio = AverageMeter()
        meter_mask_ratio_t = AverageMeter()
        meter_error_ratio = AverageMeter()
        meter_error_ratio_t = AverageMeter()
        meter_cosine_sim_t12 = AverageMeter()
        meter_cosine_sim_st1 = AverageMeter()
        meter_cosine_sim_st2 = AverageMeter()
        meter_cosine_sim_sema = AverageMeter()
        meter_cosine_sim_emat1 = AverageMeter()
        meter_cosine_sim_emat2 = AverageMeter()
        meter_conflict_ratio = AverageMeter()
        meter_uns_losses_consist = AverageMeter(20)
        meter_uns_losses_conflict = AverageMeter(20)
        dice_array_t = []
        dice_array_s = []
        sum_s = 0
        sum_t = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            # if iter_num > args['start_step1'] and iter_num % args['min_step'] == 0:
            #     for i_max in range(args['max_step']):
            #         # optimize linear coefficients
            #         icm_loss1 = -losses.l_correlation_cos_mean(teacher1, teacher2, linear_paras1)
            #         icm_loss2 = -losses.l_correlation_cos_mean(teacher2, teacher1, linear_paras2)
            #
            #         linear_optimizer1.zero_grad()
            #         linear_optimizer2.zero_grad()
            #
            #         icm_loss1.backward()
            #         icm_loss2.backward()
            #
            #         linear_optimizer1.step()
            #         linear_optimizer2.step()
            #
            #         iter_num_max += 1
            #
            #         writer.add_scalar('loss/icm_loss1_max', -icm_loss1, iter_num_max)
            #         writer.add_scalar('loss/icm_loss2_max', -icm_loss2, iter_num_max)

            num_lb = args["labeled_bs"]
            num_ulb = batch_size - num_lb

            # 1) get augmented data
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            idx_batch = sampled_batch['idx'].cuda()
            # get batched data
            if args.get("flag_sampling_based_on_lb", False):
                img_lb_w, target_lb = weak_batch[:num_lb], label_batch[:num_lb]
                img_ulb_w, img_ulb_s = weak_batch[num_lb:], strong_batch[num_lb:]
                target_ulb = label_batch[num_lb:]
                unimg_idx = idx_batch[num_lb:].cuda()
            else:
                img_lb_w, target_lb = weak_batch[num_ulb:], label_batch[num_ulb:]
                img_ulb_w, img_ulb_s = weak_batch[:num_ulb], strong_batch[:num_ulb]
                target_ulb = label_batch[:num_ulb]
                unimg_idx = idx_batch[:num_ulb].cuda()

            # 2) getting pseudo labels
            loss_ulb_consist, loss_ulb_conflict = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            with torch.no_grad():
                if not flag_start_self_learning and not args["flag_pseudo_from_student"]:
                    # if alternate_indicator.get_alternate_state():
                    if iter_num %2 == 0:
                        ema_model.train()
                        teacher1.train()
                        teacher2.eval()
                        tea = teacher2
                    else:
                        ema_model.train()
                        teacher2.train()
                        teacher1.eval()
                        tea = teacher1

                    ema_outputs_soft_1 = torch.softmax(ema_model(img_ulb_w)[0], dim=1)
                    ema_outputs_soft_1 = ema_outputs_soft_1.detach()

                    ema_outputs_soft_2 = torch.softmax(tea(img_ulb_w)[0], dim=1)
                    ema_outputs_soft_2 = ema_outputs_soft_2.detach()

                    _, pseudo_outputs_1 = torch.max(ema_outputs_soft_1, dim=1)
                    _, pseudo_outputs_2 = torch.max(ema_outputs_soft_2, dim=1)

                    mtx_bool_conflict = pseudo_outputs_1 != pseudo_outputs_2
                    conflict_ratio = mtx_bool_conflict.float().sum() / num_ulb

                    # entropy
                    entropy_1 = -torch.sum(ema_outputs_soft_1 * torch.log2(ema_outputs_soft_1 + 1e-10), dim=1)
                    entropy_2 = -torch.sum(ema_outputs_soft_2 * torch.log2(ema_outputs_soft_2 + 1e-10), dim=1)

                    # weighted sum
                    weights_1 = torch.exp(-entropy_1) / (torch.exp(-entropy_1) + torch.exp(-entropy_2))
                    weights_2 = 1 - weights_1
                    weighted_outputs = weights_1.unsqueeze(1) * ema_outputs_soft_1 + weights_2.unsqueeze(
                        1) * ema_outputs_soft_2

                    weighted_outputs = torch.pow(weighted_outputs, 1.0 / alt_param_ensemble_temp)
                    weighted_outputs = weighted_outputs / torch.sum(weighted_outputs, dim=1, keepdim=True)

                    # get final outputs
                    pseudo_logits, pseudo_outputs = torch.max(weighted_outputs, dim=1)
                    del ema_outputs_soft_1, pseudo_outputs_1, ema_outputs_soft_2, pseudo_outputs_2, weighted_outputs, entropy_1, entropy_2

                else:
                    model.eval()
                    ema_outputs_soft = torch.softmax(model(img_ulb_w)[0], dim=1)
                    pseudo_logits, pseudo_outputs = torch.max(ema_outputs_soft.detach(), dim=1)
                    mtx_bool_conflict = None
                    conflict_ratio = torch.tensor(0.0).cuda()
                    model.train()

            # 3) apply cutmix
            # if alternate_indicator.get_alternate_state() == False:
            if iter_num %2 == 1:
                del img_ulb_s
                if mtx_bool_conflict is None:
                    img_ulb_s, pseudo_outputs, pseudo_logits,target_ulb = cut_mix_bcp(
                        [img_ulb_w,
                        pseudo_outputs,
                        pseudo_logits,target_ulb],
                        [img_lb_w,
                        target_lb,
                        torch.ones_like(pseudo_logits),target_lb],
                        num_ulb//2
                    )
                else:
                    img_ulb_s, pseudo_outputs, pseudo_logits,target_ulb, mtx_bool_conflict = cut_mix_bcp(
                        [img_ulb_w,
                        pseudo_outputs,
                        pseudo_logits,target_ulb,mtx_bool_conflict],
                        [img_lb_w,
                        target_lb,
                        torch.ones_like(pseudo_logits),target_lb,torch.zeros_like(mtx_bool_conflict).bool()],
                        num_ulb//2
                    )
                    mtx_bool_conflict = mtx_bool_conflict.bool()
                    conflict_ratio = mtx_bool_conflict.float().sum() / num_ulb

            # 4) forward
            # if alternate_indicator.get_alternate_state():
            # print('before')
            # os.system('nvidia-smi')
            img = torch.cat((img_lb_w, img_ulb_w, img_ulb_s))
            pred, f_s = model(img)
            pred_lb = pred[:args["labeled_bs"]]
            pred_ulb_w, pred_ulb_s = pred[args["labeled_bs"]:].chunk(2)

            with torch.no_grad():
                pseudo_logits_stu, pseudo_outputs_stu = torch.max(torch.softmax(pred_ulb_w.detach(), dim=1), dim=1)
            del f_s, pred_ulb_w
            torch.cuda.empty_cache()
            # 5) supervised loss
            loss_lb, margin_loss = cal_sup_loss(ce_loss, dice_loss, pred_lb, target_lb, iter_num, l1_iter_num)

            # 6) unsupervised loss
            high_ratio, loss_ulb, loss_ulb_conflict, loss_ulb_consist, pseudo_label = cal_unsup_loss(
                alt_flag_conflict_mode,
                alt_flag_conflict_stu_use_more,
                args, dice_loss,
                flag_start_self_learning,
                loss_ulb_conflict,
                loss_ulb_consist,
                mtx_bool_conflict, pred_ulb_s,
                pseudo_logits, pseudo_logits_stu,
                pseudo_outputs,
                pseudo_outputs_stu,
                var_param_conflict_weight)

            # 7) total loss
            consistency_weight = get_current_consistency_weight(iter_num // 150, args)
            loss = loss_lb + consistency_weight * loss_ulb

            # 8) update student model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 9) update teacher model
            # if alternate_indicator.get_alternate_state():
            if iter_num %2 == 0:
                tea = teacher1
            else:
                tea = teacher2
            tea.train()
            pred_t, f_t = tea(img)
            pred_lb_t = pred_t[:args["labeled_bs"]]
            pred_ulb_w_t, pred_ulb_s_t = pred_t[args["labeled_bs"]:].chunk(2)
            with torch.no_grad():
                pseudo_logits_tea, pseudo_outputs_tea = torch.max(torch.softmax(pred_ulb_w_t.detach(), dim=1), dim=1)
            del pred_ulb_w_t,f_t
            torch.cuda.empty_cache()
            loss_lb_t, margin_loss_t = cal_sup_loss(ce_loss, dice_loss, pred_lb_t, target_lb, iter_num, l1_iter_num)
            high_ratio_t, loss_ce_dice_t, _, _, pseudo_label_t = cal_unsup_loss(alt_flag_conflict_mode,
                                                                                alt_flag_conflict_stu_use_more,
                                                                                args, dice_loss,
                                                                                flag_start_self_learning,
                                                                                loss_ulb_conflict,
                                                                                loss_ulb_consist,
                                                                                mtx_bool_conflict, pred_ulb_s_t,
                                                                                pseudo_logits, pseudo_logits_tea,
                                                                                pseudo_outputs,
                                                                                pseudo_outputs_tea,
                                                                                var_param_conflict_weight)
            loss_ulb_t = consistency_weight * loss_ce_dice_t
            if iter_num > args['start_step2'] and iter_num_max > 0:
                # icm_loss1 = losses.l_correlation_cos_mean(teacher1, teacher2, linear_paras1)
                # icm_loss2 = losses.l_correlation_cos_mean(teacher2, teacher1, linear_paras2)
                icm_loss1 = 0
                icm_loss2 = 0
            else:
                icm_loss1 = 0
                icm_loss2 = 0

            if not flag_start_self_learning:
                update_ema_variables(model, ema_model, args["ema_decay"], iter_num // 2, args)
                # if alternate_indicator.get_alternate_state():
                if iter_num %2 == 0:
                    # loss_ulb_t += args['coefficient'] * icm_loss1
                    loss1 = loss_lb_t + loss_ulb_t
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()
                    meter_uns_losses_t1.update(loss_ulb_t.item())
                    writer.add_scalar('loss/total_loss_tea1', loss1, iter_num)
                    writer.add_scalar('loss/loss_lb_tea1', loss_lb_t, iter_num)
                    writer.add_scalar('loss/loss_ulb_tea1', loss_ulb_t, iter_num)
                    writer.add_scalar('loss/loss_ce_dice_tea1', loss_ce_dice_t, iter_num)
                    writer.add_scalar('loss/margin_loss_tea1', margin_loss_t, iter_num)
                    writer.add_scalar('loss/caussl_loss_tea1', icm_loss1, iter_num)
                else:
                    # loss_ulb_t += args['coefficient'] * icm_loss2
                    loss2 = loss_lb_t + loss_ulb_t
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()
                    meter_uns_losses_t2.update(loss_ulb_t.item())
                    writer.add_scalar('loss/total_loss_tea2', loss2, iter_num)
                    writer.add_scalar('loss/loss_lb_tea2', loss_lb_t, iter_num)
                    writer.add_scalar('loss/loss_ulb_tea2', loss_ulb_t, iter_num)
                    writer.add_scalar('loss/loss_ce_dice_tea2', loss_ce_dice_t, iter_num)
                    writer.add_scalar('loss/margin_loss_tea2', margin_loss_t, iter_num)
                    writer.add_scalar('loss/caussl_loss_tea2', icm_loss2, iter_num)
                alternate_indicator.update()
            else:
                update_ema_variables(model, ema_model, args["ema_decay"], iter_num, args)
            torch.cuda.empty_cache()

            # 10) udpate learing rate
            if args["poly"]:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_ = base_lr
            # 11) record statistics
            mask_ratio, mask_ratio_t, error_ratio, error_ratio_t, sum_s, sum_t, dice_array_s, dice_array_t = cal_ratio(
                dice_array_s, dice_array_t, num_classes, pseudo_label, pseudo_label_t, sum_s,
                sum_t, target_ulb, unimg_idx)
            iter_num = iter_num + 1
            # --- a) writer
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('loss/total_loss', loss, iter_num)
            writer.add_scalar('loss/loss_lb', loss_lb, iter_num)
            writer.add_scalar('loss/loss_ulb', loss_ulb, iter_num)
            writer.add_scalar('loss/margin_loss', margin_loss, iter_num)

            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/conflict_ratio', conflict_ratio.item(), iter_num)
            writer.add_scalar('ratio/mask_ratio', mask_ratio, iter_num)
            writer.add_scalar('ratio/mask_ratio_t', mask_ratio_t, iter_num)
            writer.add_scalar('ratio/error_ratio', error_ratio, iter_num)
            writer.add_scalar('ratio/error_ratio_t', error_ratio_t, iter_num)
            # --- b) loggers
            logging.info(
                "iteration:{}  t-loss:{:.4f}, loss-lb:{:.4f}, loss-ulb:{:.4f}, conflict/consist:{:.4f}/{:.4f}, weight:{:.2f}, high-r:{:.2f}, high-r-t:{:.2f}, mask_ratio:{:.4f}, mask_ratio_t:{:.4f}, error_ratio:{:.4f}, error_ratio_t:{:.4f}, conflic:{}, lr:{:.4f}, alt:{}".format(
                    iter_num,
                    loss.item(), loss_lb.item(), loss_ulb.item(), loss_ulb_conflict.item(), loss_ulb_consist.item(),
                    consistency_weight, high_ratio, high_ratio_t, mask_ratio, mask_ratio_t, error_ratio, error_ratio_t,
                    int(conflict_ratio.item()), lr_, alternate_indicator.get_alternate_state()))
            # --- c) avg meters
            meter_sup_losses.update(loss_lb.item())
            meter_uns_losses.update(loss_ulb.item())

            meter_train_losses.update(loss.item())
            meter_highc_ratio.update(high_ratio.item())
            meter_highc_ratio_t.update(high_ratio_t.item())
            meter_mask_ratio.update(mask_ratio.item())
            meter_mask_ratio_t.update(mask_ratio_t.item())
            meter_error_ratio.update(error_ratio.item())
            meter_error_ratio_t.update(error_ratio_t.item())
            meter_learning_rates.update(lr_)
            meter_conflict_ratio.update(conflict_ratio.item())
            meter_uns_losses_consist.update(loss_ulb_consist.item())
            meter_uns_losses_conflict.update(loss_ulb_conflict.item())
            writer_cosine_sim(ema_model, iter_num, meter_cosine_sim_sema, meter_cosine_sim_st1, meter_cosine_sim_st2,
                              meter_cosine_sim_t12, meter_cosine_sim_emat1, meter_cosine_sim_emat2, model, teacher1,
                              teacher2, writer, img_ulb_w)
            # --- d) csv
            tmp_results = {
                'loss_total': loss.item(),
                'loss_lb': loss_lb.item(),
                'loss_ulb': loss_ulb.item(),
                'loss_ulb_conflict': loss_ulb_conflict.item(),
                'loss_ulb_consist': loss_ulb_consist.item(),
                'lweight_ub': consistency_weight,
                'high_ratio': high_ratio.item(),
                'high_ratio_t': high_ratio_t.item(),
                'mask_ratio': mask_ratio.item(),
                'mask_ratio_t': mask_ratio_t.item(),
                'error_ratio': error_ratio.item(),
                'error_ratio_t': error_ratio_t.item(),
                'conflict_ratio': conflict_ratio.item(),
                "lr": lr_}
            data_frame = pd.DataFrame(data=tmp_results, index=range(iter_num, iter_num + 1))
            if iter_num > 1 and osp.exists(csv_train):
                data_frame.to_csv(csv_train, mode='a', header=None, index_label='iter')
            else:
                data_frame.to_csv(csv_train, index_label='iter')

            if iter_num >= max_iterations:
                break

        dice_array_t.sort(key=lambda x: x[0], reverse=True)
        dice_array_s.sort(key=lambda x: x[0], reverse=True)
        avg_s = round(sum_s / len(dice_array_s), 4)
        avg_t = round(sum_t / len(dice_array_t), 4)
        writer.add_scalar('ratio/avg_dice_s', avg_s, epoch_num)
        writer.add_scalar('ratio/avg_dice_t', avg_t, epoch_num)
        t_log = "epoch:{},high_ratio_avg:{},avg_t:{},sort_dice_t:{}\n".format(epoch_num,
                                                                              round(meter_highc_ratio_t.avg, 4), avg_t,
                                                                              dice_array_t)
        s_log = "epoch:{},high_ratio_avg:{},avg_s:{},sort_dice_s:{}\n".format(epoch_num,
                                                                              round(meter_highc_ratio.avg, 4), avg_s,
                                                                              dice_array_s)
        logging.info(t_log)
        logging.info(s_log)
        logging.info(
            "epoch:{},cosine_sim_t12_avg:{},cosine_sim_st1_avg:{},cosine_sim_st2_avg:{},cosine_sim_seam_avg:{}\n".format(
                epoch_num, meter_cosine_sim_t12.avg, meter_cosine_sim_st1.avg, meter_cosine_sim_st2.avg,
                meter_cosine_sim_sema.avg))

        del dice_array_s, dice_array_t, sum_s, sum_t, avg_s, avg_t
        with open(dice_txt, "a") as f:
            f.writelines(t_log)
            f.writelines(s_log)
        writer.add_scalar('loss/total_loss_avg', meter_train_losses.avg, epoch_num)
        writer.add_scalar('loss/loss_lb_avg', meter_sup_losses.avg, epoch_num)
        writer.add_scalar('loss/loss_ulb_avg', meter_uns_losses.avg, epoch_num)
        writer.add_scalar('loss/loss_ulb_t1_avg', meter_uns_losses_t1.avg, epoch_num)
        writer.add_scalar('loss/loss_ulb_t2_avg', meter_uns_losses_t2.avg, epoch_num)

        writer.add_scalar('ratio/mask_ratio_avg', meter_mask_ratio.avg, epoch_num)
        writer.add_scalar('ratio/mask_ratio_t_avg', meter_mask_ratio_t.avg, epoch_num)
        writer.add_scalar('ratio/error_ratio_avg', meter_error_ratio.avg, epoch_num)
        writer.add_scalar('ratio/error_ratio_t_avg', meter_error_ratio_t.avg, epoch_num)
        writer.add_scalar('ratio/cosine_sim_t12_avg', meter_cosine_sim_t12.avg, epoch_num)
        writer.add_scalar('ratio/cosine_sim_st1_avg', meter_cosine_sim_st1.avg, epoch_num)
        writer.add_scalar('ratio/cosine_sim_st2_avg', meter_cosine_sim_st2.avg, epoch_num)
        writer.add_scalar('ratio/cosine_sim_sema_avg', meter_cosine_sim_sema.avg, epoch_num)
        writer.add_scalar('ratio/cosine_sim_emat1_avg', meter_cosine_sim_emat1.avg, epoch_num)
        writer.add_scalar('ratio/cosine_sim_emat2_avg', meter_cosine_sim_emat2.avg, epoch_num)
        # 12) validating
        # if (epoch_num>=100 and epoch_num % args.get("test_interval_ep", 1) == 0) or iter_num >= max_iterations:
        if iter_num >= 7000 and (epoch_num % args.get("test_interval_ep", 1) == 0 or iter_num >= max_iterations):
            model.eval()
            teacher1.eval()
            teacher2.eval()
            ema_model.eval()

            if "pancreas" in args["root_path"].lower():
                dice_sample_stu = var_all_case_Pancrease(model, args["root_path"], args['data_path'],
                                                         num_classes=num_classes, patch_size=args["patch_size"],
                                                         stride_xy=16, stride_z=16, flag_nms=True)
                dice_sample_t1 = var_all_case_Pancrease(teacher1, args["root_path"], args['data_path'],
                                                        num_classes=num_classes, patch_size=args["patch_size"],
                                                        stride_xy=16, stride_z=16, flag_nms=True)
                dice_sample_t2 = var_all_case_Pancrease(teacher2, args["root_path"], args['data_path'],
                                                        num_classes=num_classes, patch_size=args["patch_size"],
                                                        stride_xy=16, stride_z=16, flag_nms=True)
                dice_sample_ema = var_all_case_Pancrease(ema_model, args["root_path"], args['data_path'],
                                                         num_classes=num_classes, patch_size=args["patch_size"],
                                                         stride_xy=16, stride_z=16, flag_nms=True)
            else:
                dice_sample_stu = var_all_case_LA(model, args["root_path"], args['data_path'], num_classes=num_classes,
                                                  patch_size=args["patch_size"], stride_xy=18, stride_z=4,
                                                  flag_nms=True)
                dice_sample_t1 = var_all_case_LA(teacher1, args["root_path"], args['data_path'],
                                                 num_classes=num_classes, patch_size=args["patch_size"], stride_xy=18,
                                                 stride_z=4, flag_nms=True)
                dice_sample_t2 = var_all_case_LA(teacher2, args["root_path"], args['data_path'],
                                                 num_classes=num_classes, patch_size=args["patch_size"], stride_xy=18,
                                                 stride_z=4, flag_nms=True)
                dice_sample_ema = var_all_case_LA(ema_model, args["root_path"], args['data_path'],
                                                  num_classes=num_classes, patch_size=args["patch_size"], stride_xy=18,
                                                  stride_z=4, flag_nms=True)
            with open(dice_txt, "a") as f:
                f.writelines('epoch:{}, stu average metric is {}\n'.format(epoch_num, dice_sample_stu))
                f.writelines('epoch:{}, ema average metric is {}\n'.format(epoch_num, dice_sample_ema))
                f.writelines('epoch:{}, tea1 average metric is {}\n'.format(epoch_num, dice_sample_t1))
                f.writelines('epoch:{}, tea2 average metric is {}\n'.format(epoch_num, dice_sample_t2))
            if dice_sample_stu > best_dice_stu:
                best_dice_stu = dice_sample_stu
                tmp_stu_snapshot_path = os.path.join(snapshot_path, "student")
                if not os.path.exists(tmp_stu_snapshot_path):
                    os.makedirs(tmp_stu_snapshot_path, exist_ok=True)

                save_mode_path_stu = os.path.join(tmp_stu_snapshot_path,
                                                  'ep_{:0>3}_dice_{}.pth'.format(epoch_num, round(best_dice_stu, 4)))
                torch.save(model.state_dict(), save_mode_path_stu)

                save_best_path_stu = os.path.join(snapshot_path, '{}_best_stu_model.pth'.format(args["model"]))
                torch.save(model.state_dict(), save_best_path_stu)

            if dice_sample_t1 > best_dice_t1:
                best_dice_t1 = round(dice_sample_t1, 4)
                tmp_tea_snapshot_path = os.path.join(snapshot_path, "teacher1")
                if not os.path.exists(tmp_tea_snapshot_path):
                    os.makedirs(tmp_tea_snapshot_path, exist_ok=True)

                save_mode_path = os.path.join(tmp_tea_snapshot_path,
                                              'ep_{:0>3}_dice_{}.pth'.format(epoch_num, best_dice_t1))
                torch.save(teacher1.state_dict(), save_mode_path)

                save_best_path = os.path.join(snapshot_path, '{}_best_tea1_model.pth'.format(args["model"]))
                torch.save(teacher1.state_dict(), save_best_path)
            if dice_sample_t2 > best_dice_t2:
                best_dice_t2 = round(dice_sample_t2, 4)
                tmp_tea_snapshot_path = os.path.join(snapshot_path, "teacher2")
                if not os.path.exists(tmp_tea_snapshot_path):
                    os.makedirs(tmp_tea_snapshot_path, exist_ok=True)

                save_mode_path = os.path.join(tmp_tea_snapshot_path,
                                              'ep_{:0>3}_dice_{}.pth'.format(epoch_num, best_dice_t2))
                torch.save(teacher2.state_dict(), save_mode_path)

                save_best_path = os.path.join(snapshot_path, '{}_best_tea2_model.pth'.format(args["model"]))
                torch.save(teacher2.state_dict(), save_best_path)
            if dice_sample_ema > best_dice_ema:
                best_dice_ema = round(dice_sample_ema, 4)
                tmp_tea_snapshot_path = os.path.join(snapshot_path, "ema")
                if not os.path.exists(tmp_tea_snapshot_path):
                    os.makedirs(tmp_tea_snapshot_path, exist_ok=True)

                save_mode_path = os.path.join(tmp_tea_snapshot_path,
                                              'ep_{:0>3}_dice_{}.pth'.format(epoch_num, best_dice_ema))
                torch.save(ema_model.state_dict(), save_mode_path)

                save_best_path = os.path.join(snapshot_path, '{}_best_ema_model.pth'.format(args["model"]))
                torch.save(ema_model.state_dict(), save_best_path)
            # writer
            writer.add_scalar('Var_dice/Dice_ema', dice_sample_ema, epoch_num)
            writer.add_scalar('Var_dice/Best_dice_ema', best_dice_ema, epoch_num)
            writer.add_scalar('Var_dice/Dice', dice_sample_stu, epoch_num)
            writer.add_scalar('Var_dice/Best_dice', best_dice_stu, epoch_num)
            writer.add_scalar('Var_dice/Dice_t1', dice_sample_t1, epoch_num)
            writer.add_scalar('Var_dice/Best_dice_t1', best_dice_t1, epoch_num)
            writer.add_scalar('Var_dice/Dice_t2', dice_sample_t2, epoch_num)
            writer.add_scalar('Var_dice/Best_dice_t2', best_dice_t2, epoch_num)

            # csv
            tmp_results_ts = {
                'loss_total': meter_train_losses.avg,
                'loss_sup': meter_sup_losses.avg,
                'loss_unsup': meter_uns_losses.avg,
                'avg_high_ratio': meter_highc_ratio.avg,
                'avg_high_ratio_t': meter_highc_ratio_t.avg,
                'learning_rate': meter_learning_rates.avg,
                'Dice_ema': dice_sample_ema,
                'Dice_ema_best': best_dice_ema,
                'Dice_stu': dice_sample_stu,
                'Dice_stu_best': best_dice_stu,
                'Dice_t1': dice_sample_t1,
                'Dice_t1_best': best_dice_t1,
                'Dice_t2': dice_sample_t2,
                'Dice_t2_best': best_dice_t2
            }
            data_frame = pd.DataFrame(data=tmp_results_ts, index=range(epoch_num, epoch_num + 1))
            if epoch_num > 0 and osp.exists(csv_test):
                data_frame.to_csv(csv_test, mode='a', header=None, index_label='epoch')
            else:
                data_frame.to_csv(csv_test, index_label='epoch')

            # logs
            logging.info('iteration %d : dice_score: %f best_dice: %f' % (iter_num, dice_sample_ema, best_dice_ema))
            logging.info(" <<Test>> - Ep:{}  - Dice-S/T:{:.2f}/{:.2f}, Best-S:{:.2f}, Best-T:{:.2f}".format(epoch_num,
                                                                                                            dice_sample_stu * 100,
                                                                                                            dice_sample_ema * 100,
                                                                                                            best_dice_stu * 100,
                                                                                                            best_dice_ema * 100))
            logging.info("          - AvgLoss(lb/ulb/all):{:.2f}/{:.2f}/{:.2f}, highR:/{:.2f}".format(
                meter_sup_losses.avg, meter_uns_losses.avg, meter_train_losses.avg, meter_highc_ratio.avg))

            model.train()
            teacher1.train()
            teacher2.train()
            ema_model.train()

        if (epoch_num + 1) % args.get("save_interval_epoch", 1000000) == 0:
            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def writer_cosine_sim(ema_model, iter_num, meter_cosine_sim_sema, meter_cosine_sim_st1, meter_cosine_sim_st2,
                      meter_cosine_sim_t12, meter_cosine_sim_emat1, meter_cosine_sim_emat2, model, teacher1, teacher2,
                      writer, img_ulb_w):
    with torch.no_grad():
        cosine_sim_t12 = cosine_similarity(teacher1, teacher2, img_ulb_w)
        cosine_sim_st1 = cosine_similarity(model, teacher1, img_ulb_w)
        cosine_sim_st2 = cosine_similarity(model, teacher2, img_ulb_w)
        cosine_sim_sema = cosine_similarity(model, ema_model, img_ulb_w)
        cosine_sim_emat1 = cosine_similarity(ema_model, teacher1, img_ulb_w)
        cosine_sim_emat2 = cosine_similarity(ema_model, teacher2, img_ulb_w)
    writer.add_scalar('ratio/cosine_sim_t12', cosine_sim_t12, iter_num)
    writer.add_scalar('ratio/cosine_sim_st1', cosine_sim_st1, iter_num)
    writer.add_scalar('ratio/cosine_sim_st2', cosine_sim_st2, iter_num)
    writer.add_scalar('ratio/cosine_sim_sema', cosine_sim_sema, iter_num)
    writer.add_scalar('ratio/cosine_sim_emat1', cosine_sim_emat1, iter_num)
    writer.add_scalar('ratio/cosine_sim_emat2', cosine_sim_emat2, iter_num)
    meter_cosine_sim_t12.update(cosine_sim_t12.item())
    meter_cosine_sim_st1.update(cosine_sim_st1.item())
    meter_cosine_sim_st2.update(cosine_sim_st2.item())
    meter_cosine_sim_sema.update(cosine_sim_sema.item())
    meter_cosine_sim_emat1.update(cosine_sim_emat1.item())
    meter_cosine_sim_emat2.update(cosine_sim_emat2.item())

    logging.info('iter_num:{},cosine_sim_t12:{},cosine_sim_st1:{},cosine_sim_st2:{},cosine_sim_sema:{}'.format(iter_num,
                                                                                                               cosine_sim_t12,
                                                                                                               cosine_sim_st1,
                                                                                                               cosine_sim_st2,
                                                                                                               cosine_sim_sema))


def cal_confidence_loss(pre, APM, t, label):
    # pre = torch.softmax(pre, dim=1)
    # pre = pre.detach()
    # entropy = -torch.sum(pre * torch.log(pre + 1e-10), dim=1)
    # entropy /= np.log(pre.shape[1])
    # confidence = 1.0 - entropy

    probs = F.softmax(pre, dim=1)
    top2 = torch.topk(probs, 2, dim=1).values
    APM = top2[:, 0] - top2[:, 1]
    # APM = update_pseudo_margins(APM,margins,t)
    # APM = (APM - APM.min()) / (APM.max() - APM.min())
    out = torch.argmax(probs, dim=1).squeeze(0)
    mask_ind = (label == out)
    # con = torch.ones_like(confidences)
    con = 1 * mask_ind
    return metrics.cal_l1(con, APM)


# 获取模型参数并展平
def get_model_flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    flat_params = torch.cat(params)
    return flat_params


# 计算余弦相似度
def cosine_similarity(model1, model2, img_ulb_w):
    cos_sim = F.cosine_similarity(model1(img_ulb_w)[1].view(-1), model2(img_ulb_w)[1].view(-1), dim=0)
    return cos_sim.detach()


def cal_ratio(dice_array_s, dice_array_t, num_classes, pseudo_label, pseudo_label_t, sum_s, sum_t, target_ulb,
              unimg_idx):
    with torch.no_grad():
        error_mask = pseudo_label != target_ulb
        mask = [pseudo_label == -100][0]
        pseudo_label[mask] = 0
        mask_ratio = mask.bool().float().mean()
        error_ratio = (error_mask.float().sum() - mask.float().sum()) / (
                    torch.ones_like(target_ulb).sum() - mask.float().sum())
        mask_t = pseudo_label_t == -100
        error_mask_t = pseudo_label_t != target_ulb
        pseudo_label_t[mask_t] = 0
        mask_ratio_t = mask_t.bool().float().mean()
        error_ratio_t = (error_mask_t.float().sum() - mask_t.float().sum()) / (
                    torch.ones_like(target_ulb).sum() - mask_t.float().sum())
        dice_s = my_DICE(pseudo_label, target_ulb, num_classes, mask)
        dice_t = my_DICE(pseudo_label_t, target_ulb, num_classes, mask_t)
        for k in range(len(dice_s)):
            s = round(dice_s[k].cpu().item(), 4)
            t = round(dice_t[k].cpu().item(), 4)
            sum_s += s
            sum_t += t
            idx = unimg_idx[k].cpu().item()
            dice_array_s.append([s, idx])
            dice_array_t.append([t, idx])
        del dice_s, dice_t
    return mask_ratio, mask_ratio_t, error_ratio, error_ratio_t, sum_s, sum_t, dice_array_s, dice_array_t


def cal_unsup_loss(alt_flag_conflict_mode, alt_flag_conflict_stu_use_more, args, dice_loss, flag_start_self_learning,
                   loss_ulb_conflict, loss_ulb_consist, mtx_bool_conflict, pred_ulb_s, pseudo_logits, pseudo_logits_stu,
                   pseudo_outputs, pseudo_outputs_stu, var_param_conflict_weight):
    pseudo_mask = pseudo_logits.ge(args["conf_threshold"]).bool()
    high_ratio = pseudo_mask.float().mean()
    if "ce" == args["flag_ulb_loss_type"]:
        if flag_start_self_learning:
            pseudo_mask = pseudo_logits.ge(args["alt_param_threshold_self_training"]).bool()
            high_ratio = pseudo_mask.float().mean()
            pseudo_outputs[~pseudo_mask] = -100
            pseudo_label = pseudo_outputs.clone()
            loss_ulb = F.cross_entropy(pred_ulb_s, pseudo_outputs.long(), ignore_index=-100, reduction="mean")
        else:
            pseudo_outputs, pseudo_logits, conflict_tea_stu = get_compromise_pseudo_btw_tea_stu(pseudo_outputs,
                                                                                                pseudo_logits,
                                                                                                pseudo_outputs_stu,
                                                                                                pseudo_logits_stu,
                                                                                                alt_flag_conflict_mode,
                                                                                                None if alt_flag_conflict_stu_use_more else mtx_bool_conflict)
            if var_param_conflict_weight > 1 or var_param_conflict_weight < 1:
                thresh_mask = pseudo_logits.ge(args["conf_threshold"]).bool()
                pseudo_outputs[~thresh_mask] = -100
                target_consist = pseudo_outputs.clone()
                target_consist[mtx_bool_conflict] = -100
                loss_ulb_consist = F.cross_entropy(pred_ulb_s, target_consist.long(),
                                                   ignore_index=-100, reduction="mean")
                pseudo_label = target_consist.clone()
                if var_param_conflict_weight > 0 or var_param_conflict_weight < 0:
                    target_conflct = pseudo_outputs.clone()
                    target_conflct[~mtx_bool_conflict] = -100
                    loss_ulb_conflict = F.cross_entropy(pred_ulb_s, target_conflct.long(),
                                                        ignore_index=-100, reduction="mean")
                    if loss_ulb_conflict > 0.0:
                        loss_ulb = loss_ulb_consist + var_param_conflict_weight * loss_ulb_conflict
                    else:
                        loss_ulb = loss_ulb_consist * 1.0

                else:
                    loss_ulb = loss_ulb_consist * 1.0


            else:
                pseudo_mask = pseudo_logits.ge(args["conf_threshold"]).bool()
                pseudo_label = pseudo_outputs.clone()
                pseudo_label[~pseudo_mask] = -100
                loss_ulb = F.cross_entropy(pred_ulb_s, pseudo_label.long(), ignore_index=-100, reduction="mean")
    else:
        pseudo_label = pseudo_outputs.clone()
        mask = pseudo_logits.ge(args["conf_threshold"]).bool()
        pseudo_outputs[~mask] = -100
        loss_ulb = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                             pseudo_outputs.unsqueeze(1).float(),
                             ignore=-100)
    return high_ratio, loss_ulb, loss_ulb_conflict, loss_ulb_consist, pseudo_label


def cal_sup_loss(ce_loss, dice_loss, pred_lb, target_lb, iter_num, l1_iter_num):
    margin_loss = 0
    if iter_num >= l1_iter_num:
        margin_loss = args['alpha'] * cal_confidence_loss(pred_lb, 0, 0, target_lb)
        loss_lb = (ce_loss(pred_lb, target_lb.long()) +
                   dice_loss(torch.softmax(pred_lb, dim=1),
                             target_lb.unsqueeze(1).float(),
                             ignore=torch.zeros_like(target_lb).float())
                   + margin_loss
                   ) / 3.0
    else:
        loss_lb = (ce_loss(pred_lb, target_lb.long()) +
                   dice_loss(torch.softmax(pred_lb, dim=1),
                             target_lb.unsqueeze(1).float(),
                             ignore=torch.zeros_like(target_lb).float())
                   ) / 2.0
    return loss_lb, margin_loss


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        III. main process
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if __name__ == "__main__":
    # 1. set up config
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='config_3d_la_aut_25k.yml',
                        help='configuration file')

    # Basics: Data, results, model
    parser.add_argument('--root_path', type=str,
                        default='../data_split/LA', help='Name of Dataset')
    parser.add_argument('--data_path', type=str,
                        default='/home/cq/dataset/LA/Left_Atrium/data', help='Name of Dataset')
    parser.add_argument('--res_path', type=str,
                        default='./results/LA', help='Path to save resutls')
    parser.add_argument('--exp', type=str,
                        default='debug', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='vnet', help='model_name')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('-g', '--gpu_id', type=int, nargs="+", default=[0],
                        help='the id of gpu used to train the model')

    # Training Basics
    parser.add_argument('--max_iterations', type=int,
                        default=15000, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    # https://blog.csdn.net/qq_43391414/article/details/122992458
    parser.add_argument('--patch_size', type=int, nargs="+", default=[112, 112, 80],
                        help='patch size of network input')

    parser.add_argument('--max_samples', type=int,
                        default=80, help='maximum samples to train')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--test_interval_iter', type=int,
                        default=200, help='')
    parser.add_argument('--test_interval_ep', type=int,
                        default=1, help='')
    parser.add_argument('--save_interval_epoch', type=int,
                        default=1000000, help='')
    parser.add_argument("-p", "--poly", default=False,
                        action='store_true', help="whether poly scheduler")

    # label and unlabel
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=2,
                        help='labeled_batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=4,
                        help='labeled data')

    # model related
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument("--flag_pseudo_from_student", default=False,
                        action='store_true', help="using pseudo from student itself")

    # augmentation
    parser.add_argument('--cutmix_prob', type=float,
                        default=1.0, help='probability of applying cutmix')

    # unlabeled loss
    parser.add_argument('--consistency', type=float,
                        default=1.0, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=40.0, help='consistency_rampup')
    parser.add_argument("--conf_threshold",
                        type=float,
                        default=0.8,
                        help="confidence threshold for using pseudo-labels")
    parser.add_argument('--flag_ulb_loss_type', type=str,
                        default="dice", help='loss type, ce, dice, dice+ce')
    parser.add_argument("--flag_sampling_based_on_lb",
                        default=False, action='store_true', help="using dynamic cutmix")

    # steps
    parser.add_argument('--max_step', type=float,
                        default=180, help='consistency_rampup')
    parser.add_argument('--min_step', type=float,
                        default=180, help='consistency_rampup')
    parser.add_argument('--start_step1', type=float,
                        default=180, help='consistency_rampup')
    parser.add_argument('--start_step2', type=float,
                        default=180, help='consistency_rampup')
    parser.add_argument('--coefficient', '-c', type=float,
                        default=3, help='consistency_rampup')
    parser.add_argument('--alpha', '-a', type=float,
                        default=1, help='consistency_rampup')
    # parse args
    args = parser.parse_args()
    args = vars(args)

    # 2. update from the config files
    cfgs_file = args['cfg']
    cfgs_file = os.path.join('../cfgs', cfgs_file)
    with open(cfgs_file, 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # convert "1e-x" to float
    for each in options_yaml.keys():
        tmp_var = options_yaml[each]
        if type(tmp_var) == str and "1e-" in tmp_var:
            options_yaml[each] = float(tmp_var)
    # update original parameters of argparse
    update_values(options_yaml, args)
    import pprint

    # 3. setup gpus and randomness
    # if args["gpu_id"] in range(8):
    gpu_list_str = ','.join(map(str, args["gpu_id"]))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_str
    args["device_id"] = list(range(len(args["gpu_id"])))
    if not args["deterministic"]:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args["seed"] > 0:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])

    # 4. outputs and logger
    snapshot_path = "{}/{}_labeled_{}/{}".format(
        args["res_path"], args["labeled_num"], args["exp"], args["model"])
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    shutil.copy('../code/train_post_3d_aut_addema.py', snapshot_path)
    shutil.copy('../code/networks/vnet.py', snapshot_path)
    shutil.copy('../code/networks/net_factory.py', snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)
