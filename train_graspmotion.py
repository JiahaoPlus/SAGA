import argparse
import datetime
import itertools
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

from MotionFill.data.GRAB_end2end_dataloader import GRAB_DataLoader
from MotionFill.models.LocalMotionFill import Motion_CNN_CVAE
from MotionFill.models.TrajFill import Traj_MLP_CVAE
from utils.como.como_utils import get_logger, get_scheduler, save_config
from utils.Pivots_torch import Pivots_torch
from utils.Quaternions_torch import Quaternions_torch
from utils.train_helper import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='dataset/GraspMotion')
parser.add_argument('--body_model_path', type=str, default='body_utils/body_models')
parser.add_argument('--gpu_id', type=int, default='0')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=410)
parser.add_argument("--log_step", default=500, type=int)
parser.add_argument("--save_step", default=500, type=int)
parser.add_argument('--save_dir', type=str, default='logs/log_infill_2stage_end2end_grab')

# put the path of pretrained TrajFill and LocalMotionFill here
# can pretrain them separately
parser.add_argument('--pretrained_path_traj', type=str, default='PATH/TO/TRAJ_MODEL')
parser.add_argument('--pretrained_path_motion', type=str, default='PATH/TO/MOTION_MODEL')

# settings for body representation
parser.add_argument("--clip_seconds", default=2, type=int)
parser.add_argument("--clip_fps", default=30, type=int)
parser.add_argument('--body_mode', type=str, default='local_markers_3dv_4chan',
                    choices=['local_markers_3dv', 'local_markers_3dv_4chan'])
parser.add_argument('--global_rot_norm', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--with_hand', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--normalize', default='True', type=lambda x: x.lower() in ['true', '1'])

# settings for network
parser.add_argument('--downsample', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument("--conv_k", default=3, type=int)
parser.add_argument("--nz", default=512, type=int)
parser.add_argument("--traj_nz", default=512, type=int)
parser.add_argument('--traj_source', type=str, default='generated')
parser.add_argument('--traj_smoothed', default='False', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--traj_residual', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument("--n_traj_samples", default=20, type=int)

# loss weights
parser.add_argument("--weight_loss_rec_body", default=1.0, type=float)
parser.add_argument("--weight_loss_rec_body_v", default=1.0, type=float)
parser.add_argument("--weight_loss_rec_contact_lbl", default=0.05, type=float)
parser.add_argument("--weight_loss_KLD", default=0.5, type=float)
parser.add_argument("--weight_loss_rec_traj", default=1.0, type=float)
parser.add_argument("--weight_loss_rec_traj_v", default=1.0, type=float)
parser.add_argument("--weight_loss_KLD_traj", default=0.5, type=float)

parser.add_argument('--debug', default='False', type=lambda x: x.lower() in ['true', '1'])

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())


def train(writer, logger):
    if args.debug == True:
        print('debug mode')
        train_datasets = ['s1']
        test_datasets = ['s10']

    else:
        train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
        test_datasets = ['s9', 's10']

    # set dataloaders
    print('[INFO] reading training data from datasets {}...'.format(train_datasets))
    train_dataset = GRAB_DataLoader(clip_seconds=args.clip_seconds, clip_fps=args.clip_fps, 
                                    normalize=args.normalize, split='train', mode=args.body_mode, 
                                    is_debug=args.debug, markers_type='f0_p5', log_dir=logdir)
    train_dataset.read_data(train_datasets, args.dataset_dir)
    train_dataset.create_body_repr(global_rot_norm=args.global_rot_norm, with_hand=args.with_hand,
                                   smplx_model_path=args.body_model_path)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, drop_last=True)

    print('[INFO] reading test data from datasets {}...'.format(test_datasets))
    test_dataset = GRAB_DataLoader(clip_seconds=args.clip_seconds, clip_fps=args.clip_fps, 
                                   normalize=args.normalize, split='test', mode=args.body_mode, 
                                   is_debug=args.debug, markers_type='f0_p5', log_dir=logdir)
    test_dataset.read_data(test_datasets, args.dataset_dir)
    test_dataset.create_body_repr(global_rot_norm=args.global_rot_norm, with_hand=args.with_hand,
                                  smplx_model_path=args.body_model_path)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=True)

    # load motion data stat (mean, var)
    prefix = os.path.join(logdir, 'prestats_GRAB_contact_given_global')
    if args.with_hand:
        prefix += '_withHand'
    motion_stats = np.load('{}_{}.npz'.format(prefix, args.body_mode))

    traj_stats = np.load(os.path.join(logdir, 'prestats_GRAB_traj.npz'))
    traj_Xmean = torch.from_numpy(traj_stats['traj_Xmean']).float().to(device).unsqueeze(0)
    traj_Xstd = torch.from_numpy(traj_stats['traj_Xstd']).float().to(device).unsqueeze(0)

    # set train configs 
    traj_model = Traj_MLP_CVAE(nz=args.traj_nz, feature_dim=4, T=62, residual=args.traj_residual, 
                               load_path=args.pretrained_path_traj).to(device)
    motion_model = Motion_CNN_CVAE(nz=args.nz, downsample=args.downsample, in_channel=4, 
                                   kernel=args.conv_k, clip_seconds=args.clip_seconds).to(device)

    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(traj_model.parameters())),
                           lr=args.lr)
    scheduler1 = get_scheduler(optimizer1, policy='step', decay_step=200)
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(motion_model.parameters())),
                           lr=args.lr)
    scheduler2 = get_scheduler(optimizer2, policy='step', decay_step=200)

    print('loading traj model from checkpoint: %s' % args.pretrained_path_traj)
    model_cp1 = torch.load(args.pretrained_path_traj)
    traj_model.load_state_dict(model_cp1['model_dict'])
    print('loading motion model from checkpoint: %s' % args.pretrained_path_motion)
    model_cp2 = torch.load(args.pretrained_path_motion)
    motion_model.load_state_dict(model_cp2['model_dict'])

    bce_loss = nn.BCEWithLogitsLoss().to(device)
    mask_t_1 = [0, args.clip_fps*args.clip_seconds]
    mask_t_0 = list(set(range(args.clip_fps*args.clip_seconds+1)) - set(mask_t_1))
    print('Mask the markers in the following frames: ', mask_t_0)
    print('Training status is being recorded in the log.')
    
    # start training 
    total_steps = 0
    for epoch in range(args.num_epoch):
        for step, data in (enumerate(train_dataloader)):
            traj_model.train()
            motion_model.train()
            total_steps += 1
            [clip_img, traj_gt, smplx_beta, gender, rot_0_pivot, transf_matrix_smplx, 
                    smplx_params_gt, marker_start, marker_end, joint_start, joint_end] = [item.float().to(device) for item in data]
            if torch.any(clip_img.isnan()):
                logger.info('nan detected in input data, skip this batch!')
                continue
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            bs = clip_img.shape[0]
            d = clip_img.shape[-2]
            T = clip_img.shape[-1]

            traj_sr_input, traj_sr_input_unnormed, transf_rotmat, joint_start_new, \
                    joint_end_new = prepare_traj_input(joint_start, joint_end, traj_Xmean, traj_Xstd)  # Note: this is the joint forward
            
            traj_pred, traj_mu, traj_logvar = traj_model.forward(traj_gt.view(bs, -1), traj_sr_input.view(bs, -1))

            traj_mean = traj_Xmean.unsqueeze(2)
            traj_std = traj_Xstd.unsqueeze(2)
            traj_pred_unnormed = traj_pred * traj_std + traj_mean

            clip_img_input, rot_0_pivot, marker_start_new, marker_end_new, traj_input = prepare_clip_img_input_torch(clip_img, 
                marker_start, marker_end, joint_start, joint_end, joint_start_new, joint_end_new, transf_rotmat, 
                traj_pred_unnormed, traj_sr_input_unnormed, args.traj_smoothed, motion_stats)

            # mask input
            clip_img_input[:, 0, 2:, mask_t_0] = 0. # pelvis z also unknown
            clip_img_input[:, 0, -4:, :] = 0.

            # forward
            clip_img_rec, mu, logvar = motion_model(clip_img_input, clip_img)  

            """ loss """
            # traj loss
            traj_gt_v = traj_gt[:, :, 1:] - traj_gt[:, :, 0:-1]
            traj_rec_v = traj_pred[:, :, 1:] - traj_pred[:, :, 0:-1]

            weight = 0.5
            loss_rec_traj = F.l1_loss(traj_gt, traj_pred) + weight * F.l1_loss(traj_gt[:, :, 0], traj_pred[:, :, 0]) + weight * F.l1_loss(traj_gt[:, :, -1], traj_pred[:, :, -1])
            loss_rec_traj_v = F.l1_loss(traj_gt_v, traj_rec_v)
            
            KLD_traj = 0.5 * torch.mean(-1 - traj_logvar + traj_mu.pow(2) + traj_logvar.exp())
            # robust KLD
            loss_KLD_traj = torch.sqrt(1 + KLD_traj**2)-1
            nan_count = 0
            if loss_KLD_traj.isnan():
                logger.info('loss_KLD_traj is nan')
                logger.info('traj_logvar={:.4f}, traj_logvar.exp()={:.4f},\
                    traj_mu={:.4f}, traj_mu.pow(2)={:.4f}'.format(torch.mean(traj_logvar).item(), 
                    torch.mean(traj_logvar.exp()).item(), torch.mean(traj_mu).item(), 
                    torch.mean(traj_mu.pow(2)).item()))
                nan_count += 1
            if nan_count > 0:
                logger.info('skip this batch')
                continue
            
            # motion loss
            clip_img_v = clip_img[:, :, :, 1:] - clip_img[:, :, :, 0:-1]
            clip_img_rec_v = clip_img_rec[:, :, :, 1:] - clip_img_rec[:, :, :, 0:-1]  # velocity
            loss_rec_body = F.l1_loss(clip_img[:, 0, 0:-4], clip_img_rec[:, 0, 0:-4]) 
            loss_rec_body_v = F.l1_loss(clip_img_v[:, 0, 0:-4], clip_img_rec_v[:, 0, 0:-4])
            loss_rec_contact_lbl = bce_loss(clip_img_rec[:, 0, -4:], clip_img[:, 0, -4:])
            
            KLD = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())
            # robust KLD
            loss_KLD = torch.sqrt(1 + KLD**2)-1
            nan_count = 0
            if loss_KLD.isnan():
                logger.info('loss_KLD is nan')
                logger.info('logvar={:.4f}, logvar.exp()={:.4f},\
                    mu={:.4f}, mu.pow(2)={:.4f}'.format(torch.mean(logvar).item(), 
                    torch.mean(logvar.exp()).item(), torch.mean(mu).item(), torch.mean(mu.pow(2)).item()))
                nan_count += 1
            if nan_count > 0:
                logger.info('skip this batch')
                continue

            loss = args.weight_loss_rec_body * loss_rec_body + \
                   args.weight_loss_rec_body_v * loss_rec_body_v + \
                   args.weight_loss_rec_contact_lbl * loss_rec_contact_lbl + \
                   args.weight_loss_KLD * loss_KLD + \
                   args.weight_loss_rec_traj * loss_rec_traj + \
                   args.weight_loss_rec_traj_v * loss_rec_traj_v + \
                   args.weight_loss_KLD_traj * loss_KLD_traj

            loss.backward()
            torch.nn.utils.clip_grad_norm_(traj_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(motion_model.parameters(), 1.0)
            optimizer1.step()
            optimizer2.step()

            # log train loss 
            if total_steps % args.log_step == 0:
                # local motion
                writer.add_scalar('train/loss_rec_body', loss_rec_body.item(), total_steps)
                writer.add_scalar('train/loss_rec_body_v', loss_rec_body_v.item(), total_steps)
                writer.add_scalar('train/loss_rec_contact_lbl', loss_rec_contact_lbl.item(), total_steps)
                writer.add_scalar('train/loss_KLD', loss_KLD.item(), total_steps)
                # traj
                writer.add_scalar('train/loss_rec_traj', loss_rec_traj.item(), total_steps)
                writer.add_scalar('train/loss_rec_traj_v', loss_rec_traj_v.item(), total_steps)
                writer.add_scalar('train/loss_KLD_traj', loss_KLD_traj.item(), total_steps)
                lr = optimizer2.param_groups[0]['lr']
                print_str = '[Step {:d}/ Epoch {:d}] \t| L_body: {:.6f} \t L_v: {:.6f} \t L_contact: {:.6f} \t L_kld: {:.6f} \t L_traj: {:.6f} \t L_traj_v: {:.6f} \t L_traj_kld: {:.6f} \t lr: {:.6f} \t TOTAL: {:.6f}'. \
                    format(step+1, epoch+1, loss_rec_body.item(), loss_rec_body_v.item(), loss_rec_contact_lbl.item(), loss_KLD.item(), loss_rec_traj.item(), loss_rec_traj_v.item(), loss_KLD_traj.item(), lr, loss.item())
                logger.info(print_str)

            # test loss 
            if total_steps % args.log_step == 0:
                loss_rec_body_test,  loss_rec_body_v_test = 0, 0
                loss_rec_contact_lbl_test = 0
                loss_rec_traj_test = 0
                loss_rec_traj_v_test = 0
                
                with torch.no_grad():
                    for test_step, data in (enumerate(test_dataloader)):
                        traj_model.eval()
                        motion_model.eval()
                        [clip_img, traj_gt, smplx_beta, gender, rot_0_pivot, transf_matrix_smplx, 
                            smplx_params_gt, marker_start, marker_end, joint_start, joint_end] = [item.float().to(device) for item in data]

                        traj_sr_input, traj_sr_input_unnormed, transf_rotmat, joint_start_new, \
                                joint_end_new = prepare_traj_input(joint_start, joint_end, traj_Xmean, traj_Xstd)  # Note: this is the joint forward

                        traj_pred, traj_mu, traj_logvar = traj_model.forward(traj_gt.view(bs, -1), traj_sr_input.view(bs, -1))

                        traj_mean = traj_Xmean.unsqueeze(2)
                        traj_std = traj_Xstd.unsqueeze(2)
                        traj_pred_unnormed = traj_pred * traj_std + traj_mean

                        clip_img_input, rot_0_pivot, marker_start_new, marker_end_new, traj_input = prepare_clip_img_input_torch(clip_img, marker_start, marker_end, joint_start, joint_end, joint_start_new, joint_end_new, transf_rotmat, traj_pred_unnormed, traj_sr_input_unnormed, args.traj_smoothed, motion_stats)

                        # mask input
                        clip_img_input[:, 0, 2:, mask_t_0] = 0. # pelvis z also unknown
                        clip_img_input[:, 0, -4:, :] = 0.

                        # forward
                        clip_img_rec, _, _ = motion_model(clip_img_input, clip_img, is_train=False)  # z: [bs, 256, d, T], clip_img_rec: [bs, 1, d, T]

                        """ loss """
                        # traj loss
                        traj_gt_v = traj_gt[:, :, 1:] - traj_gt[:, :, 0:-1]
                        traj_rec_v = traj_pred[:, :, 1:] - traj_pred[:, :, 0:-1]
                        loss_rec_traj_test = F.l1_loss(traj_gt, traj_pred) 
                        loss_rec_traj_v_test = F.l1_loss(traj_gt_v, traj_rec_v)
                        clip_img_test_v = clip_img[:, :, :, 1:] - clip_img[:, :, :, 0:-1]  # velocity
                        clip_img_test_rec_v = clip_img_rec[:, :, :, 1:] - clip_img_rec[:, :, :, 0:-1]  # velocity
                        loss_rec_body_test += F.l1_loss(clip_img[:, 0, 0:-4], clip_img_rec[:, 0, 0:-4])
                        loss_rec_body_v_test += F.l1_loss(clip_img_test_v[:, 0, 0:-4], clip_img_test_rec_v[:, 0, 0:-4])
                        loss_rec_contact_lbl_test += bce_loss(clip_img_rec[:, 0, -4:], clip_img[:, 0, -4:])

                loss_rec_body_test = loss_rec_body_test / (test_step + 1)
                loss_rec_body_v_test = loss_rec_body_v_test / (test_step + 1)
                loss_rec_contact_lbl_test = loss_rec_contact_lbl_test / (test_step + 1)
                loss_rec_traj_test = loss_rec_traj_test / (test_step + 1)
                loss_rec_traj_v_test = loss_rec_traj_v_test / (test_step + 1)

                # log test loss
                writer.add_scalar('test/loss_rec_body_test', loss_rec_body_test, total_steps)
                writer.add_scalar('test/loss_rec_body_v_test', loss_rec_body_v_test, total_steps)
                writer.add_scalar('test/loss_rec_contact_lbl_test', loss_rec_contact_lbl_test, total_steps)
                writer.add_scalar('test/loss_rec_traj_test', loss_rec_traj_test, total_steps)
                writer.add_scalar('test/loss_rec_traj_v_test', loss_rec_traj_v_test, total_steps)
                print_str = '(Test) \t\t\t| L_body: {:.6f} \t L_v: {:.6f} \t L_contact: {:.6f} \t L_traj: {:.6f} \t L_traj_v: {:.6f}'. \
                    format(loss_rec_body_test.item(), loss_rec_body_v_test.item(), loss_rec_contact_lbl_test.item(),
                           loss_rec_traj_test.item(), loss_rec_traj_v_test.item())
                logger.info(print_str)

            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "LocalMotionFill_model.pkl")
                model_save = {'model_dict': motion_model.state_dict(), 
                              'optimizer': optimizer2.state_dict()}
                torch.save(model_save, save_path)
                save_path_traj = os.path.join(writer.file_writer.get_logdir(), "TrajFill_model.pkl")
                model_save_traj = {'model_dict': traj_model.state_dict(), 
                                   'optimizer': optimizer1.state_dict()}
                torch.save(model_save_traj, save_path_traj)

        scheduler1.step()
        scheduler2.step()


if __name__ == '__main__':
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    logdir = os.path.join(args.save_dir, ts+'_'+str(args.nz)+'_'+str(args.weight_loss_KLD)+'_'+str(args.clip_seconds)+'s_align_2_update_2net')  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()

    logger = get_logger(logdir)
    logger.info('Start') 
    save_config(logdir, args)

    train(writer, logger)


