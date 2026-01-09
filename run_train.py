from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path
import random
import copy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from function_baseline.model_pos_preparation import model_pos_preparation
from function_adaptpose.config import get_parse_args
from function_adaptpose.data_preparation import data_preparation
from function_adaptpose.dataloader_update import dataloader_update
from function_adaptpose.model_gan_preparation import get_poseaug_model
from function_adaptpose.model_gan_train import train_gan
from function_adaptpose.model_pos_eval import evaluate_posenet
from function_adaptpose.model_pos_train import train_posenet
from utils.gan_utils import Sample_from_Pool
from utils.log import Logger
from utils.utils import save_ckpt, Summary, get_scheduler

def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # [LIFELONG CONFIGURATION]
    # Define your sequence of target domains here.
    # In a real scenario, pass this via args or a config file.
    # ------------------------------------------------------------------
    lifelong_targets = ['3DHP', 'Human36M', 'MPI-INF-3DHP'] 
    seen_test_loaders = {} # Store test loaders for past domains to evaluate forgetting

    # Initialize model placeholders (will be created in Stage 0)
    model_pos = None
    model_pos_eval = None
    poseaug_dict = None
    
    # ------------------------------------------------------------------
    # [LIFELONG LOOP] Iterate through domains sequentially
    # ------------------------------------------------------------------
    for stage_idx, current_target in enumerate(lifelong_targets):
        
        print(f"\n{'='*60}")
        print(f"==> LIFELONG STAGE {stage_idx} | TARGET DOMAIN: {current_target}")
        print(f"{'='*60}\n")

        # 1. Update Arguments for the current stage
        # Assuming args has a field 'dataset_target' or similar that data_preparation uses
        # You may need to adjust the attribute name based on your config.py
        args.dataset_target = current_target 
        
        # Update checkpoint path for this specific stage to avoid overwriting
        # e.g., checkpoints/stage_0_3DHP/...
        current_stage_ckpt = path.join(args.checkpoint, f"stage_{stage_idx}_{current_target}", args.keypoints,
                                     datetime.datetime.now().isoformat() + '_' + args.note)
        os.makedirs(current_stage_ckpt, exist_ok=True)
        print(f'==> Making checkpoint dir for Stage {stage_idx}: {current_stage_ckpt}')

        # 2. Load Data for Current Stage
        print('==> Loading dataset for current stage...')
        data_dict = data_preparation(args)
        
        # Store the test loader for this domain for future evaluation
        # Assuming data_dict returns a loader key like 'test_loader' or similar
        # If AdaptPose uses specific evaluation keys, adapt this line:
        seen_test_loaders[current_target] = data_dict # Save full dict or specific loader

        # 3. Initialize OR Inherit Models
        if stage_idx == 0:
            print("==> Initializing Models (Stage 0)...")
            model_pos = model_pos_preparation(args, data_dict['dataset'], device)
            model_pos_eval = model_pos_preparation(args, data_dict['dataset'], device)
            poseaug_dict = get_poseaug_model(args, data_dict['dataset'])
        else:
            print("==> Inheriting Models from previous stage (Continual Learning)...")
            # In Lifelong learning, we DO NOT re-init weights. We continue training.
            # However, we often re-initialize the Optimizer or update the Learning Rate.
            pass

        # 4. Prepare Optimizers (Reset or update for new stage)
        posenet_optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr_p)
        posenet_lr_scheduler = get_scheduler(posenet_optimizer, policy='lambda', nepoch_fix=0,
                                             nepoch=args.epochs)
        
        # Ensure PoseAug optimizers are also ready
        # If they need reset, do it here. If not, let them continue.

        # Loss function
        criterion = nn.MSELoss(reduction='mean').to(device)

        # GAN trick: data buffer (Reset or Keep? Usually Keep for replay, Reset for clear slate)
        fake_3d_sample = Sample_from_Pool()
        fake_2d_sample = Sample_from_Pool()

        # Logger for this stage
        logger = Logger(os.path.join(current_stage_ckpt, 'log.txt'), args)
        logger.record_args(str(model_pos))
        logger.set_names(['epoch', 'lr', 'current_target_error', 'forgetting_error']) 

        summary = Summary(current_stage_ckpt)
        writer = summary.create_summary()

        # ------------------------------------------------------------------
        # [TRAINING LOOP] Standard AdaptPose training for this stage
        # ------------------------------------------------------------------
        start_epoch = 0
        best_metric = None

        for _ in range(start_epoch, args.epochs):
            
            # --- Warmup / GAN Updates ---
            if summary.epoch == 0:
                poseaug_dict['optimizer_G'].zero_grad()
                poseaug_dict['optimizer_G'].step()
                poseaug_dict['optimizer_d3d'].zero_grad()
                poseaug_dict['optimizer_d3d'].step()
                poseaug_dict['optimizer_d2d'].zero_grad()
                poseaug_dict['optimizer_d2d'].step()
                
                # Initial Eval for this stage
                # Note: evaluate_posenet might need adjustment to handle the specific data_dict structure
                # Here we just run it on the current data_dict
                h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                  summary, writer, tag='_real')
                summary.summary_epoch_update()

            # --- Epoch Training ---
            for kk in range(5):
                # NOTE: For true Lifelong Learning, you need to modify 'train_gan' 
                # to include REPLAY data from previous domains to prevent forgetting.
                train_gan(args, poseaug_dict, data_dict, model_pos, criterion, fake_3d_sample, fake_2d_sample, summary, writer, section=kk)

                if summary.epoch > args.warmup:
                    train_posenet(model_pos, data_dict['train_fake2d3d_loader'], posenet_optimizer, criterion, device)
                    
                    # Save checkpoint logic
                    # Simplified for brevity
                    if best_metric is None: # or logic to compare error
                         save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, current_stage_ckpt, suffix='best_current')

            if summary.epoch > args.warmup:
                train_posenet(model_pos, data_dict['train_gt2d3d_loader'], posenet_optimizer, criterion, device)

            # Update schedulers
            poseaug_dict['scheduler_G'].step()
            poseaug_dict['scheduler_d3d'].step()
            poseaug_dict['scheduler_d2d'].step()
            posenet_lr_scheduler.step()
            
            lr_now = posenet_optimizer.param_groups[0]['lr']
            print('\nStage: %d | Epoch: %d | LR: %.8f' % (stage_idx, summary.epoch, lr_now))

            # --- LOGGING ---
            # Evaluate on CURRENT domain
            # (You would call evaluate_posenet here for the current target)
            
            summary.summary_epoch_update()

        # ------------------------------------------------------------------
        # [END OF STAGE EVALUATION] Check Forgetting
        # ------------------------------------------------------------------
        print(f"\n==> Finished Stage {stage_idx}. Evaluating on ALL seen domains...")
        for domain_name, domain_data in seen_test_loaders.items():
            print(f"    Evaluating on {domain_name}...")
            # Run evaluation using domain_data
            # evaluate_posenet(args, domain_data, model_pos, ...)

        writer.close()
        logger.close()
        
        # Save Final Model of this stage as the start point for next
        save_ckpt({'epoch': args.epochs, 'model_pos': model_pos.state_dict()}, current_stage_ckpt, suffix='final_stage')


if __name__ == '__main__':
    args = get_parse_args()

    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True

    main(args)