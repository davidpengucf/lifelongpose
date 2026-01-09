from __future__ import print_function, absolute_import, division

import os
import os.path as path
import random
import copy

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from function_adaptpose.config import get_parse_args
from function_adaptpose.data_preparation import data_preparation
from function_baseline.model_pos_preparation import model_pos_preparation
from function_adaptpose.model_pos_eval import evaluate

def main(args):
    print('==> Using settings {}'.format(args))
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # [LIFELONG CONFIGURATION]
    # Define the sequence of domains you want to evaluate on.
    # This should match your training order (or be a superset of it).
    # ------------------------------------------------------------------
    lifelong_domains = ['Human36M', '3DHP', '3DPW'] 
    
    # Store results for final summary
    results_log = {}

    print("==> Loading Model Architecture...")
    # We need to initialize the model once. 
    # We load the initial dataset just to get the skeleton metadata for the model init.
    # (Assuming architecture is shared/compatible across domains)
    args.dataset_target = lifelong_domains[0]
    init_data_dict = data_preparation(args)
    model_pos = model_pos_preparation(args, init_data_dict['dataset'], device)

    # ------------------------------------------------------------------
    # [LOAD CHECKPOINT]
    # ------------------------------------------------------------------
    assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])
    
    model_pos.eval() # Ensure eval mode

    # ------------------------------------------------------------------
    # [LIFELONG EVALUATION LOOP]
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"==> STARTING LIFELONG EVALUATION ON: {lifelong_domains}")
    print(f"{'='*60}\n")

    for domain in lifelong_domains:
        print(f"==> preparing data for domain: {domain}...")
        
        # 1. Update target arg to load correct dataset
        # We use a deepcopy of args to avoid polluting global state if needed, 
        # though simple assignment is usually fine here.
        current_args = copy.deepcopy(args)
        current_args.dataset_target = domain
        
        # 2. Load Data
        # Note: data_preparation usually prints a lot of logs, you might want to suppress them
        data_dict = data_preparation(current_args)

        # 3. Select Loader and Evaluate based on Domain Type
        # AdaptPose uses different keys/params for H36M vs 3DHP
        p1, p2 = 0.0, 0.0

        if 'Human36M' in domain:
            print(f"==> Evaluating on {domain} (H36M Protocol)...")
            loader = data_dict.get('H36M_test')
            if loader:
                # H36M typically uses standard evaluation without 'tag' or 'flipaug' in some baselines,
                # but check your specific evaluate() signature.
                p1, p2 = evaluate(loader, model_pos, device, pad=current_args.pad)
            else:
                print(f"[Warning] H36M_test loader not found in data_dict.")

        elif '3DHP' in domain or '3DPW' in domain:
            print(f"==> Evaluating on {domain} (3DHP Protocol)...")
            loader = data_dict.get('mpi3d_loader')
            if loader:
                # 3DHP often uses flip augmentation and specific tags
                p1, p2 = evaluate(loader, model_pos, device, flipaug='_flip', pad=current_args.pad, tag='3dhp')
            else:
                 print(f"[Warning] mpi3d_loader not found in data_dict.")
        
        else:
            print(f"[Warning] Unknown domain type: {domain}. Skipping...")
            continue

        # 4. Log Results
        print(f'   -> {domain} Results: P1 (MPJPE): {p1:.2f} mm | P2 (P-MPJPE): {p2:.2f} mm')
        results_log[domain] = {'P1': p1, 'P2': p2}

    # ------------------------------------------------------------------
    # [FINAL SUMMARY]
    # ------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("LIFELONG EVALUATION SUMMARY")
    print(f"Checkpoint: {args.evaluate}")
    print(f"{'='*60}")
    print(f"{'Domain':<20} | {'MPJPE (P1)':<15} | {'P-MPJPE (P2)':<15}")
    print("-" * 56)
    
    for domain in lifelong_domains:
        if domain in results_log:
            res = results_log[domain]
            print(f"{domain:<20} | {res['P1']:<15.2f} | {res['P2']:<15.2f}")
        else:
            print(f"{domain:<20} | {'N/A':<15} | {'N/A':<15}")
    print("-" * 56)


if __name__ == '__main__':
    args = get_parse_args()
    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

    main(args)