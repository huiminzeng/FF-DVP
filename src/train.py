import os
import copy
import argparse
import numpy as np
import random 

import torch

from model import *
from dataset import *

from client_utils import LocalUpdate

parser = argparse.ArgumentParser()

# dataset arguments
parser.add_argument('--data_path', type=str, default=None, help="path \
                    to dataset")
parser.add_argument('--dataset', type=str, default=None, help="name \
                    of dataset")

# federated arguments (Notation for the arguments followed from paper)
parser.add_argument('--epochs', type=int, default=4,
                    help="number of rounds of training")
parser.add_argument('--batch_size', type=int, default=18,
                    help="local batch size: B")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_users', type=int, default=None,
                    help="number of users: K")
parser.add_argument('--local_ep', type=int, default=10,
                    help="the number of local epochs: E")

parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='SGD weight decay (default: 1d-4)')

# other arguments
parser.add_argument('--save_path', default='trained_models', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

# specify whether seed
parser.add_argument('--seed', default=0, type=int)

# CLIP
parser.add_argument('--model_name', default='ViT-L/14@336px', type=str)
parser.add_argument('--adapter_patch_size', default=32, type=int)
parser.add_argument('--adapter_prompt_dim', default=768, type=int)
parser.add_argument('--adapter_num_tokens', default=20, type=int)

# CelebA
parser.add_argument('--label_a', default=20, type=int)
parser.add_argument('--label_y', default=31, type=int)

# fair_reg
parser.add_argument('--fairness_notion', default='eqodds', type=str)
parser.add_argument('--lambda_fair_clip', default=None, type=float) # debias the visual feature
parser.add_argument('--lambda_fair_cls', default=None, type=float) # debias the classification


def average_weights(w_mmp):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w_mmp[0])
    for key in w_avg.keys():
        for i in range(1, len(w_mmp)):
            w_avg[key] += w_mmp[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w_mmp))
    return w_avg

def main():
    global args
    args = parser.parse_args()
    
    if args.dataset == 'celeba':
        if args.label_y == 2 and args.label_a == 20:
            far_mode = 'attractiveness_vs_gender'
        elif args.label_y == 31 and args.label_a == 20:
            far_mode = 'smiling_vs_gender'
        elif args.label_y == 39 and args.label_a == 20:
            far_mode = 'age_vs_gender'

    elif args.dataset == 'fairface':
        if args.label_y == 0 and args.label_a == 1:
            far_mode = 'age_vs_gender'
        
    save_dir = os.path.join(args.save_path, args.dataset, 'num_clients_' + str(args.num_users), 'adapter_num_tokens_' + str(args.adapter_num_tokens), 
                            far_mode, args.fairness_notion + '_clip_reg_' + str(args.lambda_fair_clip) + '_cls_reg_' + str(args.lambda_fair_cls), 'seed_' + str(args.seed))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving model to: ", save_dir)
    
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'celeba' or args.dataset == 'fairface':
        global_model, process = get_model_clip(args)
        
    else:
        exit("wrong dataset!!!")

    # load dataset and user groups
    user_groups, _ = get_dataset(args=args, process=process)

    global_model.cuda()
    global_model.train()
    global_model.float()

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    for epoch in range(args.epochs):
        local_weights = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, user_data=user_groups[idx], user_id=idx, process=process)
            adapter_w = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(adapter_w))
        
        # update global weights
        global_adapter_weights = average_weights(local_weights)

        # update global weights
        for name, param in global_weights.items():
            if name in global_adapter_weights.keys():
                global_weights[name] = global_adapter_weights[name]

        global_model.load_state_dict(global_weights)

    global_model_save_name = os.path.join(save_dir, 'model.checkpoint')
    adapter_state_dict = {}
    for name, param in global_model.state_dict().items():
        if ('adapter' in name and 'encoder' not in name) or ('classifier' in name):
            adapter_state_dict[name] = param

    global_model_checkpoint = {'state_dict': adapter_state_dict}
    torch.save(global_model_checkpoint, global_model_save_name)

if __name__ == "__main__":
    main()