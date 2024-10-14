import os
import copy
import argparse
import numpy as np

import torch

from model import *
from dataset import *

from eval_metrics import compute_accurate_metrics, compute_fairness_metrics

parser = argparse.ArgumentParser()

# dataset arguments
parser.add_argument('--data_path', type=str, default=None, help="path \
                    to dataset")
parser.add_argument('--dataset', type=str, default=None, help="name \
                    of dataset")

# federated arguments (Notation for the arguments followed from paper)
parser.add_argument('--epochs', type=int, default=5,
                    help="number of rounds of training")
parser.add_argument('--batch_size', type=int, default=16,
                    help="local batch size: B")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_users', type=int, default=100,
                    help="number of users: K")
parser.add_argument('--local_ep', type=int, default=5,
                    help="the number of local epochs: E")

parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='SGD weight decay (default: 1d-4)')

# other arguments
parser.add_argument('--load_path', default='trained_models/baselines', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

# specify whether seed
parser.add_argument('--seed', default=0, type=int)

# CLIP
parser.add_argument('--model_name', default='ViT-L/14@336px', type=str)
parser.add_argument('--adapter_patch_size', default=32, type=int)
parser.add_argument('--adapter_prompt_dim', default=768, type=int)
parser.add_argument('--adapter_num_tokens', default=5, type=int)

# CelebA
parser.add_argument('--label_a', default=20, type=int)
parser.add_argument('--label_y', default=31, type=int)

# fair_reg
parser.add_argument('--fairness_notion', default=None, type=str)
parser.add_argument('--lambda_fair_clip', default=None, type=float) # debias the visual feature
parser.add_argument('--lambda_fair_cls', default=None, type=float) # debias the classification


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

    load_dir = os.path.join(args.load_path, args.dataset, 'num_clients_' + str(args.num_users), 'adapter_num_tokens_' + str(args.adapter_num_tokens), 
                            far_mode, args.fairness_notion + '_clip_reg_' + str(args.lambda_fair_clip) + '_cls_reg_' + str(args.lambda_fair_cls))
    
    print("we are load model from: ", load_dir)

    load_name_list = []
    for i in range(len(os.listdir(load_dir))): # this is hard-coded
        seeds_name = 'seed_' + str(i)
        load_name_temp = os.path.join(load_dir, seeds_name, 'model.checkpoint')
        load_name_list.append(load_name_temp)

    avg_acc_his = []
    global_acc_his = []
    acc_std_his = []
    demo_parity_his = []
    eq_odds_his = []

    # load dataset and user groups
    global_model, process = get_model_clip(args)
    del global_model
    _, test_dataset = get_dataset(args=args, process=process)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=args.num_workers, 
                                            pin_memory=True)
    
    for load_name in load_name_list:
        if args.dataset == 'celeba' or args.dataset == 'fairface':
            global_model, process = get_model_clip(args)
            
        else:
            exit("wrong dataset!!!")
            
        if os.path.isfile(load_name):
            print("=> loading checkpoint '{}'".format(load_name))
            checkpoint = torch.load(load_name)
            
            # load adpated weights
            adapted_state_dict = checkpoint['state_dict']
            own_state = global_model.state_dict()
            for name, param in adapted_state_dict.items():
                own_state[name] = copy.deepcopy(param.data)
            global_model.load_state_dict(own_state)
            global_model.eval()
            global_model.cuda()
            global_model.float()
            print("=> loaded checkpoint !! ")

        else:
            exit("No model found!")
                
        prompts = get_text_prompts(args)
        token_prompts_gt = tokenize(prompts).cuda()

        targets_his = []
        predictions_his = []
        sensitive_attributes_his = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                inputs, sensitive_attributes, targets = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

                # Inference
                image_features = global_model.encode_image(inputs)
                text_features = global_model.encode_text(token_prompts_gt)

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                logit_scale = global_model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                predictions = torch.argmax(logits_per_image, dim=-1) <= 1 # convert to binary

                targets_his += targets.tolist()
                predictions_his += predictions.tolist()
                sensitive_attributes_his += sensitive_attributes.tolist()
                
        acc_mean, acc, acc_std = compute_accurate_metrics(np.array(predictions_his), np.array(sensitive_attributes_his), np.array(targets_his))
        demo_parity, eq_odds = compute_fairness_metrics(np.array(predictions_his), np.array(sensitive_attributes_his), np.array(targets_his))
        
        avg_acc_his.append(acc_mean)
        global_acc_his.append(acc)
        acc_std_his.append(acc_std)
        demo_parity_his.append(demo_parity)
        eq_odds_his.append(eq_odds)

        del global_model

    avg_acc_mean = np.mean(avg_acc_his)
    acc_gap_mean = np.mean(acc_std_his)
    demo_parity_mean = np.mean(demo_parity_his)
    eq_odds_mean = np.mean(eq_odds_his)

    avg_acc_std = np.std(avg_acc_his)
    acc_gap_std = np.std(acc_std_his)
    demo_parity_std = np.std(demo_parity_his)
    eq_odds_std = np.std(eq_odds_his)

    print("avg acc, mean: {:.3f}, std: {:.3f}".format(avg_acc_mean, avg_acc_std))
    print("acc gap, mean: {:.3f}, std: {:.3f}".format(acc_gap_mean, acc_gap_std))
    print("demo parity, mean: {:.3f}, std: {:.3f}".format(demo_parity_mean, demo_parity_std))
    print("eq odds, mean: {:.3f}, std: {:.3f}".format(eq_odds_mean, eq_odds_std))

def get_text_prompts(args):
    if args.dataset == 'celeba':
        if args.label_y == 2 and args.label_a == 20:   # 2 for attractivness, 20 for gender
            biased_candidate_prompt = np.array(['A photo of a male, and he is attractive.',
                                                'A photo of a female, and she is attractive.',
                                                'A photo of a male, and he is not attractive.',
                                                'A photo of a female, and she is not attractive.'])
            
        elif args.label_y == 31 and args.label_a == 20:    # 31 for smiling, 20 for gender
            biased_candidate_prompt = np.array(['A photo of a male, and he is smiling.',
                                                'A photo of a female, and she is smiling.',
                                                'A photo of a male, and he is not smiling.',
                                                'A photo of a female, and she is not smiling.'])

        elif args.label_y == 39 and args.label_a == 20:  # 39 for age, 20 for gender
            biased_candidate_prompt = np.array(['A photo of a male, and he is young.',
                                                'A photo of a female, and she is young.',
                                                'A photo of a male, and he is not young.',
                                                'A photo of a female, and she is not young.'])

    elif args.dataset == 'fairface':
        if args.label_y == 0 and args.label_a == 1: # 0 for age, 1 for gender
            biased_candidate_prompt = np.array(['A photo of a male, and he is young.',
                                                'A photo of a female, and she is young.',
                                                'A photo of a male, and he is not young.',
                                                'A photo of a female, and she is not young.'])
        
    return biased_candidate_prompt

if __name__ == "__main__":
    main()