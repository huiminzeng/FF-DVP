import pdb
import numpy as np
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import *
from core import fair_reg
from model import *

from eval_metrics import compute_accurate_metrics

class LocalUpdate(object):
    def __init__(self, args, user_data, user_id, process):
        self.args = args
        self.client_id = user_id
        self.process = process
        self.train_loader, self.val_loader = self.train_val_split(user_data)
    
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        optimizer = self.build_optimizer(model)
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='none')

        scaler = torch.cuda.amp.GradScaler()
        
        # gender axis
        gender_prompt = ["A photo of a male.", "A photo of a female."]
        gender_tokens = tokenize(gender_prompt).cuda()
        gender_axes = model.encode_text(gender_tokens)

        best_acc = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.train_loader):
                inputs, sensitive_attributes, targets = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                batch_size = inputs.shape[0]
                optimizer.zero_grad()
                contrastive_targets, prompts, gt_prompt = self.batch_stats(batch)
                contrastive_targets = contrastive_targets.long().cuda()
                token_prompts = tokenize(prompts).cuda()
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # forward pass of the adapted model
                    logits_per_classification, logits_per_image, logits_per_text, image_features, text_features = model(inputs, token_prompts)

                    # clip contrastive loss
                    loss_clip = (loss_img(logits_per_image, torch.arange(batch_size).cuda()) + loss_txt(logits_per_text, torch.arange(batch_size).cuda()))/2
                    
                    # debiasing visual feature
                    image_gender_text = image_features @ gender_axes.T
                    prob_text = F.softmax(100 * image_gender_text, dim=1)

                    target_probs = torch.tensor([0.5,0.5]).cuda().unsqueeze(0).repeat(batch_size, 1)
                    clip_reg = torch.mean(criterion_kl(torch.log(prob_text), target_probs).sum(dim=-1))

                    # classification loss
                    loss_ce = criterion_ce(logits_per_classification, targets)

                    # classification reg
                    cls_reg = fair_reg(self.args.fairness_notion, logits_per_classification, sensitive_attributes, targets)

                    loss =  loss_clip + self.args.lambda_fair_clip * clip_reg + \
                                    0.1 * (loss_ce + self.args.lambda_fair_cls * cls_reg)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_loss.append(loss.item())
                optimizer.zero_grad()

                # break

                if batch_idx % 30 == 0:
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            # Inference with classifier
                            token_prompts = tokenize(prompts).cuda()
                            logits_per_classification, _, _, _, _  = model(inputs, token_prompts)
                            predictions = torch.argmax(logits_per_classification, dim=-1)

                    acc = torch.mean((predictions==targets).float())
                    print('Global Round: {}, Local Epoch: {} [{}/{}], Train  Loss: {:.4f},  Reg: {:.4f},  Acc: {:.4f}'.format(
                        global_round, iter, batch_idx, len(self.train_loader), loss_clip.item(), clip_reg.item(), acc.item()))
                    
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            acc = self.validate(model)
            if acc > best_acc:
                best_acc = acc
                adapter_state_dict = {}
                for name, param in model.state_dict().items():
                    if ('adapter' in name and 'encoder' not in name) or ('classifier' in name):
                        adapter_state_dict[name] = copy.deepcopy(param)

        return adapter_state_dict
    
    def validate(self, model):
        """ Returns the validate accuracy and loss.
        """
        model.eval()
        
        targets_his = []
        predictions_his = []
        sensitive_attributes_his = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader): 
                inputs, sensitive_attributes, targets = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                contrastive_targets, prompts, gt_prompt = self.batch_stats(batch)
                token_prompts = tokenize(prompts).cuda()
                logits_per_classification, _, _, _, _ = model(inputs, token_prompts)

                predictions = torch.argmax(logits_per_classification, dim=-1) # convert to binary

                targets_his += targets.tolist()
                predictions_his += predictions.tolist()
                sensitive_attributes_his += sensitive_attributes.tolist()

        _, acc, _ = compute_accurate_metrics(np.array(predictions_his), np.array(sensitive_attributes_his), np.array(targets_his))

        print("Client ID: {}, Val Acc {:.3f}".format(self.client_id, acc))
        print("=" * 50)

        return acc

    def build_optimizer(self, model):
        trainable_parameters = 0
        all_parameters = 0
        params = [{'params': [p for (n, p) in model.named_parameters() if 'adapter' in n and 'encoder' not in n]}]
        for name, param in model.named_parameters():
            all_parameters += param.numel()
            if ('adapter' in name and 'encoder' not in name) or ('classifier' in name):
                param.requires_grad = True
                trainable_parameters += param.numel()
            else:
                param.requires_grad = False
        
        print("num trainable: {}, fraction: {:.4f}%".format(trainable_parameters, (trainable_parameters/all_parameters)*100))
        optimizer = torch.optim.Adam(params, lr=self.args.lr)

        return optimizer

    def train_val_split(self, user_data):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        inputs, sensitive_attributes, targets = user_data[0], user_data[1], user_data[2]
        num_samples = len(inputs)

        np.random.seed(0)
        idxs = np.random.choice(num_samples, num_samples, replace=False)

        idxs_train = idxs[:int(0.8*num_samples)]
        idxs_val = idxs[int(0.8*num_samples):]

        inputs_arr = np.array(inputs)
        sensitive_attributes_arr = np.array(sensitive_attributes)
        targets_arr = np.array(targets)

        if self.args.dataset == 'celeba' or self.args.dataset == 'fairface':
            train_dataset = myCeleba(data_path=inputs_arr[idxs_train], 
                                    sensitive_rows=sensitive_attributes_arr[idxs_train], 
                                    target_rows=targets_arr[idxs_train], 
                                    process=self.process)
            
            val_dataset = myCeleba(data_path=inputs_arr[idxs_val], 
                                    sensitive_rows=sensitive_attributes_arr[idxs_val], 
                                    target_rows=targets_arr[idxs_val], 
                                    process=self.process)
        
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)        
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)

        return train_loader, val_loader 

    def batch_stats(self, batch):
        sensitive_attributes, targets = batch[1], batch[2]
        batch_size = sensitive_attributes.shape[0]

        group_1 = (targets == 1) * (sensitive_attributes == 1)
        group_2 = (targets == 1) * (sensitive_attributes == 0)
        group_3 = (targets == 0) * (sensitive_attributes == 1)
        group_4 = (targets == 0) * (sensitive_attributes == 0)

        group_1_id = group_1 * 0
        group_2_id = group_2 * 1
        group_3_id = group_3 * 2
        group_4_id = group_4 * 3
        group_id_all = group_1_id + group_2_id + group_3_id + group_4_id

        biased_prompt, gt_prompt = self.get_text_prompts(group_id_all)
        
        prompts = []
        for i in range(batch_size):
            prompts.append(biased_prompt[i])

        return group_id_all, prompts, gt_prompt


    def get_text_prompts(self, group_id_all):
        if self.args.dataset == 'celeba':
            if self.args.label_y == 2 and self.args.label_a == 20:   # 2 for attractivness, 20 for gender
                biased_candidate_prompt = np.array(['A photo of a male, and he is attractive.',
                                                    'A photo of a female, and she is attractive.',
                                                    'A photo of a male, and he is not attractive.',
                                                    'A photo of a female, and she is not attractive.'])
                
            elif self.args.label_y == 31 and self.args.label_a == 20:    # 31 for smiling, 20 for gender
                biased_candidate_prompt = np.array(['A photo of a male, and he is smiling.',
                                                    'A photo of a female, and she is smiling.',
                                                    'A photo of a male, and he is not smiling.',
                                                    'A photo of a female, and she is not smiling.'])
                
            elif self.args.label_y == 39 and self.args.label_a == 20:  # 39 for age, 20 for gender
                biased_candidate_prompt = np.array(['A photo of a male, and he is young.',
                                                    'A photo of a female, and she is young.',
                                                    'A photo of a male, and he is not young.',
                                                    'A photo of a female, and she is not young.'])

        elif self.args.dataset == 'fairface':
            if self.args.label_y == 0 and self.args.label_a == 1: # 0 for age, 1 for gender
                biased_candidate_prompt = np.array(['A photo of a male, and he is young.',
                                                    'A photo of a female, and she is young',
                                                    'A photo of a male, and he is not young.',
                                                    'A photo of a female, and she is not young.'])
        
        return biased_candidate_prompt[group_id_all], biased_candidate_prompt

