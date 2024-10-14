import torch

def fair_reg(fairness_notion, preds, a, y):
    probs = torch.softmax(preds, dim=-1) # [batch_size, 2]
    mask = probs[:, 1] >= 0.5
    
    if fairness_notion == "dp":
        # |P(y_hat=1|a=0) - P(y_hat=1|a=1)|
        pred_dis = torch.sum((probs[:, 1] * mask) * (a==1)) / (torch.sum((a==1)) + 1e-6) \
                    - torch.sum((probs[:, 1] * mask) * (a==0)) / (torch.sum((a==0)) + 1e-6)
    
        reg = torch.abs(pred_dis)

    elif fairness_notion == "eqodds":
        # |P(y_hat=1|y=1,a=0) - P(y_hat=1|y=1,a=1)| + |P(y_hat=1|y=0,a=0) - P(y_hat=1|y=0,a=1)|
        pred_dis_1 = torch.sum(probs[:, 1] * a * y) / (torch.sum(a * y) + 1e-6) \
                    - torch.sum(probs[:, 1] * (1 - a) * y) / (torch.sum((1 - a) * y) + 1e-6)
        
        pred_dis_2 = torch.sum(probs[:, 1] * a * (1 - y)) / (torch.sum(a * (1 - y)) + 1e-6) + \
                    - torch.sum(probs[:, 1] * (1 - a) * (1 - y)) / (torch.sum((1 - a) * (1 - y)) + 1e-6)
        
        pred_dis = torch.abs(pred_dis_1) + torch.abs(pred_dis_2)
        reg = pred_dis
        
    return reg