# Fair Federated Learning with Biased Vision-Language Models

This repository is the PyTorch impelementation for the [paper](https://aclanthology.org/2024.findings-acl.595.pdf) "Fair Federated Learning with Biased Vision-Language Models".

<img src=media/pipeline.jpg width=800>

We propose a Fair Federated Deep Visiual Prompting (FF-DVP) framework for fine-tuning CLIP for federated applications in a fairness-aware fashion. As implied by its name, FF-DVP trains a fair FL model with fairness-aware deep visual prompting (DVP). Moreover, FF-DVP incorporates modality-fused classification heads to learn client-specific knowledge and fairness constraints. These modules explicitly address a unique kind of bias in FL, namely the bias triggered by data heterogeneity.

## Requirements

For our running environment see requirements.txt

## Datasets
- We used CelebA and FairFace dataset. After download them, please modify the training arugments `--data_path` and `--dataset` accordingly.
- Example folder structure
```
    ├── ...
    ├── data                   
    │   ├── celeba 
    │   └── fairface
    ├── src
    │   └── ...
    └── ...
```
## Scripts.

- Training:
   - Example
       ```
       python3 src/train.py --data_path --dataset --label_a --label_y --num_users --lambda_fair_cls --lambda_fair_clip;
       ```
   - Hyperparameters
      ```
      --data_path               # path to datasets
      --dataset                 # select from 'celeba', 'fairface'
      --label_a                 # demographic attribute (20 for gender)
      --label_y                 # predictive attribute
      --num_users               # number of federated clients
      --lambda_fair_cls         # lambda_2 for fairness regularizer on local classification heads
      --lambda_fair_clip        # lambda_1 for debiasing CLIP visual representations
      ```
    - After executing the script, the trained models will be automatically saved to a new folder `trained_models`

- Evaluation: 
    - Example
       ```
       python3 src/eval.py --data_path --dataset --label_a --label_y --num_users --lambda_fair_cls --lambda_fair_clip;

       ```
    - After executing the script, the evaluation metrics will be printed.

## Acknowledgement

During the implementation, we base our code mostly on existing repos (e.g., DVP). Many thanks to these authors for their great work!
