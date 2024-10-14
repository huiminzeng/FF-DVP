import os
import glob
import pandas as pd
import numpy as np
    
from .utils import *
from .datasets import myCeleba

def get_dataset(args, process=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    dataset = args.dataset
    num_users = args.num_users
    
    if dataset == 'celeba':
        label_a = args.label_a
        label_y = args.label_y
        data_dir = os.path.join(args.data_path, args.dataset, "img_align_celeba")

        # specifying the zip file name 
        data_path = sorted(glob.glob(data_dir + '/*.jpg'))
        data_path = sorted(data_path, key=lambda x:x[-10:])

        # get the label of images
        label_path = os.path.join(args.data_path, dataset, "list_attr_celeba.txt")

        file_rows = open(label_path).readlines()
        training_targeted_attribute = file_rows[1].split(' ')[label_y]
        print("performing classification w.r.t. ", training_targeted_attribute)
        label_list = file_rows[2:]
            
        data_label = []
        for i in range(len(label_list)):
            data_label.append(label_list[i].split())

        # transform label into 0 and 1
        for m in range(len(data_label)):
            data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
            data_label[m] = [int(p) for p in data_label[m]]
        
        data_path_arr = np.array(data_path)
        data_label_arr = np.array(data_label)
    
    elif dataset == 'fairface':
        label_a = args.label_a
        label_y = args.label_y
        
        label_file = pd.read_csv(os.path.join(args.data_path, dataset, 'fairface_label_train.csv')).values
        input_id = []
        label_data = []
        for i in range(len(label_file)):
            current_row = label_file[i]
            image_id = os.path.join(args.data_path, dataset, current_row[0])
            age = current_row[1]
            gender = current_row[2]

            if age in ['20-29', '30-39']:
                age_num = 1
            elif age in ['40-49', '50-59', '60-69', 'more than 70']:
                age_num = 0
            else:
                continue

            if gender == 'Male':
                gender_num = 1
            else:
                gender_num = 0
            
            input_id.append(image_id)
            label_data.append([age_num, gender_num])

        data_path_arr = np.array(input_id)
        data_label_arr = np.array(label_data)

    train_ids, test_ids = group_split(data_label_arr, label_y, label_a, 20000)
    train_data_path = data_path_arr[train_ids]  # split into train/val on clients
    train_labels = data_label_arr[train_ids]    # split into train/val on clients
    
    user_groups = split_clients(train_data_path, train_labels, label_y, label_a, num_users)

    test_data_path = data_path_arr[test_ids]
    test_labels = data_label_arr[test_ids]
    test_dataset = myCeleba(data_path=test_data_path, 
                            sensitive_rows=test_labels[:, label_a], 
                            target_rows=test_labels[:, label_y], 
                            process=process)

    return user_groups, test_dataset