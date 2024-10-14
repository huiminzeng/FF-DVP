import numpy as np

def group_split(data_label_arr, label_y, label_a):
    pos_male = (data_label_arr[:,label_y] == 1) * (data_label_arr[:,label_a] == 1)
    pos_female = (data_label_arr[:,label_y] == 1) * (data_label_arr[:,label_a] == 0)

    neg_male = (data_label_arr[:,label_y] == 0) * (data_label_arr[:,label_a] == 1)
    neg_female = (data_label_arr[:,label_y] == 0) * (data_label_arr[:,label_a] == 0)

    pos_male_id = np.where(pos_male==1)[0]
    pos_female_id = np.where(pos_female==1)[0]
    neg_male_id = np.where(neg_male==1)[0]
    neg_female_id = np.where(neg_female==1)[0]
    
    num_pos_male = len(pos_male_id)
    num_pos_female = len(pos_female_id)
    num_neg_male = len(neg_male_id)
    num_neg_female = len(neg_female_id)
    
    train_pos_male_id = list(pos_male_id[:int(num_pos_male * 0.8)])
    train_pos_female_id = list(pos_female_id[:int(num_pos_female*0.8)])
    train_neg_male_id = list(neg_male_id[:int(num_neg_male * 0.8)])
    train_neg_female_id = list(neg_female_id[:int(num_neg_female*0.8)])

    test_pos_male_id = list(pos_male_id[int(num_pos_male * 0.8):])
    test_pos_female_id = list(pos_female_id[int(num_pos_female*0.8):])
    test_neg_male_id = list(neg_male_id[int(num_neg_male * 0.8):])
    test_neg_female_id = list(neg_female_id[int(num_neg_female*0.8):])

    train_ids = train_pos_male_id + train_pos_female_id + train_neg_male_id + train_neg_female_id
    test_ids = test_pos_male_id + test_pos_female_id + test_neg_male_id + test_neg_female_id

    return train_ids, test_ids


def Gaussian_stats(user_groups):
    num_users = len(user_groups)
    
    num_data_list = []
    mean_list = []
    std_list = []
    for i in range(num_users):
        user_data = np.array(user_groups[i][0])
        
        mean = np.mean(user_data, axis=0)
        std = np.std(user_data, axis=0)
        
        num_data = len(user_data)

        num_data_list.append(num_data)
        mean_list.append(mean)
        std_list.append(std)
    
    num_data_list = np.array(num_data_list).reshape(num_users, 1)
    mean_list = np.array(mean_list)
    std_list = np.array(std_list)

    weighted_mean = np.sum(mean_list * num_data_list, axis=0) / np.sum(num_data_list)
    weighted_std = np.sum(std_list * num_data_list, axis=0) / np.sum(num_data_list)

    return weighted_mean, weighted_std