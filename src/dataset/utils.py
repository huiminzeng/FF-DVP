import numpy as np

def split_clients(train_data_path, train_labels, label_y, label_a, num_users):
    num_items = len(train_data_path) // num_users
    dict_users = {}
    
    target_feature = train_labels[:, label_y]
    sensitive_feature = train_labels[:, label_a]
    
    group_1 = (target_feature == 1) * (sensitive_feature == 1)
    group_2 = (target_feature == 1) * (sensitive_feature == 0)
    group_3 = (target_feature == 0) * (sensitive_feature == 1)
    group_4 = (target_feature == 0) * (sensitive_feature == 0)

    group_1_idxs = np.where(group_1==1)[0]
    group_2_idxs = np.where(group_2==1)[0]
    group_3_idxs = np.where(group_3==1)[0]
    group_4_idxs = np.where(group_4==1)[0]

    group_idx = [group_1_idxs, group_2_idxs, group_3_idxs, group_4_idxs]
    group_idxs_num = [len(group_idx[0]), len(group_idx[1]), len(group_idx[2]), len(group_idx[3])]

    for i in range(num_users):
        # set a random seed for each user to determine the distribution
        np.random.seed(i)
        group_prob = np.random.choice(range(1,11), 4, replace=False)
        group_prob = group_prob / group_prob.sum()
        user_data = []
        user_sensitive = []
        user_label = []

        for j in range(4):
            group_j_prob = group_prob[j]
            num_samples = int(num_items * group_j_prob)

            if num_samples > len(group_idx[j]):
                num_samples = len(group_idx[j]) // 4
            
            group_j_idx_selected = set(np.random.choice(group_idx[j], num_samples, replace=False))
            group_j_idx = set(group_idx[j]) - group_j_idx_selected
            group_idx[j] = list(group_j_idx)
        
            group_j_data = train_data_path[list(group_j_idx_selected)]
            group_j_sensitive = sensitive_feature[list(group_j_idx_selected)]
            group_j_target = target_feature[list(group_j_idx_selected)]

            user_data += list(group_j_data)
            user_sensitive += list(group_j_sensitive)
            user_label += list(group_j_target)
        
        group_idxs_num = [len(group_idx[0]), len(group_idx[1]), len(group_idx[2]), len(group_idx[3])]

        while True:
            if len(user_data) < num_items:
                remaining_items = num_items - len(user_data)
                max_remaining = group_idxs_num.index(max(group_idxs_num))

                if remaining_items <= len(group_idx[max_remaining]):
                    additional_idx_selected = set(np.random.choice(group_idx[max_remaining], remaining_items, replace=False))
                
                else:
                    additional_idx_selected = set(np.random.choice(group_idx[max_remaining], len(group_idx[max_remaining]), replace=False))

                additional_idx_remaining = set(group_idx[max_remaining]) - additional_idx_selected
                group_idx[max_remaining] = list(additional_idx_remaining)

                additional_data = train_data_path[list(additional_idx_selected)]
                additional_sensitive = sensitive_feature[list(additional_idx_selected)]
                additional_label = target_feature[list(additional_idx_selected)]

                user_data += list(additional_data)
                user_sensitive += list(additional_sensitive)
                user_label += list(additional_label)
                
                group_idxs_num = [len(group_idx[0]), len(group_idx[1]), len(group_idx[2]), len(group_idx[3])]

            else:
                break

        dict_users[i] = [user_data, user_sensitive, user_label]
        
    return dict_users

def group_split(data_label_arr, label_y, label_a, num_samples):
    pos_male = (data_label_arr[:,label_y] == 1) * (data_label_arr[:,label_a] == 1)
    pos_female = (data_label_arr[:,label_y] == 1) * (data_label_arr[:,label_a] == 0)

    neg_male = (data_label_arr[:,label_y] == 0) * (data_label_arr[:,label_a] == 1)
    neg_female = (data_label_arr[:,label_y] == 0) * (data_label_arr[:,label_a] == 0)

    pos_male_id = np.where(pos_male==1)[0]
    pos_female_id = np.where(pos_female==1)[0]
    neg_male_id = np.where(neg_male==1)[0]
    neg_female_id = np.where(neg_female==1)[0]

    # data_distribution specifies the distribution 4 groups:
    num_pos_male = int(0.25 * num_samples)
    num_pos_female = int(0.25 * num_samples)                              
    num_neg_male = int(0.25 * num_samples)
    num_neg_female = int(0.25 * num_samples)

    train_pos_male_id = list(pos_male_id[:num_pos_male])
    train_pos_female_id = list(pos_female_id[:num_pos_female])
    train_neg_male_id = list(neg_male_id[:num_neg_male])
    train_neg_female_id = list(neg_female_id[:num_neg_female])

    test_pos_male_id = list(pos_male_id[num_pos_male : num_pos_male + num_pos_male // 4])
    test_pos_female_id = list(pos_female_id[num_pos_female : num_pos_female + num_pos_female // 4])
    test_neg_male_id = list(neg_male_id[num_neg_male : num_neg_male + num_neg_male // 4])
    test_neg_female_id = list(neg_female_id[num_neg_female : num_neg_female + num_neg_female // 4])

    train_ids = train_pos_male_id + train_pos_female_id + train_neg_male_id + train_neg_female_id
    test_ids = test_pos_male_id + test_pos_female_id + test_neg_male_id + test_neg_female_id

    return train_ids, test_ids
