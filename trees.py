from math import log
import operator

def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feature_vector in data_set:
        current_label = feature_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def create_data_set():
    data_set = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feature_vector in data_set:
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis+1:])
            ret_data_set.append(reduced_feature_vector)
    return ret_data_set

def choose_best_feature(data_set):
    num_features = len(data_set[0]) - 1;
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0; best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        print(i)
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i , value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
            print(sub_data_set)
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.key():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_count(class_list)

    best_feature = choose_best_feature(data_set)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in data_set]
    unique_vals = set(feature_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(
                split_data_set(data_set, best_feature, value),
                sub_labels)
    return my_tree
