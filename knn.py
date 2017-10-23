from numpy import *
import operator

def create_data_set():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify_0(in_x, data_set, labels, k):
    data_size = data_set.shape[0]

    # Euclidian distances calculation
    diff_mat = tile(in_x, (data_size, 1)) - data_set
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5

    # sort distances
    sorted_distances = distances.argsort()

    # vote k nearest
    class_count={}
    for i in range(k):
        vote_label = labels[sorted_distances[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]
