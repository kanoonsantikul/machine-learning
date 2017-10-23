import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(node_text, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(
            node_text,
            xy=parent_pt,
            xycoords='axes fraction',
            xytext=center_pt,
            textcoords='axes fraction',
            va="center",
            ha="center",
            bbox=node_type,
            arrowprops=arrow_args)

def plot_mid_text(cntr_pt, parent_pt, text_string):
    x_mid = (parent_pt[0]-cntr_pt[0])/2.0 + cntr_pt[0]
    y_mid = (parent_pt[1]-cntr_pt[1])/2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, text_string)

def plot_tree(my_tree, parent_pt, node_text):
    num_leafs = get_num_leafs(my_tree)
    get_tree_depth(my_tree)
    first_string = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)

    plot_mid_text(cntr_pt, parent_pt, node_text)
    plot_node(first_string, cntr_pt, parent_pt, decision_node)

    second_dict = my_tree[first_string]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d

    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0/ plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d

def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w; plot_tree.y_off = 1.0;
    plot_tree(in_tree, (0.5,1.0), '')
    plt.show()

def get_num_leafs(my_tree):
    num_leafs = 0
    first_string = list(my_tree.keys())[0]
    second_dict = my_tree[first_string]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs +=1
    return num_leafs

def get_tree_depth(my_tree):
    max_depth = 0
    first_string = list(my_tree.keys())[0]
    second_dict = my_tree[first_string]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
            {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return list_of_trees[i]
