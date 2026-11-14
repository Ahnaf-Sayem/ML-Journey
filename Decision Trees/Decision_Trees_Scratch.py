from collections import  Counter

import numpy as np


class Node:
    def __init__(self, feature_no = None, threshold = None, left = None, right = None, *,  value = None):
        self.feature_no = feature_no
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self):
        return (self.value is not None)

class Decision_Trees:
    def __init__(self, min_sample_split = 30, depth = 0, ):
        self.min_sample_split = min_sample_split
        self.depth = depth

    def fit(self,x_train, y_train):
        self.x_train = x_train # X and Y has to be 2d
        self.y_train = y_train
        self.root = self.grow(x_train, y_train)# work to do
    def grow(self, x, y, depth = 0):
        #stopping criteria
        self.x = x
        self.y = y
        unique_labels = len(np.unique(self.y))

        if (depth  >= self.depth or len(x[:,0]) < self.min_sample_split or  unique_labels == 1) :
            return Node(value = self.most_common_y_value(self.y) ) # work to do
        features = [ _ for _ in range(len(x[0,:]))]
        best_feature, threshold = self.best_split(features, self.x, self.y)
        left_idx, right_idx = self.split(self.x,best_feature, threshold)
        left = self.grow(x[left_idx, :], y[left_idx, :], depth + 1)
        right = self.grow(x[right_idx, :], y[right_idx, :], depth + 1)
        return Node(best_feature, threshold, left, right)

    def split(self,x,best_feature, threshold):
        left_row_idx = np.argwhere(x[:, best_feature] > threshold)
        left_row_idx_flattened = left_row_idx.flatten()
        right_row_idx = np.argwhere(x[:, best_feature] > threshold)
        right_row_idx_flattened = right_row_idx.flatten()
        return left_row_idx_flattened, right_row_idx_flattened

    def info_gain(self ,x, y, feature, threshold):
        # Finding before messiness
        len_p0 = len(np.argwhere( y[:,0] == 0))
        len_p1 = len(y) - len_p0
        p0 = len_p0 / len(y)
        p1 = len_p1 / len(y)
        before_messiness = 1 - (p0 ** 2 + p1 ** 2)


        # After messiness
        'working on left side only here .'
        left_row_idx = np.argwhere(x[:, feature] <= threshold)
        left_row_idx_flattened = left_row_idx.flatten()
        left_side_x = x[left_row_idx_flattened, :]
        left_side_y = y[left_row_idx_flattened, :]
        len_left_side_p0 = len(np.argwhere(left_side_y == 0))
        len_left_side_p1 = len(left_side_y) - len_left_side_p0
        left_side_po = len_left_side_p0 / len(left_side_y)
        left_side_p1 = len_left_side_p1 / len(left_side_y)
        left_side_messiness = 1 - (left_side_po ** 2 + left_side_p1 ** 2)

        'working on right side only here'
        right_row_idx = np.argwhere(x[:, feature] > threshold)
        right_row_idx_flattened = right_row_idx.flatten()
        right_side_x = x[right_row_idx_flattened, :]
        right_side_y = y[right_row_idx_flattened, :]
        len_right_side_p0 = len(np.argwhere(right_side_y == 0))
        len_right_side_p1 = len(right_side_y) - len_right_side_p0
        right_side_p0 = len_right_side_p0/ len(right_side_y)
        right_side_p1 = len_right_side_p1 / len(right_side_y)
        right_side_messiness  = 1 - (right_side_p0 ** 2 + right_side_p1 ** 2)

        after_messiness = ((len(left_side_y) / len(y)) * left_side_messiness) + ((len(right_side_y) / len(y)) * right_side_messiness)
        'figuring out net gain'
        info_gain = before_messiness - after_messiness
        return info_gain
    def best_split(self, features, x, y):
        thresholds = np.unique(x)
        gain = -1
        best_feature = 'nothing'
        best_threshold = 'nothing'
        for feature in features:
            for threshold in thresholds:
                current_gain = self.info_gain(x, y, feature, threshold)
                if current_gain >  gain :
                    best_feature = feature
                    best_threshold = threshold
                    gain = current_gain
        return best_feature, best_threshold

    def most_common_y_value(self,y):
        counter_obj = Counter(y.ravel())
        return counter_obj.most_common(1)[0][0]
    def predict(self,x_data):
        self.x_data = x_data # it has to be a 2d numpy array too

        pass
    def traverse_tree(self):
        pass
