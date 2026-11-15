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

class Decision_Tree:
    def __init__(self, min_sample_split = 50, depth = 15, ):
        self.min_sample_split = min_sample_split
        self.depth = depth

    def fit(self,x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.root = self.grow(x_train, y_train)

    def grow(self, x, y, depth = 0):
        #stopping criteria
        unique_labels = len(np.unique(y))

        if (depth  >= self.depth or len(x) < self.min_sample_split or  unique_labels == 1) :
            return Node(value = self.most_common_y_value(y))

        features = [ _ for _ in range(len(x[0, :]))]
        best_feature, threshold = self.best_split(features, x, y)
        left_idx, right_idx = self.split(x,best_feature, threshold)
        left = self.grow(x[left_idx, :], y[left_idx, :], depth + 1)
        right = self.grow(x[right_idx, :], y[right_idx, :], depth + 1)
        return Node(best_feature, threshold, left, right)

    def split(self,x,best_feature, threshold):
        left_row_idx = np.argwhere(x[:, best_feature] <= threshold)
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
        'working on left side messiness only here'
        left_row_idx = np.argwhere(x[:, feature] <= threshold)
        left_row_idx_flattened = left_row_idx.flatten()
        if len(left_row_idx_flattened) == 0:
            return  -1 #Basically it is gonna skip when it fails to divide people since thres = max puts all of them in one side
        left_side_x = x[left_row_idx_flattened, :]
        left_side_y = y[left_row_idx_flattened, :]
        len_left_side_p0 = len(np.argwhere(left_side_y == 0))
        len_left_side_p1 = len(left_side_y) - len_left_side_p0
        left_side_po = len_left_side_p0 / len(left_side_y)
        left_side_p1 = len_left_side_p1 / len(left_side_y)
        left_side_messiness = 1 - (left_side_po ** 2 + left_side_p1 ** 2)

        'working on right side messiness only here'
        right_row_idx = np.argwhere(x[:, feature] > threshold)
        right_row_idx_flattened = right_row_idx.flatten()
        if len(right_row_idx_flattened) == 0:
            return -1 #Basically it is gonna skip when it fails to divide people since thres = max puts all of them in one side
        right_side_x = x[right_row_idx_flattened, :]
        right_side_y = y[right_row_idx_flattened, :]
        len_right_side_p0 = len(np.argwhere(right_side_y == 0))
        len_right_side_p1 = len(right_side_y) - len_right_side_p0
        right_side_p0 = len_right_side_p0/ len(right_side_y)
        right_side_p1 = len_right_side_p1 / len(right_side_y)
        right_side_messiness  = 1 - (right_side_p0 ** 2 + right_side_p1 ** 2)

        # After messiness
        after_messiness = ((len(left_side_y) / len(y)) * left_side_messiness) + ((len(right_side_y) / len(y)) * right_side_messiness)

        'Figuring out net gain'
        info_gain = before_messiness - after_messiness
        return info_gain

    def best_split(self, features, x, y):
        gain = -1
        best_feature = 'nothing'
        best_threshold = 'nothing'

        for feature in features:
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                current_gain = self.info_gain(x, y, feature, threshold)
                if current_gain >  gain :
                    best_feature = feature
                    best_threshold = threshold
                    gain = current_gain
        return best_feature, best_threshold

    def most_common_y_value(self,y):
        obj = np.array([float(np.mean(y))])
        obj.reshape( -1, 1)
        return obj

    def predict(self,x_data):
        self.x_data = x_data
        y_pred = [ ]
        for each_row in self.x_data:
            if len(y_pred) == 0:
                y_pred = self.traverse_tree(each_row.reshape(1, self.x_data.shape[1]), self.root)
            else:
                y_pred = np.vstack([y_pred,self.traverse_tree(each_row.reshape(1, self.x_data.shape[1]), self.root)])
        return y_pred

    def traverse_tree(self,x,node):
        if  node.is_leaf():
            return node.value
        elif x[:,node.feature_no] <= node.threshold:
            return self.traverse_tree(x,node.left)
        elif x[:,node.feature_no] > node.threshold:
            return self.traverse_tree(x, node.right)


