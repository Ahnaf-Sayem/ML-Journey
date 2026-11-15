from sklearn.metrics import roc_auc_score
from Decision_Trees_Scratch import Decision_Tree
from Data_cleaning import x_train, x_val, y_train, y_val,x_test,y_test

dt = Decision_Tree()
dt.fit(x_train,y_train)

train_probs = dt.predict(x_train).flatten()
val_probs = dt.predict(x_val).flatten()
test_probs = dt.predict(x_test).flatten()

y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()

train_auc = roc_auc_score(y_train, train_probs)
val_auc = roc_auc_score(y_val, val_probs)
test_auc = roc_auc_score(y_test, test_probs)

print(f"Train AUC: {train_auc}")
print(f"Val AUC: {val_auc}")
print(f"Test AUC: {test_auc}")
