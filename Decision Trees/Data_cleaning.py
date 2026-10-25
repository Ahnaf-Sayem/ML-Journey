import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\j\Downloads\ML-Journey\Decision Trees\loan_200_people.csv').drop(columns = ['person'])

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

y_train = np.array(df_train['default']).reshape(-1,1)
Y_test = np.array(df_test['default'])
y_val = np.array(df_val['default'])

df_train = df_train.drop(columns = 'default')
df_val = df_val.drop(columns = 'default')
df_test = df_test.drop(columns = 'default')

X_train = np.array(df_train)



