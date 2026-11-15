import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
csv_path = Path(__file__).parent / 'loan_2000_people.csv'
df = pd.read_csv(csv_path).drop(columns=['person'])


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

y_train = np.array(df_train['default']).reshape(-1, 1)
y_test = np.array(df_test['default']).reshape(-1, 1)
y_val = np.array(df_val['default']).reshape(-1, 1)

df_train = df_train.drop(columns = 'default')
df_val = df_val.drop(columns = 'default')
df_test = df_test.drop(columns = 'default')

x_train = np.array(df_train)
x_val = np.array(df_val)
x_test = np.array(df_test)
