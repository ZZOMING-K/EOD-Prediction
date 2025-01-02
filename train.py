#라이브러리 import 
import pandas as pd 
import numpy as np  
import seaborn as sns 
import matplotlib.pyplot as plt 

import random
import os 
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from sklearn.metrics import *
from sklearn.utils.class_weight import compute_class_weight # class weight 설정
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def make_features(df) : 
    df['term']  = [36 if x == 1 else 60 for x in df['term1']]
    df['tot_rec'] = df['funded_amnt'] * (df['term'] / 12) * df['int_rate'] # 총 이자 


    df['rec_rate'] = df['tot_rec'] / (df['total_rec_int']+1) # 총 이자 / 지금까지 납부된 이자
    df['rec_inc_rate'] = df['tot_rec'] / df['annual_inc'] # 총 이자 / 연 소득

    # 총 상환금액 / 지금까지 상환된 금액 
    df['fund_return'] = (df['installment'] * df['term']) / (df['funded_amnt'] - df['out_prncp'] + df['total_rec_int'] ) 

    # 월 상환액 대비 소득 
    df['month_inc_rate'] = (df['annual_inc'] / 12) / df['installment']
    # 리볼빙 잔액 / 현재 잔고 
    df['bal_rate'] = (df['revol_bal'] ) / (df['tot_cur_bal']+1)
    # 활성화 계좌 수 / 총 계좌 수
    df['total_open_acc'] = (df['open_acc'] ) / (df['total_acc']) 

    df['total_rec_late_group'] = df.apply(lambda x : 1 if( x['total_rec_late_fee'] > 0)  else 0 , axis = 1) 
    df['fico'] = (df['fico_range_low'] + df['fico_range_high']) / 2

    df['fico_int_rate'] = df['fico'] / df['int_rate'] 
    df['total_rec_int_rate'] = df['total_rec_int'] / df['int_rate']
    return df  

# 데이터 불러오기 
df = pd.read_csv('./data/train.csv')

df = make_features(df)

#행 /열 삭제 
df = df.drop([20327,28088] , axis=0).reset_index(drop = True) #이상치 처리 

drop_col = ['purpose4','funded_amnt_inv' , 
    'home_ownership1' , 'home_ownership3' , 'home_ownership4' , 'out_prncp_inv',
    fico_range_high , fico_range_low , tot_rec]

df = df.drop(drop_col , axis = 1)

# 데이터 분리
target = 'depvar'

x = df.drop(target , axis = 1)
y = df[target]

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size = 0.2 , random_state= 42 , stratify=y)


# 데이터 균형 맞추기
classes = classes = np.unique(y_train)
weights = compute_class_weight(class_weight = 'balanced' , classes = classes , y = y_train)
class_weights = dict(zip(classes , weights))
class_weights

#best params 
cat_best_params = {'iterations': 750, 
                    'learning_rate': 0.1703949292786075, 
                    'depth': 10, 
                    'l2_leaf_reg': 4.342950596496278,
                    'bagging_temperature': 0.469116738085111, 
                    'border_count': 241, 
                    'random_strength': 3.9953879909695225}

xgb_best_params = {'n_estimators': 703, 
                   'max_depth': 9, 
                   'learning_rate': 0.03948704517904488,
                   'subsample': 0.7281109724667661,
                   'colsample_bytree': 0.87443340115465, 
                   'gamma': 2.6378985452975794}

lgbm_best_params =  {'n_estimators': 797, 
                     'max_depth': 7, 
                     'learning_rate': 0.0366459821922061, 
                     'num_leaves': 77, 
                     'min_child_samples': 89,
                     'subsample': 0.5729914405334258, 
                     'colsample_bytree': 0.5172347974868119}

# Voting
voting_clf = VotingClassifier(
    estimators=[
        ('cat', cat_clf),
        ('xgb', xgb_clf),
        ('lgbm', lgbm_clf)
    ],
    voting='soft' 
)

voting_clf.fit(x_train, y_train)
voting_pred = voting_clf.predict(x_test)
voting_f1_macro = f1_score(y_test, voting_pred, average='macro')

print("\n=== Soft Voting Classifier ===")
print(classification_report(y_test, voting_pred))
print(f"Voting F1 Macro: {voting_f1_macro:.4f}")
