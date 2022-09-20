import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import pickle
import requests
import io
import json
import os


githubEndpoint = "https://api.github.com/repos/rakkyo150/RankedMapData/releases/latest"
headers={'Authorization': f'token {os.environ["GITHUB_TOKEN"]}'}
githubResponse = requests.get(url=githubEndpoint,headers=headers)
releaseJson = githubResponse.json()
secondHeaders = {'Accept': 'application/octet-stream'}
csvResponse = requests.get(url=releaseJson["assets"][0]["browser_download_url"],
                           headers=secondHeaders)
df = pd.read_csv(io.BytesIO(csvResponse.content), sep=",", index_col=0, encoding="utf-8")

# For local run
# df=pd.read_csv("outcome.csv",index_col=0,encoding="utf-8")

# 必要なカラムを選択
df=df[['bpm','duration','difficulty','sageScore','njs',
       'offset','notes','bombs','obstacles','nps','events','chroma','errors','warns',
       'resets','stars']]

'''
# 箱ひげ図で外れ値の確認
plt.figure(tight_layout=True)
plt.rcParams['figure.subplot.bottom'] = 0.20
df.plot.box()
plt.xticks(rotation=90)
plt.savefig("./box.png")
plt.close()
'''

# bpm1000以上を除外
df=df[df['bpm']<1000]

# 欠損値を含む行を削除
df=df.dropna(how='any')

# 同じ値が同じ数字になり、違う値が違う数値になる
print(df['chroma'].value_counts())
df['chroma']=df['chroma'].factorize(sort=True)[0]
print(df['chroma'].value_counts())

print(df['difficulty'].value_counts())
df.loc[df['difficulty']=="Easy",'difficulty']=0
df.loc[df['difficulty']=="Normal",'difficulty']=1
df.loc[df['difficulty']=="Hard",'difficulty']=2
df.loc[df['difficulty']=="Expert",'difficulty']=3
df.loc[df['difficulty']=="ExpertPlus",'difficulty']=4
print(df['difficulty'].value_counts())

# 型がobjectになっているので
df['difficulty'] = df['difficulty'].astype('float64')

describe = df.describe()
print(describe)

print(df.dtypes)

with open('./describe.json',mode='w') as f:
    describe.to_json(f,indent=4)

# データセット分割
t=df['stars'].values
x=df.drop(labels=['stars'],axis=1).values
columns=df.drop(labels=['stars'],axis=1).columns

x_train_val,x_test,t_train_val,t_test=train_test_split(x,t,test_size=0.2,random_state=0)

print(x_test.shape)

# 標準化
from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()
normalized_x_train_val=standardScaler.fit_transform(x_train_val)

print(standardScaler.mean_)
print(standardScaler.var_)
normalized_x_test=(x_test-standardScaler.mean_)/np.sqrt(standardScaler.var_)

with open('./standardScaler.pickle', mode='wb') as f:
    pickle.dump(standardScaler, f)

# 学習
# model=MLPRegressor(random_state=0,max_iter=10000)
# train score: 0.9634951347836535
# test score: 0.9373156381840932

# model=LinearRegression()
# "trainScore": 0.8958441039360735
# "testScore": 0.9074964858127781


estimator=MLPRegressor(random_state=0,max_iter=1000,solver="adam",activation="relu",verbose=True)
param_grid=[{
    'solver':['sgd'],
    'max_iter':[1000],
    'hidden_layer_sizes':[1500],
}]
cv=5
tuned_model=GridSearchCV(estimator=estimator,
                         param_grid=param_grid,
                         cv=cv,
                         return_train_score=False)

tuned_model.fit(normalized_x_train_val,t_train_val)

print(tuned_model.best_params_)
model=tuned_model.best_estimator_


train_pred=model.predict(normalized_x_train_val)
pred = model.predict(normalized_x_test)

# old bad way
# train RMSE: 0.5417871476868293
# test RMSE: 0.8000724981830256
# train score: 0.9683632363282518
# test score: 0.9332618120022189

print(f'train RMSE: {np.sqrt(mean_squared_error(t_train_val,train_pred))}')
print(f'test RMSE: {np.sqrt(mean_squared_error(t_test, pred))}')
print(f'train score: {model.score(normalized_x_train_val,t_train_val)}')
print(f'test score: {model.score(normalized_x_test,t_test)}')

# new appropriate way
# train RMSE: 0.6312606845960649
# test RMSE: 0.7419870097808028
# train score: 0.9570510908896043
# test score: 0.9426004707446569

for i in range(t_test.shape[0]):
    c = t_test[i]
    p = pred[i]
    print('[{0}] correct:{1:.3f}, predict:{2:.3f} ({3:.3f})'.format(i, c, p, c-p))

str={
    'train RMSE': np.sqrt(mean_squared_error(t_train_val,train_pred)),
    'test RMSE': np.sqrt(mean_squared_error(t_test, pred)),
    "trainScore":model.score(x_train_val,t_train_val),
    "testScore":model.score(x_test,t_test)
}
with open('./modelEvaluation.json',mode='w') as f:
    json.dump(str,f,indent=4)


# 多重共線性(入力変数同士の相関)の確認
plt.figure(tight_layout=True,figsize=[9.6,8])
sns.heatmap(df.corr(),annot=True)
plt.savefig("./correlation.png")
plt.close()

# 学習済みモデルを保存
with open('./model.pickle', mode='wb') as f:
    pickle.dump(model,f)