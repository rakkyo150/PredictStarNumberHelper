import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
       'offset','notes','bombs','obstacles','nps','events','chroma','warns',
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


meanStd=pd.DataFrame({'mean':df.mean(),'std':df.std()})
print(meanStd)
meanStd.to_csv("meanStd.csv")

# 標準化
df = (df - df.mean()) / df.std()

# データセット分割
t=df['stars'].values
x=df.drop(labels=['stars'],axis=1).values
columns=df.drop(labels=['stars'],axis=1).columns

x_train_val,x_test,t_train_val,t_test=train_test_split(x,t,test_size=0.2,random_state=0)
x_train,x_val,t_train,t_val=train_test_split(x_train_val,t_train_val,test_size=0.3,random_state=0)

print(x_test.shape)

# 学習
# model=MLPRegressor(random_state=0,max_iter=10000)
# train score: 0.9634951347836535
# test score: 0.9373156381840932

# model=LinearRegression()
# "trainScore": 0.8958441039360735
# "testScore": 0.9074964858127781


estimator=MLPRegressor(random_state=0,max_iter=10000,solver="adam",activation="relu")
param_grid=[{
    'hidden_layer_sizes':[590,600,610],
}]
cv=5
tuned_model=GridSearchCV(estimator=estimator,
                         param_grid=param_grid,
                         cv=cv,
                         return_train_score=False)

tuned_model.fit(x_train_val,t_train_val)

print(tuned_model.best_params_)
model=tuned_model.best_estimator_


# print(f'train score: {model.score(x_train,t_train)}')
print(f'train score: {model.score(x_train_val,t_train_val)}')
print(f'test score: {model.score(x_test,t_test)}')

pred = model.predict(x_test)
for i in range(x_test.shape[0]):
    c = t_test[i]
    p = pred[i]
    print('[{0}] correct:{1:.3f}, predict:{2:.3f} ({3:.3f})'.format(i, c, p, c-p))



str={
    "trainScore":model.score(x_train_val,t_train_val),
    "testScore":model.score(x_test,t_test)
}
with open('./modelScore.json',mode='w') as f:
    json.dump(str,f,indent=4)


# 多重共線性(入力変数同士の相関)の確認
plt.figure(tight_layout=True,figsize=[9.6,8])
sns.heatmap(df.corr(),annot=True)
plt.savefig("./correlation.png")
plt.close()

# 学習済みモデルを保存
with open('./model.pickle', mode='wb') as f:
    pickle.dump(model,f)