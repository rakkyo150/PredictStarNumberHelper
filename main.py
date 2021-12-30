import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
import requests
import io
import json

'''
githubEndpoint = "https://api.github.com/repos/rakkyo150/RankedMapData/releases/latest"
githubResponse = requests.get(url=githubEndpoint)
releaseJson = githubResponse.json()
secondHeaders = {'Accept': 'application/octet-stream'}
csvResponse = requests.get(url=releaseJson["assets"][0]["browser_download_url"],
                           headers=secondHeaders)
df = pd.read_csv(io.BytesIO(csvResponse.content), sep=",", index_col=0, encoding="utf-8")
'''

df=pd.read_csv("outcome.csv",index_col=0,encoding="utf-8")

print(df.describe())
print(df.columns)
print(df.shape)
print(df.index)

'''
fig=plt.figure()
axes=fig.subplots(1,2)
df.stars.plot.box(ax=axes[0])
df[df["stars"]>10].stars.plot.box(ax=axes[1])
print(plt.show())
'''

# 必要なカラムを選択
df=df[['bpm','duration','difficulty','sageScore','njs',
       'offset','notes','bombs','obstacles','nps','length','events','chroma','warns',
       'resets','stars']]

print(df.columns)

print(df.difficulty.unique())
print(df.difficulty.value_counts())

'''
print(df.createdAt.head())
df["createdAt"]=pd.to_datetime(df["createdAt"])
print(df.createdAt)
'''

for i in [ 'difficulty', 'chroma']:
    # 同じ値が同じ数字になり、違う値が違う数値になる
    df[i]=df[i].factorize()[0]
    print(df[i].head())

# 欠損値確認
print(df.isnull().any())
# 欠損値を含む行を削除
df=df.dropna(how='any')
# 欠損値再確認
print(df.isnull().any())

# bpm1000以上を除外
df=df[df['bpm']<1000]


# 箱ひげ図で外れ値の確認
plt.figure(tight_layout=True)
plt.rcParams['figure.subplot.bottom'] = 0.20
df.plot.box()
plt.xticks(rotation=90)
plt.savefig("dataframe")
plt.close()

normalizedDf = (df - df.mean()) / df.std()
plt.figure(tight_layout=True)
plt.rcParams['figure.subplot.bottom'] = 0.20
normalizedDf.plot.box()
plt.xticks(rotation=90)
plt.savefig("normalizeDataframe")
plt.close()

# データセット分割
t=df['stars'].values
x=df.drop(labels=['stars'],axis=1).values
columns=df.drop(labels=['stars'],axis=1).columns

x_train,x_test,t_train,t_test=train_test_split(x,t,test_size=0.3,random_state=0)

print(x_train.shape)
print(x_test.shape)
print(t_train.shape)
print(t_test.shape)

# 学習
model=LinearRegression()
model.fit(x_train,t_train)

print(model.coef_)
plt.figure(tight_layout=True)
plt.rcParams['figure.subplot.bottom'] = 0.15
plt.bar(x=columns,height=model.coef_)
plt.xticks(rotation=90)
plt.savefig("coef")
plt.close()

print(model.intercept_)

print(f'train score: {model.score(x_train,t_train)}')
print(f'test score: {model.score(x_test,t_test)}')

str={
    "trainScore":model.score(x_train,t_train),
    "testScore":model.score(x_test,t_test)
}
with open('modelScore.json',mode='w') as f:
    json.dump(str,f,indent=4)


# 多重共線性(入力変数同士の相関)の確認
print(df.corr())
plt.figure(tight_layout=True,figsize=[9.6,8])
sns.heatmap(df.corr(),annot=True)
plt.savefig("corr")
plt.close()

# 学習済みモデルを保存
with open('model.pickle', mode='wb') as f:
    pickle.dump(model,f,protocol=2)
