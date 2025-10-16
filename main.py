import io
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor

githubEndpoint = "https://api.github.com/repos/rakkyo150/RankedMapData/releases/latest"
headers = {'Authorization': f'token {os.environ["GITHUB_TOKEN"]}'}
githubResponse = requests.get(url=githubEndpoint, headers=headers)
# For local run, uncomment below code
# githubResponse = requests.get(url=githubEndpoint)
releaseJson = githubResponse.json()
secondHeaders = {'Accept': 'application/octet-stream'}
csvResponse = requests.get(url=releaseJson["assets"][0]["browser_download_url"],
                           headers=secondHeaders)
df = pd.read_csv(io.BytesIO(csvResponse.content), sep=",", index_col=0, encoding="utf-8")


# 必要なカラムを選択
df = df[['bpm', 'duration', 'difficulty', 'sageScore', 'njs',
         'offset', 'notes', 'bombs', 'obstacles', 'nps', 'events', 'chroma', 'errors', 'warns',
         'resets', 'stars']]

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
df = df[df['bpm'] < 1000]

# 欠損値を含む行を削除
df = df.dropna(how='any')

# 同じ値が同じ数字になり、違う値が違う数値になる
print(df['chroma'].value_counts())
df['chroma'] = df['chroma'].factorize(sort=True)[0]
print(df['chroma'].value_counts())

print(df['difficulty'].value_counts())
df.loc[df['difficulty'] == "Easy", 'difficulty'] = 0
df.loc[df['difficulty'] == "Normal", 'difficulty'] = 1
df.loc[df['difficulty'] == "Hard", 'difficulty'] = 2
df.loc[df['difficulty'] == "Expert", 'difficulty'] = 3
df.loc[df['difficulty'] == "ExpertPlus", 'difficulty'] = 4
print(df['difficulty'].value_counts())

# 型がobjectになっているので
df['difficulty'] = df['difficulty'].astype('float64')

describe = df.describe()
print(describe)

print(df.dtypes)

with open('./describe.json', mode='w') as f:
    describe.to_json(f, indent=4)

# データセット分割
t = df['stars'].values
x = df.drop(labels=['stars'], axis=1).values
columns = df.drop(labels=['stars'], axis=1).columns

x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.2, random_state=0)

print(x_test.shape)

# model=MLPRegressor(random_state=0,max_iter=10000)
# train score: 0.9634951347836535
# test score: 0.9373156381840932

# model=LinearRegression()
# "trainScore": 0.8958441039360735
# "testScore": 0.9074964858127781

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[("standard_scaler",StandardScaler()),
                       ("estimator", MLPRegressor(random_state=0, max_iter=1000, solver="adam",
                        activation="relu", verbose=True))])
from sklearn import set_config
set_config(display='diagram')
print(pipe)

param_grid = [{
    'estimator__solver': ['sgd'],
    'estimator__max_iter': [1000],
    'estimator__hidden_layer_sizes': [1500],
}]
cv = 5
tuned_model = GridSearchCV(estimator=pipe,
                           param_grid=param_grid,
                           cv=cv,
                           return_train_score=False)

tuned_model.fit(x_train_val,t_train_val)

print(tuned_model.best_params_)
model = tuned_model.best_estimator_

train_pred = model.predict(x_train_val)
pred = model.predict(x_test)

# old bad way
# train RMSE: 0.5417871476868293
# test RMSE: 0.8000724981830256
# train score: 0.9683632363282518
# test score: 0.9332618120022189

print(f'train RMSE: {np.sqrt(mean_squared_error(t_train_val, train_pred))}')
print(f'test RMSE: {np.sqrt(mean_squared_error(t_test, pred))}')
print(f'train score: {model.score(x_train_val, t_train_val)}')
print(f'test score: {model.score(x_test, t_test)}')

# new appropriate way
# train RMSE: 0.6312606845960649
# test RMSE: 0.7419870097808028
# train score: 0.9570510908896043
# test score: 0.9426004707446569

for i in range(t_test.shape[0]):
    c = t_test[i]
    p = pred[i]
    print('[{0}] correct:{1:.3f}, predict:{2:.3f} ({3:.3f})'.format(i, c, p, c - p))

model_evaluation = {
    'Train RMSE': np.sqrt(mean_squared_error(t_train_val, train_pred)),
    'Test RMSE': np.sqrt(mean_squared_error(t_test, pred)),
    "Train Score": model.score(x_train_val, t_train_val),
    "Test Score": model.score(x_test, t_test)
}
with open('./modelEvaluation.json', mode='w') as f:
    json.dump(model_evaluation, f, indent=4)

# 多重共線性(入力変数同士の相関)の確認
plt.figure(tight_layout=True, figsize=[9.6, 8])
sns.heatmap(df.corr(), annot=True)
plt.savefig("./correlation.png")
plt.close()

# 学習済みモデルを保存
with open('./model.pickle', mode='wb') as f:
    pickle.dump(model, f)

import image_maker
image_maker.json_to_png_for_describe('describe.json')
image_maker.json_to_png_for_evaluation('modelEvaluation.json')

with open('./model.pickle', mode='rb') as f:
    model = pickle.load(f)

# Start to convert pickle into ONNX format
from skl2onnx import to_onnx
onx = to_onnx(model, np.array(x_train_val)[1:])

with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

import onnxruntime as rt
sess = rt.InferenceSession("model.onnx")

"""
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = sess.get_inputs()[0].type
print("Input type  :", input_type)
output_name = sess.get_outputs()[0].name
print("Output name  :", output_name)
output_shape = sess.get_outputs()[0].shape
print("Output shape :", output_shape)
output_type = sess.get_outputs()[0].type
print("Output type  :", output_type)
"""

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(x_train_val[0])
convert_test_data = [x_train_val[0]]
print("Test input : " + ", ".join(map(str, convert_test_data)))
pickle_result = model.predict(convert_test_data)
print("Pickle output : " + ", ".join(map(str, pickle_result)))
onnx_result = sess.run([output_name], {input_name: convert_test_data})
print("Onnx output : " + ", ".join(map(str, onnx_result[0])))