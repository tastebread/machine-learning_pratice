import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact
import time
from matplotlib import rc
import plotly.express as px
import scipy.stats as stats
import mplcyberpunk
from pandas_profiling import ProfileReport
from scipy.stats.contingency import association
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import warnings
plt.style.use('cyberpunk')#시각화 테마
pd.set_option('display.max_columns',None) ##모든 열을 출력한다
warnings.filterwarnings(action='ignore') #경고문구 삭제
train = pd.read_csv('train[house].csv')
test = pd.read_csv('test[house].csv')
submission = pd.read_csv('sample_submission[house].csv')


df_num = train[['GrLivArea','LotArea','SalePrice']]
df_t = train[['GrLivArea']]
#만일 예측을 한다고 가정했을때 가장 간단하고 직관적인 방법으로 평균이나 중간값을 이용해 보는것도 좋은 선택이다
#기준모델(baseline) : 예측모델을 구체적으로 만들기 전에 간단하면서도 직관적으로 최소한의 성능을 나타내는 기준을 기준모델이라고 한다

#분류문제에서의 기준 모델을 만드는 방법 - > 타겟의 최빈 클래스
#회귀문제 -> 타겟의 평균값
#시계열회귀문제 : 이전 타임스탬프 값

predict = train['SalePrice'].mean()
errors = predict - train['SalePrice']

print(errors)

#MAE 구하기
mean_absolute_error = errors.abs().mean() #abs -> 절대값 취하기
print(mean_absolute_error)

#기준모델 그려보기
x = train['GrLivArea']
y = train['SalePrice']

#sns.lineplot(x=x,y=predict, color='red')
#sns.scatterplot(x=x,y=y, color='blue')
#plt.show()
print(f'예측한 주택 가격 ${predict:,.0f} 이며 절대평균 에러가 ${mean_absolute_error:,.0f}')

#cols = ['GrLivArea','LotArea','SalePrice']
#sns.pairplot(train[cols], height=2)
#plt.savefig('SalcePrice.png')

#예측모델 활용해보기
"""
회귀분석에서 중요한 개념은 예측값과 잔차 이다 
예측값 : 만들어진 모델이 추정하는 값
잔차 : 예측값과 관측값 차이 (오차(error)는 모집단에서의 예측값과 관측값 차이를 말합니다)
회귀선은 잔차 제곱들의 합인(RSS)를 최소화 하는 직선 으로 이 값이 회귀모델의 비용함수가 된다
머신러닝에서는 이렇게 비용함수를 최소화 하는 모델을 찾는 과정을 학습!! 이라고 한다
OLS : 잔차제곱합을 최소화 하는 방법
"""

#seaborn regplot 으로 그려보자

#sns.regplot(x=train['GrLivArea'], y=train['SalePrice'])
#plt.savefig('GrLivArea-SalePrice regplot.png')

#먼저 GrLivArea 가 3500~4500 사이의 데이터를 알아봄
print(train[(train['GrLivArea'] > 3500) & (train['GrLivArea'] < 4500)])

"""
그래프를 보면 3500~4500 까지는 주택에 대한 가격 정보가별로 없다
이때는 선형모델을 사용해서 어림잡아 예측이 가능함
"""

#만약 기존 데이터의 범위를 넘어서는 값을 예측하고 싶을땐 어떻게 해야할까?
"""
선형회귀 직선은 독립변수(x) 와 종속변수(y) 간의 관계를 요약해준다
종속변수 : 반응변수,레이블,타겟등으로 불림
독립변수 : 예측변수,설명,특성,등으로 불림
"""

#사이킷런을 사용해서 선형회귀 모델을 만들어 보기
"""
먼저 모델을 만들고 데이터 분석하기 위해서는 다음과 같은 데이터 구조가 필요함
1.특성데이터와 타겟 데이터를 나눠준다
2.특성행렬은 주로 x로 표현하고 보통 2차원 행렬로 numpy,pandas로 만들어줌 
3.타겟배열은 주로 y로 표현하고 보통 1차원 형태이다 numpy,pandas.Series 로 표현한다


우선 여러분께서 풀어야 하는 문제를 풀기에 적합한 모델을 선택하여 클래스를 찾아본 후 
관련 속성이나 하이퍼파라미터를 확인해 봅니다.
문제에 따라서 약간 차이가 있겠지만 위에서 살펴본 것과 같이 데이터를 준비합니다.
fit() 메소드를 사용하여 모델을 학습합니다.
'predict()' 메소드를 사용하여 새로운 데이터를 예측합니다.
"""

#단순선형회귀모델
model = LinearRegression()

feature = ['GrLivArea']
target = ['SalePrice']
X_train = train[feature]
y_train = train[target]

#모델 학습
model.fit(X_train,y_train)

#새로운 데이터 샘플을 하나 선택해서 모델을 통해 예측해보기
# X_test = [[7000]]
# y_pred = model.predict(X_test)

# print(f'4000~7000 GrLivArea를 가지는 주택의 예상가격은 {int(y_pred)}$ 입니다')

#전체 테스트 데이터를 이용해서 모델을 통해 예측해보기
X_test = [[x] for x in df_t['GrLivArea']]
y_pred = model.predict(X_test)

#이제 train 데이터와 ,test 데이터에 대한 예측을 파란점으로 표현해보기
# plt.scatter(X_train,y_train,color='black',linewidth=1)
# plt.scatter(X_test,y_pred, color='blue',linewidth=1)
# plt.savefig('train-test dataset predict.png')

"""
비슷하게 나오고 있음을 시각화를 통해 알수 있다
궁금한점 : 모델이 주택의 크기와 가격 사이에서 어떤 관계를 학습했을까?
질문의 답을 하기위해 coef_,intercept 속성을 확인해봐야함
"""

#선형회귀모델의 계수(coef_)
print(model.coef_)

#절편(intercept)
print(model.intercept_)

#예측함수를 하나 만들어보고 coef 의 영향을 알아보자
def makingpredict(good):
    y_pred = model.predict([[good]])
    pred = f'{int(good)} 주택 가격 예측 : {int(y_pred)}$ (1 당 추가금 : {int(model.coef_[0])})'

    return pred

print(makingpredict(6000))