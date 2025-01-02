# 🏆 EOD Prediction: 대출 채무 불이행 예측

* 이어드림스쿨 4기 모의경진대회 | **2등** 🥈
* P2P 대부 업체의 고객 데이터를 통한  대출 채무 불이행 가능성을 예측하는 신뢰성 높은 머신러닝 모델 개발 
* 데이터 제공 : `AI CONNECT`
<br>


## 📌 목표 
* 고객 특성 데이터를 바탕으로 채무 불이행 가능성을 사전에 예측 
* 금융 리스크 관리 및 의사결정 프로세스 강화

<br>  

##  👀 프로세스

<img width="977" alt="image" src="https://github.com/user-attachments/assets/e6da00c4-2707-41af-9bbd-54e73a71e87b" />

<br>  

## 🔍 데이터 탐색 (EDA)

* 데이터 탐색(EDA)을 통해 데이터의 전반적인 구조와 특성을 이해
* 변수 간의 관계를 분석하여 예측 성능에 중요한 영향을 미치는 핵심 변수를 파악
* 효과적인 전처리 및 피처 엔지니어링 전략을 수립

<br>
  
1. 피처 ↔ 타겟 분석

* 범주형 피처
    * 시각화 : Barplot, Mosaic Plot
    * 통계 검정: 카이제곱 검정

* 수치형 피처
    * 시각화: Boxplot, Histogram, KDE Plot

2. 피처 ↔ 피처 분석
    * 시각화: Pairplot, Heatmap
    * 인사이트: 변수 간 상관관계 및 잠재적 패턴 확인.
<br>

## 🚀 모델선정 

베이스라인 모델 선정 → CatBoostClassifier 선정
* 별도의 전처리 없이 Boosting 계열 모델이 우수한 성능을 보임
<br>

|Model|F1 Score|
|------|---|
|Logisitic Regression|0.4036|
|RandomForest|0.6520|
|GradientBoosting|0.6548|
|**LightGBM**|**0.7004**|
|**XGBoost** |**0.6984**|
|**CatBoost**|**0.7070**|

<br>

## 🔧  성능향상 시도 

성능향상을 위해 다음과 같은 전략을 시도

① 파생변수 생성

② 불균형 데이터 처리 

* Oversampling , Cost-Sensitive Learning, Ensemble 기법 시도 
* 가장 성능이 높게 나온 `Cost-Sensitive Learning` 선택

③ 피처 선택

* feature importance를 기준으로 모델에 큰 영향을 미치지 않는 변수 제거 

④ 앙상블 및 하이퍼파라미터 튜닝
* optuna를 활용해 하이퍼 파라미터 튜닝 
* Stacking, Voting 앙상블 시도 

<br>


## ✅ 결과

* F1 Score(Macro) : 0.7070 → 0.7484
* 베이스라인 성능 대비 5.86 % 성능 향상
* SHAP 분석을 통해 대출 불이행 여부에 영향을 미치는 주요 변수 파악 및 해석 

<br>

## 🌟 주요 성과 및 기대효과

* 대출 채무 불이행 리스크를 사전에 예측하여 금융사의 리스크 관리 능력 향상.
* 신뢰성 높은 의사결정을 가능하게 하는 데이터 기반 인사이트 제공.

<br>

## 🛠 기술 스택

* 프로그래밍 언어: Python
* 라이브러리: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
