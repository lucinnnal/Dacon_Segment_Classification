from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import *

# 랜덤 포레스트 모델 초기화
rf = RandomForestClassifier(random_state=42)

# 하이퍼파라미터 그리드 정의
param_grid = {
    'n_estimators': [100, 200, 300],          # 트리의 개수
    'max_depth': [10, 20, 30],                # 트리의 최대 깊이
    'min_samples_split': [2, 5, 10],           # 노드 분할 최소 샘플 수
    'min_samples_leaf': [1, 2, 4],             # 리프 노드 최소 샘플 수
    'max_features': ['auto', 'sqrt', 'log2']  # 분할에 사용할 최대 특성 수
}

# 그리드 서치 객체 생성
model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# 모델 학습
model.fit(X, y_encoded)

# 최적의 하이퍼파라미터 출력
print("최적 하이퍼파라미터:", model.best_params_)

# row-level 예측 수행
y_test_pred = model.best_estimator_(X_test)
# 예측 결과를 변환
y_test_pred_labels = le_target.inverse_transform(y_test_pred)

# row 단위 예측 결과를 test_data에 추가
test_data = test_df.copy()  # 원본 유지
test_data["pred_label"] = y_test_pred_labels

submission = test_data.groupby("ID")["pred_label"] \
    .agg(lambda x: x.value_counts().idxmax()) \
    .reset_index()

submission.columns = ["ID", "Segment"]

submission.to_csv('./base_submit.csv',index=False)