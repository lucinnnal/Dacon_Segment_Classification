from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils import *

# 랜덤 포레스트 모델 초기화
# XGBoost 모델 초기화
model = XGBClassifier(n_estimators=200, max_depth=20, learning_rate=0.1, min_child_weight=2, colsample_bytree=0.8, subsample=0.8, objective='multi:softmax', num_class=len(le_target.classes_))

# 모델 학습
print("Train Start")
model.fit(X, y_encoded)

# row-level 예측 수행
print("Inferencing...")
y_test_pred = model.predict(X_test)
# 예측 결과를 변환
y_test_pred_labels = le_target.inverse_transform(y_test_pred)

print("Start recording answers")
# row 단위 예측 결과를 test_data에 추가
test_data = test_df.copy()  # 원본 유지
test_data["pred_label"] = y_test_pred_labels

submission = test_data.groupby("ID")["pred_label"] \
    .agg(lambda x: x.value_counts().idxmax()) \
    .reset_index()

submission.columns = ["ID", "Segment"]

submission.to_csv('./base_submit.csv',index=False)

# model save
import joblib

joblib.dump(model, "random_forest_model.pkl")