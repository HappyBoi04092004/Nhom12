import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import json

print("Đang đọc dữ liệu...")
# Đọc dữ liệu
#df = pd.read_csv('Exam_Score_Prediction.csv')
#df = pd.read_csv('data/Exam_Score_Prediction.csv')
df = pd.read_csv('../data/Exam_Score_Prediction.csv')



# Chọn các features (đầu vào)
features = ['TestScore_Reading', 'TestScore_Science', 'GPA', 'StudyHours', 'AttendanceRate']
X = df[features].values

# Chọn target (điểm cần dự đoán) - giả sử là TestScore_Math
target = 'TestScore_Math'
y = df[target].values

print(f"\nFeatures: {features}")
print(f"Target: {target}")
print(f"Shape X: {X.shape}, Shape y: {y.shape}")

# Chia dữ liệu thành train và test
print("\nĐang chia dữ liệu train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train model
print("\nĐang train model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên test set
print("Đang đánh giá model...")
y_pred = model.predict(X_test)

# Tính metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== KẾT QUẢ ===")
print(f"R² Score: {r2:.6f}")
print(f"RMSE: {rmse:.6f}")

# Lưu model
print("\nĐang lưu model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Lưu metrics
metrics = {
    'R2': float(r2),
    'RMSE': float(rmse)
}

with open('model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print(" Đã lưu model.pkl")
print(" Đã lưu model_metrics.json")
print("\nHoàn thành!")

