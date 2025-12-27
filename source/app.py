from flask import Flask, render_template, request, jsonify
import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

# Đường dẫn đến file metrics
METRICS_FILE = 'model_metrics.json'
MODEL_FILE = 'model.pkl'

def load_metrics():
    """Load R2 và RMSE từ file"""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Giá trị mặc định nếu chưa có file
        return {
            'R2': 0.0,
            'RMSE': 0.0
        }

def load_model():
    """Load model đã được train từ file"""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' không tồn tại. "
            "Vui lòng chạy train_model.py để train model trước."
        )

@app.route('/')
def index():
    """Trang chủ hiển thị form nhập liệu"""
    metrics = load_metrics()
    return render_template('index.html', r2=metrics['R2'], rmse=metrics['RMSE'])

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để nhận dữ liệu và trả về dự đoán"""
    try:
        # Lấy dữ liệu từ request
        data = request.json
        
        # Kiểm tra các trường bắt buộc
        required_fields = ['TestScore_Reading', 'TestScore_Science', 'GPA', 'StudyHours', 'AttendanceRate']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Thiếu trường: {field}'}), 400
        
        # Chuyển đổi dữ liệu thành numpy array
        features = np.array([[
            float(data['TestScore_Reading']),
            float(data['TestScore_Science']),
            float(data['GPA']),
            float(data['StudyHours']),
            float(data['AttendanceRate'])
        ]])
        
        # Load model và dự đoán
        model = load_model()
        prediction = model.predict(features)[0]
        
        # Đảm bảo prediction không âm (điểm số thường >= 0)
        prediction = max(0, prediction)
        
        # Load metrics
        metrics = load_metrics()
        
        return jsonify({
            'prediction': round(float(prediction), 2),
            'target': 'TestScore_Math',  # Thông tin về điểm được dự đoán
            'r2': round(float(metrics['R2']), 6),
            'rmse': round(float(metrics['RMSE']), 6)
        })
        
    except ValueError as e:
        return jsonify({'error': f'Dữ liệu không hợp lệ: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Lỗi khi dự đoán: {str(e)}'}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """API endpoint để lấy metrics"""
    metrics = load_metrics()
    return jsonify(metrics)

if __name__ == '__main__':
    # Kiểm tra model có tồn tại không
    if not os.path.exists(MODEL_FILE):
        print(f"  Cảnh báo: File '{MODEL_FILE}' không tồn tại.")
        print("   Vui lòng chạy 'python train_model.py' để train model trước.")
        print("   Ứng dụng vẫn sẽ chạy nhưng sẽ báo lỗi khi dự đoán.")
    else:
        print(f" Đã tìm thấy model: {MODEL_FILE}")
    
    # Kiểm tra metrics
    if os.path.exists(METRICS_FILE):
        metrics = load_metrics()
        print(f" Metrics: R² = {metrics['R2']:.6f}, RMSE = {metrics['RMSE']:.6f}")
    else:
        print(f" Cảnh báo: File '{METRICS_FILE}' không tồn tại.")
    
    print("\n Khởi động server Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)

