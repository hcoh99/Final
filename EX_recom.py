
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer

# === [1] 모델 구조 정의 ===
class MultiLabelNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === [2] 운동 환경 설정 ===
exercise_env_map = {
    "bench presses": "gym",
    "deadlifts": "gym",
    "overhead presses": "gym",
    "squats": "both",
    "cycling": "both",
    "running": "both",
    "brisk walking": "home",
    "dancing": "home",
    "yoga": "home",
    "walking": "home",
    "swimming": "exclude"  # 추천에서 제외
}

# === [3] 객체 및 모델 로딩 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = torch.load("exercise_scaler.pkl", map_location=device)
encoders = torch.load("exercise_encoders.pkl", map_location=device)
mlb = torch.load("exercise_mlb.pkl", map_location=device)

input_dim = 7
output_dim = len(mlb.classes_)

model = MultiLabelNet(input_dim, output_dim).to(device)
model.load_state_dict(torch.load("multilabel_exercise_model.pt", map_location=device))
model.eval()

# === [4] 사용자 입력 ===
user_input = {
    'Sex': 'Male',
    'Age': 23,
    'Height': 1.72,
    'Weight': 65,
    'Hypertension': 'No',
    'Fitness Goal': 'Weight Gain',
    'Level': 'Normal_Weight',
    'Has Gym Membership': 'yes'  # "yes" or "no"
}

# === [5] 인코딩 및 전처리 ===
for col in ['Sex', 'Hypertension', 'Fitness Goal', 'Level']:
    le = encoders[col]
    user_input[col] = le.transform([user_input[col]])[0] if user_input[col] in le.classes_ else 0

input_vector = [
    user_input['Sex'], user_input['Age'], user_input['Height'], user_input['Weight'],
    user_input['Hypertension'], user_input['Fitness Goal'], user_input['Level']
]
input_scaled = scaler.transform([input_vector])
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

# === [6] 예측 및 운동 필터링 ===
with torch.no_grad():
    output = model(input_tensor)
    threshold = 0.3  # 낮춰서 추천 더 많이 하도록
    predicted = (output.cpu().numpy() > threshold).astype(int)
    all_recommended = mlb.inverse_transform(predicted)[0] if predicted.any() else []

filtered_exercises = []
for ex in all_recommended:
    env = exercise_env_map.get(ex.lower(), "both")
    if env == "exclude":
        continue
    if user_input["Has Gym Membership"].lower() == "no" and env == "gym":
        continue
    filtered_exercises.append(ex)

# === [7] 출력 (최대 5개)
print("\n🏋️ 추천 운동 결과:")
if filtered_exercises:
    for i, ex in enumerate(filtered_exercises[:5], 1):  # ✅ 최대 5개
        print(f"{i}. {ex.capitalize()}")
else:
    print("❗ 추천된 운동이 없습니다.")