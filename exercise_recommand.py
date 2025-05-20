
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# === [1] 파일 경로 설정 ===
gym_path = "/Users/ohheungchan/workspace/univ/[2025-1]FinalProject/gym_recommendation.xlsx"
exercise_meta_path = "/Users/ohheungchan/workspace/univ/[2025-1]FinalProject/megaGymDataset.csv"
exercise_location_path = "/Users/ohheungchan/workspace/univ/[2025-1]FinalProject/exercise_location_labels.csv"

# === [2] 데이터 불러오기 ===
df_gym = pd.read_excel(gym_path)
df_meta = pd.read_csv(exercise_meta_path, encoding='utf-8')
df_location = pd.read_csv(exercise_location_path, encoding='utf-8')

# === [3] 레이블 인코딩 ===
label_cols = ['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal', 'Fitness Type', 'Level']
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df_gym[col] = le.fit_transform(df_gym[col])
    encoders[col] = le

# === [4] 입력 피처 스케일링 ===
feature_cols = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']
X = df_gym[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === [5] KNN 모델 학습 ===
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_scaled)

# === [6] 사용자 입력 ===
new_user = {
    'Sex': 'Male',
    'Age': 25,
    'Height': 1.55,
    'Weight': 25,
    'Hypertension': 'No',
    'Diabetes': 'No',
    'BMI': 24.5,
    'Level': 'Normal',
    'Fitness Goal': 'Weight Loss',
    'Fitness Type': 'Cardio',
    'Workout Environment': 'Home'
}

# === [7] 사용자 전처리 ===
new_input = pd.DataFrame([new_user])
for col in label_cols:
    le = encoders[col]
    value = new_input[col].values[0]
    if value in le.classes_:
        new_input[col] = le.transform([value])[0]
    else:
        print(f"⚠️ '{value}' is not in {col} label classes. Setting to -1.")
        new_input[col] = -1  # Unknown 클래스 처리
new_input_scaled = scaler.transform(new_input[feature_cols])

# === [8] 유사 사용자 탐색 ===
distances, indices = knn.kneighbors(new_input_scaled)

# === [9] 홈 운동 여부 확인 함수 ===
def is_home_friendly(exercise_name):
    row = df_location[df_location['Exercise'].str.strip().str.lower() == exercise_name.strip().lower()]
    if row.empty:
        return True  # 없으면 기본적으로 홈에서 가능하다고 가정
    return 'home' in row.iloc[0]['Location'].strip().lower()

# === [10] 유사 사용자 운동 수집 ===
exercise_list = []
for i in indices[0]:
    raw = str(df_gym.loc[i, 'Exercises'])
    split_exercises = raw.replace(' and ', ',').replace(' or ', ',').split(',')
    for ex in split_exercises:
        ex = ex.strip()
        if not ex:
            continue
        if new_user['Workout Environment'].lower() == 'home' and not is_home_friendly(ex):
            continue
        exercise_list.append(ex)

# === [11] 운동 빈도 기반 추천 Top 5 ===
exercise_counts = Counter(exercise_list)
top_exercises = [ex for ex, _ in exercise_counts.most_common(5)]

# === [12] 출력 ===
print("\n🏋️ 최종 추천 운동 Top 5:")
for idx, ex in enumerate(top_exercises, 1):
    print(f"{idx}. {ex}")