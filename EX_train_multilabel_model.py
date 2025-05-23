# === multilabel_model_train.py ===
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
import re
import pickle

# === 1. 데이터 로딩 ===
df = pd.read_csv("gym_for_recommendation.csv")

# === 2. Exercises 분리 ===
def split_exercises(text):
    return [e.strip().lower() for e in re.split(r',| and | or ', str(text)) if e.strip()]
df['Exercise_List'] = df['Exercises'].apply(split_exercises)

# === 3. Feature 및 Label 인코딩 ===
feature_cols = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Fitness Goal', 'Level']
label_col = 'Exercise_List'

encoders = {}
for col in ['Sex', 'Hypertension', 'Fitness Goal', 'Level']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df[label_col])

# === 4. PyTorch Dataset 정의 ===
class ExerciseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = ExerciseDataset(X_scaled, Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# === 5. 모델 정의 ===
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

input_dim = X.shape[1]
output_dim = Y.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiLabelNet(input_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# === 6. 학습 ===
for epoch in range(30):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# === 7. 저장 ===
torch.save(model.state_dict(), "multilabel_exercise_model.pt")
torch.save(scaler, "exercise_scaler.pkl")
torch.save(encoders, "exercise_encoders.pkl")
torch.save(mlb, "exercise_mlb.pkl")