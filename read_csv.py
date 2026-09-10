import csv, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

def convert_float(array):
    for i, num in enumerate(array):
        array[i] = float(num)
    return array
    # return [float(num) for num in array]

def convert_binary(text_arr, value_1 = 'Yes'):
    for i, text in enumerate(text_arr):
        if text == value_1:
            text_arr[i] = 1
        else:
            text_arr[i] = 0
    return text_arr
    # return 1 if text == value_1 else 0
    # return [1 if text == value_1 else 0 for text in text_arr]
    # return int(text == value_1)

def convert_category(txt, cat_list):
    result = [0 for cat in cat_list]
    if txt in cat_list:
        result[cat_list.index(txt)] = 1
    return result
    # return [1 if cat == txt else 0 for cat in cat_list]

class SimpleNN(nn.Module):
    def __init__(self, input_dim=26, output_dim=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 輸入特徵 26 個，輸出 64 個神經元
        self.norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)  # 中間層：輸出 32 個結果
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)  # 輸出 1 個結果

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

with open('datasets/diabetes_risk_prediction_dataset.csv', 'r') as f:
    reader = list(csv.reader(f))
    data = []
    label = []
    for row in reader:
        if row[0] == 'Patient_ID':
            continue
        if '' not in row:
            patient_data = convert_float([row[1], row[4], row[5], row[7], row[9], row[11], row[14], row[15], row[16], row[17], row[20], row[21]])
            patient_data.extend(convert_binary([row[28], row[29], row[30], row[31], row[32]]))
            patient_data.extend(convert_binary([row[35]], 'Urban'))
            patient_data.extend(convert_category(row[2], ['Male', 'Female']))
            patient_data.extend(convert_category(row[26], ['Current', 'Former']))
            patient_data.extend(convert_category(row[27], ['Frequently', 'Occasionally']))
            patient_data.extend(convert_category(row[33], ['Good', 'Average']))
            data.append(patient_data)
            label.append(int(row[37]))

continuous_features = [row[:12] for row in data]
binary_features = [row[12:] for row in data]
continuous_scaled = StandardScaler().fit_transform(continuous_features)
# print(continuous_scaled[0:4])
# print(data[-10:-6])
# print(label[-10:-6])
# exit()
tensor_data = torch.tensor(data[:-50], dtype=torch.float32)
# tensor_data = torch.cat([torch.tensor(continuous_scaled[:-50], dtype=torch.float32), torch.tensor(binary_features[:-50], dtype=torch.float32)], dim=1)
tensor_label = torch.tensor(label[:-50], dtype=torch.int64).view(-1, 1)
# torch.set_printoptions(precision=1, sci_mode=False)

# 區分 training, validatino, testing set
dataset = TensorDataset(tensor_data, tensor_label)
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [int(0.8*len(dataset)), int(0.1*len(dataset)), len(dataset)-int(0.8*len(dataset))-int(0.1*len(dataset))],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SimpleNN()
# print(model)
criterion = nn.MSELoss()  # 範例：均方誤差（迴歸問題）
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
# 開始訓練
print("Start Training...")
epoch = 50
for epoch in range(epoch):
    model.train()
    train_loss = 0
    for batch_data, batch_label in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_data.size(0)
    epoch_train_loss = train_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0
    pointer = True
    for batch_data, batch_label in val_loader:
        outputs = model(batch_data)
        loss = criterion(outputs, batch_label)
        val_loss += loss.item() * batch_data.size(0)
        # if pointer == True:
            # print(outputs.tolist()[:10])
            # print(batch_label.tolist()[:10])
            # pointer = False
    epoch_val_loss = val_loss / len(val_loader.dataset)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

print("Finish training and validating.")
torch.save(model.state_dict(), 'db_checkpoint.pth')
print("參數已成功存成 .pth 檔案！")
# print("Now start testing data in test_dataset")