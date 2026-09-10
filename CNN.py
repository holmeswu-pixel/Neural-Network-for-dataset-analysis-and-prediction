import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        
        # 卷積層 1：輸入 1 個通道（灰階），輸出 32 個特徵圖，卷積核 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # 卷積層 2：輸入 32 個通道，輸出 64 個特徵圖，卷積核 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 最大池化層：2x2 視窗，會把圖片寬高各減半
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 丟棄層：防止過擬合（Overfitting）
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全連接層 1：經過兩次池化後，28x28 的圖片會變成 7x7
        # 64 個通道 x 7 x 7 = 3136 個節點 -> 轉換為 128 個節點
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # 全連接層 2（輸出層）：128 個節點 -> 10 個節點（對應 0~9 個數字）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x 的原始形狀: [batch_size, 1, 28, 28]
        
        # 第一層卷積 + 激活函數 (ReLU) + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 形狀變為: [batch_size, 32, 14, 14]
        
        # 第二層卷積 + 激活函數 (ReLU) + 池化
        x = self.pool(F.relu(self.conv2(x)))  # 形狀變為: [batch_size, 64, 7, 7]
        
        x = self.dropout1(x)
        
        # 展平（Flatten）：將 3D 的特徵圖壓平成 1D 向量，準備送入全連接層
        x = x.view(-1, 64 * 7 * 7)            # 形狀變為: [batch_size, 3136]
        
        # 全連接層 1 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        # 輸出層
        x = self.fc2(x)                       # 形狀變為: [batch_size, 10]
        
        return x

# ==========================================
# 1. 定義「訓練」函數 (單一 Epoch)
# ==========================================
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # 將模型切換為【訓練模式】（會啟用 Dropout）
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 將資料與標籤移至 GPU 或 CPU
        data, target = data.to(device), target.to(device)
        
        # 核心三步驟：
        optimizer.zero_grad()   # 1. 清空上一步的梯度
        output = model(data)    # 2. 前向傳播（預測結果）
        loss = criterion(output, target) # 3. 計算損失
        
        loss.backward()         # 4. 反向傳播（計算梯度）
        optimizer.step()        # 5. 更新網路權重
        
        running_loss += loss.item()
        
        # 每 200 個 batch 印一次進度
        if batch_idx % 200 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


# ==========================================
# 2. 定義「測試/驗證」函數
# ==========================================
def test(model, device, test_loader, criterion):
    model.eval()  # 將模型切換為【評估模式】（會關閉 Dropout）
    test_loss = 0
    correct = 0
    
    # 測試時不需要計算梯度，加上這行可以節省記憶體並加速
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 累加測試損失
            test_loss += criterion(output, target).item() * data.size(0)
            
            # 取得預測機率最高的類別 (Index)
            pred = output.argmax(dim=1, keepdim=True)
            
            # 比較預測值與真實標籤，累加正確人數
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")

if __name__ == '__main__':
    # 2. 實例化模型
    model = MNISTCNN()
    print(model)

    # ==========================================
    # 0. 偵測硬體與初始化
    # ==========================================
    # 如果電腦有 NVIDIA 顯示卡，就用 GPU (cuda) 跑，否則用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用的運算裝置: {device}")

    # 實例化模型並移至指定裝置
    model = MNISTCNN().to(device)

    # 定義損失函數（多分類任務通常用交叉熵）
    criterion = nn.CrossEntropyLoss()

    # 定義優化器（Adam 是目前最通用且高效的優化器）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的均值和標準差
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # ==========================================
    # 3. 開始執行訓練（假設跑 3 個 Epoch）
    # ==========================================
    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
    torch.save(model.state_dict(), 'model_checkpoint_cnn.pth')
    print("參數已成功存成 .pth 檔案！")