import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

class CNN_MNIST(nn.Module):
    """
    CNN模型用于MNIST手写数字分类
    
    网络结构:
    - 两个卷积层 + 池化层
    - 两个全连接层
    - Dropout正则化
    """
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一个卷积块: Conv -> ReLU -> MaxPool
        x = self.pool1(F.relu(self.conv1(x)))
        
        # 第二个卷积块: Conv -> ReLU -> MaxPool
        x = self.pool2(F.relu(self.conv2(x)))
        
        # 展平特征图
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_data(batch_size=64):
    """
    加载MNIST数据集
    
    Args:
        batch_size (int): 批次大小
        
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 下载和加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 下载和加载测试集
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    """
    训练CNN模型
    
    Args:
        model: CNN模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_epochs (int): 训练轮数
        learning_rate (float): 学习率
        
    Returns:
        训练历史记录
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 计算训练准确率
        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # 测试阶段
        test_accuracy = evaluate_model(model, test_loader, verbose=False)
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
    
    training_time = time.time() - start_time
    print(f'\n训练完成! 总用时: {training_time:.2f}秒')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

def evaluate_model(model, test_loader, verbose=True):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        verbose (bool): 是否打印详细信息
        
    Returns:
        test_accuracy: 测试准确率
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    if verbose:
        print(f'\n测试准确率: {accuracy:.2f}%')
        
        # 打印分类报告
        print('\n分类报告:')
        print(classification_report(all_targets, all_predictions, 
                                  target_names=[str(i) for i in range(10)]))
        
        # 绘制混淆矩阵
        plot_confusion_matrix(all_targets, all_predictions)
    
    return accuracy

def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(history['train_losses'], label='训练损失')
    ax1.set_title('训练损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history['train_accuracies'], label='训练准确率')
    ax2.plot(history['test_accuracies'], label='测试准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, test_loader, num_samples=10):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].cpu().numpy().squeeze()
        true_label = labels[i].cpu().item()
        pred_label = predictions[i].cpu().item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'真实: {true_label}, 预测: {pred_label}')
        axes[i].axis('off')
        
        # 如果预测错误，用红色标题
        if true_label != pred_label:
            axes[i].set_title(f'真实: {true_label}, 预测: {pred_label}', color='red')
    
    plt.tight_layout()
    plt.show()

def save_model(model, filepath='mnist_cnn_model.pth'):
    """保存模型"""
    torch.save(model.state_dict(), filepath)
    print(f'模型已保存到: {filepath}')

def load_model(filepath='mnist_cnn_model.pth'):
    """加载模型"""
    model = CNN_MNIST().to(device)
    model.load_state_dict(torch.load(filepath))
    print(f'模型已从 {filepath} 加载')
    return model

def main():
    """主函数"""
    print("=" * 50)
    print("CNN MNIST 手写数字分类器")
    print("=" * 50)
    
    # 设置超参数
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # 加载数据
    print("\n1. 加载数据...")
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE)
    
    # 创建模型
    print("\n2. 创建模型...")
    model = CNN_MNIST().to(device)
    print(f'模型参数总数: {sum(p.numel() for p in model.parameters())}')
    
    # 训练模型
    print("\n3. 训练模型...")
    history = train_model(model, train_loader, test_loader, 
                         num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    
    # 评估模型
    print("\n4. 评估模型...")
    final_accuracy = evaluate_model(model, test_loader)
    
    # 可视化结果
    print("\n5. 可视化结果...")
    plot_training_history(history)
    visualize_predictions(model, test_loader)
    
    # 保存模型
    print("\n6. 保存模型...")
    save_model(model)
    
    print(f"\n训练完成! 最终测试准确率: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()