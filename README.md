# CNN MNIST 手写数字分类器

一个使用卷积神经网络(CNN)对MNIST手写数字数据集进行分类的完整实现。该项目包含模型定义、训练、评估、可视化和模型保存功能。

## 项目概述

本项目实现了一个高性能的CNN模型，用于识别手写数字。最终在MNIST测试集上达到了**99.18%**的准确率。

### 关键特性

- **现代CNN架构**: 使用两个卷积层和池化层进行特征提取
- **自动数据预处理**: 包含数据标准化和增强功能
- **实时训练监控**: 显示训练进度和性能指标
- **全面的评估**: 包含分类报告和混淆矩阵
- **可视化工具**: 训练曲线和预测结果可视化
- **模型持久化**: 自动保存训练好的模型

## 环境要求

### 依赖包
```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.19.0
scikit-learn>=0.24.0
seaborn>=0.11.0
```

### 安装指南

1. 创建虚拟环境：
```bash
python3 -m venv mnist_env
source mnist_env/bin/activate  # Linux/Mac
# 或者
mnist_env\Scripts\activate     # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用

运行完整的训练和评估流程：

```bash
python mnist_cnn_classifier.py
```

这将自动执行以下步骤：
1. 下载并加载MNIST数据集
2. 创建CNN模型
3. 训练模型（10个epochs）
4. 评估模型性能
5. 生成可视化结果
6. 保存训练好的模型

### 程序输出示例

```
使用设备: cpu
==================================================
CNN MNIST 手写数字分类器
==================================================

1. 加载数据...
训练集大小: 60000
测试集大小: 10000

2. 创建模型...
模型参数总数: 421642

3. 训练模型...
Epoch [1/10] - Train Loss: 0.2329, Train Acc: 92.81%, Test Acc: 98.30%
Epoch [2/10] - Train Loss: 0.0893, Train Acc: 97.40%, Test Acc: 98.79%
...
Epoch [10/10] - Train Loss: 0.0250, Train Acc: 99.25%, Test Acc: 99.18%

训练完成! 总用时: 139.75秒

4. 评估模型...
测试准确率: 99.18%

6. 保存模型...
模型已保存到: mnist_cnn_model.pth
```

## API 文档

### 核心类 - CNN_MNIST

```python
class CNN_MNIST(nn.Module):
    """
    CNN模型用于MNIST手写数字分类
    
    网络结构:
    - 两个卷积层 + 池化层
    - 两个全连接层  
    - Dropout正则化
    """
```

#### 构造函数

```python
def __init__(self):
    """
    初始化CNN模型
    
    网络层结构:
    - conv1: Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    - pool1: MaxPool2d(kernel_size=2, stride=2) 
    - conv2: Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    - pool2: MaxPool2d(kernel_size=2, stride=2)
    - fc1: Linear(64 * 7 * 7, 128)
    - dropout: Dropout(0.5)
    - fc2: Linear(128, 10)
    """
```

#### 前向传播

```python
def forward(self, x):
    """
    前向传播函数
    
    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, 1, 28, 28)
        
    Returns:
        torch.Tensor: 输出张量，形状为 (batch_size, 10)，包含10个类别的概率
        
    流程:
        1. 第一个卷积块: conv1 -> ReLU -> pool1
        2. 第二个卷积块: conv2 -> ReLU -> pool2  
        3. 展平: flatten
        4. 全连接层: fc1 -> ReLU -> dropout -> fc2
    """
```

### 核心函数

#### 数据加载函数

```python
def load_mnist_data():
    """
    加载并预处理MNIST数据集
    
    Returns:
        tuple: (train_loader, test_loader)
            - train_loader: 训练数据加载器，batch_size=64
            - test_loader: 测试数据加载器，batch_size=1000
            
    数据预处理:
        - 转换为张量
        - 标准化: mean=0.1307, std=0.3081
        - 训练集：随机采样
        - 测试集：顺序采样
    """
```

#### 训练函数

```python
def train_model(model, train_loader, test_loader, device, epochs=10):
    """
    训练CNN模型
    
    Args:
        model (CNN_MNIST): 要训练的模型
        train_loader (DataLoader): 训练数据加载器
        test_loader (DataLoader): 测试数据加载器  
        device (torch.device): 计算设备 (CPU/GPU)
        epochs (int): 训练轮数，默认10
        
    Returns:
        tuple: (train_losses, train_accuracies, test_accuracies)
            - train_losses: 每个epoch的平均训练损失
            - train_accuracies: 每个epoch的训练准确率
            - test_accuracies: 每个epoch的测试准确率
            
    训练配置:
        - 优化器: Adam, lr=0.001
        - 损失函数: CrossEntropyLoss
        - 学习率调度: StepLR, step_size=5, gamma=0.1
    """
```

#### 评估函数

```python
def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    
    Args:
        model (CNN_MNIST): 训练好的模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备
        
    Returns:
        tuple: (accuracy, y_true, y_pred)
            - accuracy (float): 测试准确率 (0-1之间)
            - y_true (list): 真实标签列表
            - y_pred (list): 预测标签列表
            
    评估指标:
        - 整体准确率
        - 每个类别的精确率、召回率、F1分数
        - 混淆矩阵
    """
```

#### 可视化函数

```python
def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    """
    绘制训练曲线
    
    Args:
        train_losses (list): 训练损失历史
        train_accuracies (list): 训练准确率历史  
        test_accuracies (list): 测试准确率历史
        
    生成图表:
        - 左图: 训练损失随epoch变化
        - 右图: 训练和测试准确率对比
    """

def plot_predictions(model, test_loader, device, num_samples=10):
    """
    可视化预测结果
    
    Args:
        model (CNN_MNIST): 训练好的模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备
        num_samples (int): 要显示的样本数量
        
    显示内容:
        - 原始手写数字图像
        - 真实标签  
        - 模型预测结果
        - 预测置信度
    """
```

#### 模型保存和加载

```python
def save_model(model, filepath):
    """
    保存训练好的模型
    
    Args:
        model (CNN_MNIST): 要保存的模型
        filepath (str): 保存路径，推荐使用 .pth 扩展名
        
    保存内容:
        - 模型结构参数
        - 训练好的权重
        - 模型状态字典
    """

def load_model(filepath, device):
    """
    加载保存的模型
    
    Args:
        filepath (str): 模型文件路径
        device (torch.device): 目标设备
        
    Returns:
        CNN_MNIST: 加载的模型实例
        
    注意:
        - 模型将自动设置为评估模式
        - 确保文件路径正确
    """
```

## 模型架构详解

### 网络结构

```
输入: (batch_size, 1, 28, 28)
│
├─ Conv2d(1→32, 3x3, padding=1) → ReLU → MaxPool2d(2x2)
│  输出: (batch_size, 32, 14, 14)
│
├─ Conv2d(32→64, 3x3, padding=1) → ReLU → MaxPool2d(2x2)  
│  输出: (batch_size, 64, 7, 7)
│
├─ Flatten → Linear(3136→128) → ReLU → Dropout(0.5)
│  输出: (batch_size, 128)
│
└─ Linear(128→10)
   输出: (batch_size, 10)
```

### 参数统计

- **总参数数量**: 421,642
- **卷积层参数**: 
  - Conv1: 320 (32×1×3×3 + 32)
  - Conv2: 18,496 (64×32×3×3 + 64)
- **全连接层参数**:
  - FC1: 401,536 (3136×128 + 128)  
  - FC2: 1,290 (128×10 + 10)

### 训练配置

```python
训练参数:
- epochs: 10
- batch_size: 64 (训练), 1000 (测试)
- learning_rate: 0.001
- optimizer: Adam
- loss_function: CrossEntropyLoss
- scheduler: StepLR (step_size=5, gamma=0.1)
- device: auto-detect (CUDA/CPU)
```

## 性能指标

### 训练结果

- **最终测试准确率**: 99.18%
- **训练时间**: ~140秒 (CPU)
- **收敛特性**: 在第3个epoch后快速收敛

### 各类别性能

| 数字 | 精确率 | 召回率 | F1分数 | 支持数 |
|------|--------|--------|--------|--------|
| 0    | 0.99   | 1.00   | 0.99   | 980    |
| 1    | 0.99   | 1.00   | 1.00   | 1135   |
| 2    | 0.99   | 0.99   | 0.99   | 1032   |
| 3    | 1.00   | 0.99   | 0.99   | 1010   |
| 4    | 0.99   | 0.99   | 0.99   | 982    |
| 5    | 0.98   | 0.99   | 0.99   | 892    |
| 6    | 1.00   | 0.99   | 0.99   | 958    |
| 7    | 0.98   | 0.99   | 0.99   | 1028   |
| 8    | 1.00   | 0.99   | 0.99   | 974    |
| 9    | 0.99   | 0.99   | 0.99   | 1009   |

## 高级使用

### 自定义训练参数

```python
# 修改main函数中的参数
def main():
    # 自定义参数
    EPOCHS = 20          # 增加训练轮数
    BATCH_SIZE = 32      # 修改批次大小
    LEARNING_RATE = 0.0001  # 调整学习率
    
    # 创建自定义数据加载器
    train_loader = DataLoader(train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)
    
    # 使用自定义优化器
    optimizer = optim.SGD(model.parameters(), 
                         lr=LEARNING_RATE, 
                         momentum=0.9)
```

### 模型推理示例

```python
# 加载预训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('mnist_cnn_model.pth', device)

# 单张图片预测
def predict_single_image(image_path):
    """
    对单张图片进行预测
    
    Args:
        image_path (str): 图片路径
        
    Returns:
        int: 预测的数字 (0-9)
    """
    # 加载和预处理图片
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()
    
    return prediction
```

### 模型优化建议

1. **数据增强**:
   ```python
   transform = transforms.Compose([
       transforms.RandomRotation(10),
       transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])
   ```

2. **学习率调度**:
   ```python
   scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
   ```

3. **早停机制**:
   ```python
   best_accuracy = 0
   patience = 3
   patience_counter = 0
   
   for epoch in range(epochs):
       # 训练代码...
       if current_accuracy > best_accuracy:
           best_accuracy = current_accuracy
           patience_counter = 0
           # 保存最佳模型
       else:
           patience_counter += 1
           if patience_counter >= patience:
               print("Early stopping...")
               break
   ```

## 故障排除

### 常见问题

1. **CUDA内存不足**:
   ```bash
   RuntimeError: CUDA out of memory
   ```
   解决方案: 减少batch_size或使用CPU训练

2. **依赖包版本冲突**:
   ```bash
   ImportError: cannot import name 'xxx'
   ```
   解决方案: 更新依赖包或使用虚拟环境

3. **字体警告** (可忽略):
   ```
   UserWarning: Glyph xxx missing from font(s) DejaVu Sans
   ```
   这是中文字体显示问题，不影响程序运行

### 性能优化

1. **使用GPU加速**:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **并行数据加载**:
   ```python
   train_loader = DataLoader(train_dataset, 
                           batch_size=64, 
                           shuffle=True, 
                           num_workers=4)
   ```

3. **混合精度训练**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

## 文件结构

```
├── mnist_cnn_classifier.py    # 主程序文件
├── requirements.txt           # 依赖包列表  
├── README.md                 # 项目文档
├── mnist_cnn_model.pth       # 保存的模型文件 (训练后生成)
└── mnist_env/                # 虚拟环境目录
```

## 版本历史

- **v1.0**: 初始版本
  - 基础CNN实现
  - MNIST数据集支持
  - 训练和评估功能

## 许可证

本项目使用MIT许可证 - 详见LICENSE文件

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 致谢

- PyTorch团队提供的深度学习框架
- MNIST数据集的创建者
- 开源社区的支持和贡献
