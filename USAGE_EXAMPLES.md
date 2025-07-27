# 使用示例和教程

## 目录

1. [快速开始](#快速开始)
2. [基础使用示例](#基础使用示例)
3. [高级使用场景](#高级使用场景)
4. [自定义配置](#自定义配置)
5. [图像处理示例](#图像处理示例)
6. [模型部署](#模型部署)
7. [常见问题解答](#常见问题解答)

---

## 快速开始

### 1. 环境设置

```bash
# 创建项目目录
mkdir mnist_project
cd mnist_project

# 克隆或下载项目文件
# 如果文件已存在，请跳过此步骤

# 创建虚拟环境
python3 -m venv mnist_env
source mnist_env/bin/activate  # Linux/Mac
# 或
mnist_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行基本示例

```bash
# 运行完整的训练和评估流程
python mnist_cnn_classifier.py
```

预期输出：
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
...
训练完成! 总用时: 139.75秒

4. 评估模型...
测试准确率: 99.18%

6. 保存模型...
模型已保存到: mnist_cnn_model.pth
```

---

## 基础使用示例

### 示例1: 数据加载和检查

```python
import torch
import matplotlib.pyplot as plt
from mnist_cnn_classifier import load_mnist_data

# 加载数据
train_loader, test_loader = load_mnist_data()

# 检查数据形状
print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# 获取一个批次的数据
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"图像批次形状: {images.shape}")
print(f"标签批次形状: {labels.shape}")

# 可视化几个样本
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(images[i].squeeze(), cmap='gray')
    axes[row, col].set_title(f'标签: {labels[i]}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()
```

### 示例2: 创建和检查模型

```python
from mnist_cnn_classifier import CNN_MNIST
import torch

# 创建模型
model = CNN_MNIST()
print(model)

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")

# 测试前向传播
device = torch.device('cpu')
model.to(device)
dummy_input = torch.randn(1, 1, 28, 28).to(device)
output = model(dummy_input)
print(f"输出形状: {output.shape}")
print(f"输出概率: {torch.softmax(output, dim=1)}")
```

### 示例3: 简单训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from mnist_cnn_classifier import CNN_MNIST, load_mnist_data

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader, test_loader = load_mnist_data()

# 创建模型
model = CNN_MNIST().to(device)

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练一个epoch
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx >= 10:  # 只训练10个批次作为示例
        break
        
    data, target = data.to(device), target.to(device)
    
    # 前向传播
    output = model(data)
    loss = criterion(output, target)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if batch_idx % 5 == 0:
        print(f'批次 {batch_idx}, 损失: {loss.item():.4f}')
```

---

## 高级使用场景

### 场景1: 自定义数据增强

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

# 定义增强变换
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 创建数据集
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform_train
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform_test
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"增强后训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
```

### 场景2: 模型性能分析

```python
import time
import torch
from mnist_cnn_classifier import CNN_MNIST, load_mnist_data, evaluate_model

# 性能分析函数
def analyze_model_performance(model, test_loader, device):
    """分析模型性能"""
    model.eval()
    
    # 推理时间分析
    inference_times = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= 10:  # 只测试10个批次
                break
                
            data = data.to(device)
            
            # 测量推理时间
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
    
    # 计算统计信息
    avg_time = sum(inference_times) / len(inference_times)
    samples_per_second = test_loader.batch_size / avg_time
    
    print(f"平均推理时间: {avg_time:.4f} 秒/批次")
    print(f"处理速度: {samples_per_second:.1f} 样本/秒")
    
    # 内存使用分析
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"GPU内存使用: {memory_allocated:.1f} MB")
        print(f"GPU内存保留: {memory_reserved:.1f} MB")

# 使用示例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_MNIST().to(device)
_, test_loader = load_mnist_data()

analyze_model_performance(model, test_loader, device)
```

### 场景3: 模型解释性分析

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from mnist_cnn_classifier import CNN_MNIST, load_mnist_data

def visualize_feature_maps(model, input_image, device):
    """可视化特征图"""
    model.eval()
    
    # 注册钩子函数
    feature_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # 在卷积层注册钩子
    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))
    
    # 前向传播
    with torch.no_grad():
        input_tensor = input_image.unsqueeze(0).to(device)
        _ = model(input_tensor)
    
    # 可视化特征图
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    # 显示conv1特征图（前32个中的前16个）
    conv1_features = feature_maps['conv1'][0]  # [32, 14, 14]
    for i in range(16):
        row, col = i // 8, i % 8
        axes[row, col].imshow(conv1_features[i].cpu(), cmap='viridis')
        axes[row, col].set_title(f'Conv1-{i}')
        axes[row, col].axis('off')
    
    # 显示conv2特征图（前64个中的前16个）
    conv2_features = feature_maps['conv2'][0]  # [64, 7, 7]
    for i in range(16):
        row, col = (i // 8) + 2, i % 8
        axes[row, col].imshow(conv2_features[i].cpu(), cmap='viridis')
        axes[row, col].set_title(f'Conv2-{i}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_MNIST().to(device)
_, test_loader = load_mnist_data()

# 获取一个测试样本
data_iter = iter(test_loader)
images, labels = next(data_iter)
sample_image = images[0]  # 选择第一个样本

visualize_feature_maps(model, sample_image, device)
```

---

## 自定义配置

### 配置1: 修改网络架构

```python
import torch.nn as nn
import torch.nn.functional as F

class Enhanced_CNN_MNIST(nn.Module):
    """增强版CNN模型"""
    def __init__(self, num_classes=10):
        super(Enhanced_CNN_MNIST, self).__init__()
        
        # 更深的网络结构
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # 第三个卷积块
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        # 全连接层
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# 使用增强模型
enhanced_model = Enhanced_CNN_MNIST()
print(f"增强模型参数数量: {sum(p.numel() for p in enhanced_model.parameters()):,}")
```

### 配置2: 自定义训练参数

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class TrainingConfig:
    """训练配置类"""
    def __init__(self):
        self.epochs = 20
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.scheduler_type = 'cosine'  # 'step', 'cosine', 'plateau'
        self.early_stopping_patience = 5
        self.save_best_model = True
        
def create_optimizer_and_scheduler(model, config):
    """根据配置创建优化器和调度器"""
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 创建学习率调度器
    if config.scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1
        )
    elif config.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
    else:
        scheduler = None
    
    return optimizer, scheduler

# 使用示例
config = TrainingConfig()
model = CNN_MNIST()
optimizer, scheduler = create_optimizer_and_scheduler(model, config)

print(f"优化器: {type(optimizer).__name__}")
print(f"调度器: {type(scheduler).__name__}")
print(f"学习率: {config.learning_rate}")
```

---

## 图像处理示例

### 示例1: 全局变换处理

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 模拟图像处理函数（基于run_global_transform.py的概念）
def apply_brightness_adjustment(image, factor=1.2):
    """调整图像亮度"""
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def apply_contrast_adjustment(image, factor=1.5):
    """调整图像对比度"""
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

def apply_gamma_correction(image, gamma=0.8):
    """应用伽马校正"""
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255).astype(np.uint8)

# 示例使用
def demonstrate_global_transforms():
    """演示全局变换"""
    # 创建示例图像（或加载真实图像）
    original = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # 应用不同的变换
    bright = apply_brightness_adjustment(original, factor=1.5)
    contrast = apply_contrast_adjustment(original, factor=2.0)
    gamma = apply_gamma_correction(original, gamma=0.6)
    
    # 可视化结果
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    images = [original, bright, contrast, gamma]
    titles = ['原图', '亮度增强', '对比度增强', '伽马校正']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

demonstrate_global_transforms()
```

### 示例2: 点变换处理

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟点变换函数（基于run_point_transform.py的概念）
def threshold_transform(image, threshold=128):
    """二值化变换"""
    return np.where(image >= threshold, 255, 0).astype(np.uint8)

def negative_transform(image):
    """负片变换"""
    return (255 - image).astype(np.uint8)

def log_transform(image, c=50):
    """对数变换"""
    return np.clip(c * np.log(1 + image), 0, 255).astype(np.uint8)

def power_transform(image, gamma=0.5, c=1):
    """幂函数变换"""
    normalized = image / 255.0
    transformed = c * np.power(normalized, gamma)
    return (transformed * 255).astype(np.uint8)

# 示例使用
def demonstrate_point_transforms():
    """演示点变换"""
    # 创建示例图像
    original = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # 应用不同的点变换
    binary = threshold_transform(original, threshold=128)
    negative = negative_transform(original)
    log_img = log_transform(original, c=50)
    power_img = power_transform(original, gamma=0.5)
    
    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    images = [original, binary, negative, log_img, power_img]
    titles = ['原图', '二值化', '负片', '对数变换', '幂函数变换']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < 3:
            row, col = 0, i
        else:
            row, col = 1, i - 3
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    # 隐藏最后一个子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

demonstrate_point_transforms()
```

---

## 模型部署

### 部署1: TorchScript导出

```python
import torch
from mnist_cnn_classifier import CNN_MNIST, load_model

def export_to_torchscript(model_path, output_path):
    """将模型导出为TorchScript格式"""
    
    # 加载模型
    device = torch.device('cpu')  # 生产部署通常使用CPU
    model = load_model(model_path, device)
    model.eval()
    
    # 创建示例输入
    example_input = torch.randn(1, 1, 28, 28)
    
    # 转换为TorchScript
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存TorchScript模型
    traced_model.save(output_path)
    print(f"TorchScript模型已保存到: {output_path}")
    
    return traced_model

# 使用示例
traced_model = export_to_torchscript('mnist_cnn_model.pth', 'mnist_model_traced.pt')

# 测试TorchScript模型
test_input = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    output = traced_model(test_input)
    prediction = output.argmax(dim=1).item()
    print(f"TorchScript模型预测: {prediction}")
```

### 部署2: REST API服务

```python
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

# 全局变量存储模型
model = None
device = None
transform = None

def load_model_for_serving():
    """加载模型用于服务"""
    global model, device, transform
    
    device = torch.device('cpu')  # 生产环境通常使用CPU
    
    # 加载TorchScript模型
    model = torch.jit.load('mnist_model_traced.pt', map_location=device)
    model.eval()
    
    # 定义预处理变换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("模型加载成功，服务已启动")

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        # 获取图像数据
        data = request.json
        image_data = data['image']  # base64编码的图像
        
        # 解码图像
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 预处理
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    load_model_for_serving()
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 部署3: 批量推理脚本

```python
import torch
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

class BatchInference:
    """批量推理类"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def predict_single_image(self, image_path):
        """预测单张图像"""
        try:
            # 加载和预处理图像
            image = Image.open(image_path)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            return {
                'image_path': image_path,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities[0].tolist()
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e)
            }
    
    def predict_batch(self, image_folder, output_file):
        """批量预测文件夹中的所有图像"""
        
        # 获取所有图像文件
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, file))
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 批量预测
        results = []
        for image_path in tqdm(image_files, desc="处理图像"):
            result = self.predict_single_image(image_path)
            results.append(result)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_file}")
        
        # 统计信息
        successful_predictions = sum(1 for r in results if 'prediction' in r)
        print(f"成功预测: {successful_predictions}/{len(results)} 张图像")

# 使用示例
if __name__ == "__main__":
    # 创建批量推理实例
    batch_inference = BatchInference('mnist_model_traced.pt', device='cpu')
    
    # 批量预测
    # batch_inference.predict_batch('test_images/', 'predictions.json')
    
    print("批量推理系统已准备就绪")
```

---

## 常见问题解答

### Q1: 如何解决CUDA内存不足问题？

**解决方案：**

```python
# 方法1: 减少批次大小
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 从64减少到32

# 方法2: 使用梯度累积
def train_with_gradient_accumulation(model, train_loader, optimizer, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = criterion(output, target) / accumulation_steps  # 缩放损失
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# 方法3: 释放缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Q2: 如何提高模型训练速度？

**解决方案：**

```python
# 方法1: 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

# 方法2: 使用pin_memory和num_workers
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 方法3: 编译模型（PyTorch 2.0+）
# model = torch.compile(model)
```

### Q3: 如何处理过拟合问题？

**解决方案：**

```python
# 方法1: 增加正则化
import torch.nn as nn

class RegularizedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 网络层定义
        self.dropout1 = nn.Dropout(0.3)  # 增加dropout
        self.dropout2 = nn.Dropout(0.5)
        
    def forward(self, x):
        # ... 前向传播
        x = self.dropout1(x)  # 在多个位置添加dropout
        # ...
        x = self.dropout2(x)
        return x

# 方法2: 早停机制
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = val_score
            self.counter = 0
        return False

# 方法3: 数据增强
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Q4: 如何保存和恢复训练状态？

**解决方案：**

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, filename, device):
    """加载训练检查点"""
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

# 使用示例
# 保存检查点
save_checkpoint(model, optimizer, scheduler, epoch, loss, 'checkpoint.pth')

# 恢复训练
start_epoch, last_loss = load_checkpoint(
    model, optimizer, scheduler, 'checkpoint.pth', device
)
```

这份使用示例文档提供了从基础到高级的各种使用场景，帮助用户快速上手并深入使用项目功能。