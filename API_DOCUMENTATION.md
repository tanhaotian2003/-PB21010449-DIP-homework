# API 文档

## 目录

1. [mnist_cnn_classifier.py](#mnist_cnn_classifierpy)
2. [run_global_transform.py](#run_global_transformpy)
3. [run_point_transform.py](#run_point_transformpy)
4. [数据结构](#数据结构)
5. [错误处理](#错误处理)
6. [最佳实践](#最佳实践)

---

## mnist_cnn_classifier.py

### 类定义

#### CNN_MNIST

```python
class CNN_MNIST(nn.Module):
    """
    用于MNIST手写数字分类的卷积神经网络模型
    
    继承自 torch.nn.Module，实现了一个包含两个卷积层和两个全连接层的CNN架构。
    """
```

**初始化方法**:

```python
def __init__(self) -> None:
    """
    初始化CNN模型的所有层
    
    网络架构:
        - conv1: 1→32通道卷积层，3x3卷积核，填充1
        - pool1: 2x2最大池化层
        - conv2: 32→64通道卷积层，3x3卷积核，填充1  
        - pool2: 2x2最大池化层
        - fc1: 全连接层，3136→128
        - dropout: 50%丢弃率
        - fc2: 输出层，128→10
        
    Returns:
        None
    """
```

**前向传播方法**:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    定义网络的前向传播过程
    
    Args:
        x (torch.Tensor): 输入张量
            - 形状: (batch_size, 1, 28, 28)
            - 数据类型: float32
            - 值范围: 通常为标准化后的 [-1, 1]
            
    Returns:
        torch.Tensor: 输出张量
            - 形状: (batch_size, 10)
            - 数据类型: float32
            - 含义: 10个类别的未归一化概率分数
            
    处理流程:
        1. 输入 → conv1 → ReLU → pool1
        2. → conv2 → ReLU → pool2
        3. → 展平 → fc1 → ReLU → dropout
        4. → fc2 → 输出
        
    注意:
        - 在训练模式下启用dropout
        - 在评估模式下禁用dropout
    """
```

### 函数定义

#### 数据处理函数

```python
def load_mnist_data() -> Tuple[DataLoader, DataLoader]:
    """
    加载和预处理MNIST数据集
    
    功能:
        - 下载MNIST数据集（如果不存在）
        - 应用数据变换和标准化
        - 创建数据加载器
        
    数据变换:
        - ToTensor(): 将PIL图像转换为张量
        - Normalize(mean=0.1307, std=0.3081): 标准化像素值
        
    Returns:
        Tuple[DataLoader, DataLoader]: 训练和测试数据加载器
            - train_loader: 
                * batch_size=64
                * shuffle=True
                * 训练集大小: 60,000
            - test_loader:
                * batch_size=1000  
                * shuffle=False
                * 测试集大小: 10,000
                
    异常:
        RuntimeError: 数据下载失败时抛出
        IOError: 数据文件损坏时抛出
        
    示例:
        >>> train_loader, test_loader = load_mnist_data()
        >>> print(f"训练批次数: {len(train_loader)}")
        >>> print(f"测试批次数: {len(test_loader)}")
    """
```

#### 模型训练函数

```python
def train_model(
    model: CNN_MNIST, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    device: torch.device, 
    epochs: int = 10
) -> Tuple[List[float], List[float], List[float]]:
    """
    训练CNN模型并返回训练历史
    
    Args:
        model (CNN_MNIST): 要训练的模型实例
        train_loader (DataLoader): 训练数据加载器
        test_loader (DataLoader): 测试数据加载器  
        device (torch.device): 计算设备 ('cuda' 或 'cpu')
        epochs (int, optional): 训练轮数，默认10
        
    Returns:
        Tuple[List[float], List[float], List[float]]: 训练历史记录
            - train_losses: 每个epoch的平均训练损失
            - train_accuracies: 每个epoch的训练准确率 (0-100)
            - test_accuracies: 每个epoch的测试准确率 (0-100)
            
    训练配置:
        - 优化器: Adam (lr=0.001)
        - 损失函数: CrossEntropyLoss
        - 学习率调度器: StepLR (step_size=5, gamma=0.1)
        
    训练过程:
        1. 前向传播计算损失
        2. 反向传播计算梯度
        3. 优化器更新参数
        4. 每个epoch后在测试集上评估
        5. 每100个batch打印进度
        
    性能监控:
        - 实时显示训练损失和准确率
        - 自动计算并显示测试准确率
        - 记录总训练时间
        
    异常:
        RuntimeError: GPU内存不足时抛出
        ValueError: 数据形状不匹配时抛出
        
    示例:
        >>> model = CNN_MNIST()
        >>> device = torch.device('cuda')
        >>> losses, train_acc, test_acc = train_model(
        ...     model, train_loader, test_loader, device, epochs=5
        ... )
    """
```

#### 模型评估函数

```python
def evaluate_model(
    model: CNN_MNIST, 
    test_loader: DataLoader, 
    device: torch.device
) -> Tuple[float, List[int], List[int]]:
    """
    评估训练好的模型性能
    
    Args:
        model (CNN_MNIST): 训练好的模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备
        
    Returns:
        Tuple[float, List[int], List[int]]: 评估结果
            - accuracy (float): 整体准确率 (0-1之间)
            - y_true (List[int]): 真实标签列表
            - y_pred (List[int]): 预测标签列表
            
    评估指标:
        - 准确率: 正确预测数 / 总样本数
        - 分类报告: 精确率、召回率、F1分数
        - 混淆矩阵: 各类别间的预测分布
        
    输出信息:
        - 整体测试准确率
        - 每个类别的详细性能指标
        - 支持数（每个类别的样本数）
        
    注意:
        - 模型自动设置为评估模式
        - 禁用梯度计算以提高效率
        - 支持批量预测
        
    示例:
        >>> accuracy, y_true, y_pred = evaluate_model(model, test_loader, device)
        >>> print(f"测试准确率: {accuracy:.4f}")
    """
```

#### 可视化函数

```python
def plot_training_curves(
    train_losses: List[float], 
    train_accuracies: List[float], 
    test_accuracies: List[float]
) -> None:
    """
    绘制训练过程的损失和准确率曲线
    
    Args:
        train_losses (List[float]): 训练损失历史
        train_accuracies (List[float]): 训练准确率历史
        test_accuracies (List[float]): 测试准确率历史
        
    图表特性:
        - 双子图布局 (1行2列)
        - 左图: 训练损失 vs. Epoch
        - 右图: 训练/测试准确率 vs. Epoch
        - 自动图例和网格
        - 中文标题和标签
        
    可视化元素:
        - 损失曲线: 蓝色实线
        - 训练准确率: 绿色实线  
        - 测试准确率: 红色虚线
        - 数据点标记
        - 坐标轴标签
        
    输出:
        - 显示matplotlib图表
        - 自动调整布局
        
    注意:
        - 需要安装matplotlib
        - 可能出现中文字体警告（不影响功能）
        
    示例:
        >>> plot_training_curves(losses, train_acc, test_acc)
    """
```

```python
def plot_predictions(
    model: CNN_MNIST, 
    test_loader: DataLoader, 
    device: torch.device, 
    num_samples: int = 10
) -> None:
    """
    可视化模型预测结果
    
    Args:
        model (CNN_MNIST): 训练好的模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备
        num_samples (int, optional): 显示样本数量，默认10
        
    显示内容:
        - 原始手写数字图像 (28x28灰度图)
        - 真实标签 (0-9)
        - 模型预测结果 (0-9)
        - 预测置信度 (softmax概率)
        
    图表布局:
        - 网格布局: 2行 x (num_samples//2)列
        - 每个子图显示一个样本
        - 标题包含真实值和预测值
        - 颜色编码: 正确预测(绿色), 错误预测(红色)
        
    图像处理:
        - 自动反标准化显示
        - 灰度图像映射
        - 关闭坐标轴显示
        
    预测过程:
        - 模型设为评估模式
        - 禁用梯度计算
        - softmax归一化输出概率
        
    示例:
        >>> plot_predictions(model, test_loader, device, num_samples=8)
    """
```

#### 模型I/O函数

```python
def save_model(model: CNN_MNIST, filepath: str) -> None:
    """
    保存训练好的模型到文件
    
    Args:
        model (CNN_MNIST): 要保存的模型
        filepath (str): 保存路径，建议使用.pth扩展名
        
    保存内容:
        - 模型状态字典 (state_dict)
        - 模型参数和缓冲区
        - 优化器状态 (如果需要)
        
    文件格式:
        - PyTorch原生格式 (.pth)
        - 二进制序列化
        - 支持跨平台加载
        
    注意:
        - 仅保存模型参数，不保存模型结构
        - 加载时需要先创建相同的模型结构
        - 确保有足够的磁盘空间
        
    异常:
        IOError: 文件写入失败时抛出
        PermissionError: 权限不足时抛出
        
    示例:
        >>> save_model(model, 'mnist_cnn_model.pth')
        >>> print("模型已保存")
    """
```

```python
def load_model(filepath: str, device: torch.device) -> CNN_MNIST:
    """
    从文件加载已保存的模型
    
    Args:
        filepath (str): 模型文件路径
        device (torch.device): 目标设备
        
    Returns:
        CNN_MNIST: 加载的模型实例
        
    加载过程:
        1. 创建新的模型实例
        2. 加载状态字典
        3. 设置为评估模式
        4. 移动到指定设备
        
    兼容性:
        - 支持CPU和GPU间的模型转换
        - 自动处理设备映射
        - 向后兼容旧版本模型
        
    异常:
        FileNotFoundError: 文件不存在时抛出
        RuntimeError: 模型结构不匹配时抛出
        
    示例:
        >>> device = torch.device('cpu')
        >>> model = load_model('mnist_cnn_model.pth', device)
        >>> model.eval()
    """
```

#### 主函数

```python
def main() -> None:
    """
    主程序入口函数
    
    执行流程:
        1. 打印程序标题和分隔符
        2. 自动检测并设置计算设备
        3. 加载MNIST数据集
        4. 创建CNN模型并显示参数数量
        5. 训练模型10个epochs
        6. 评估模型性能
        7. 生成训练曲线图
        8. 显示预测结果可视化
        9. 保存训练好的模型
        10. 打印最终结果摘要
        
    设备检测:
        - 优先使用CUDA GPU (如果可用)
        - 回退到CPU (如果无GPU)
        - 自动打印使用的设备信息
        
    输出文件:
        - mnist_cnn_model.pth: 保存的模型文件
        - 训练曲线图 (matplotlib显示)
        - 预测结果图 (matplotlib显示)
        
    性能统计:
        - 模型参数总数
        - 训练总时间
        - 最终测试准确率
        - 各类别详细性能指标
        
    注意:
        - 首次运行会下载MNIST数据集
        - 训练过程显示实时进度
        - 支持键盘中断 (Ctrl+C) 安全退出
        
    示例:
        >>> if __name__ == "__main__":
        ...     main()
    """
```

---

## run_global_transform.py

### 函数定义

#### 全局变换函数

```python
def apply_global_transform(image: np.ndarray, transform_type: str, **kwargs) -> np.ndarray:
    """
    对图像应用全局变换
    
    Args:
        image (np.ndarray): 输入图像
            - 形状: (H, W) 或 (H, W, C)
            - 数据类型: uint8 或 float32
            - 值范围: [0, 255] 或 [0, 1]
        transform_type (str): 变换类型
            - 'brightness': 亮度调整
            - 'contrast': 对比度调整
            - 'gamma': 伽马校正
            - 'histogram_eq': 直方图均衡化
        **kwargs: 变换参数
            
    Returns:
        np.ndarray: 变换后的图像
            - 保持输入图像的形状和数据类型
            - 像素值被裁剪到有效范围
            
    支持的变换:
        1. 亮度调整: brightness_factor (float, default=1.2)
        2. 对比度调整: contrast_factor (float, default=1.5)  
        3. 伽马校正: gamma (float, default=0.8)
        4. 直方图均衡化: 无额外参数
        
    异常:
        ValueError: 不支持的变换类型
        TypeError: 图像数据类型错误
        
    示例:
        >>> bright_img = apply_global_transform(img, 'brightness', brightness_factor=1.3)
        >>> gamma_img = apply_global_transform(img, 'gamma', gamma=0.6)
    """
```

#### 亮度调整

```python
def adjust_brightness(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    调整图像亮度
    
    Args:
        image (np.ndarray): 输入图像
        factor (float): 亮度因子
            - > 1.0: 增加亮度
            - < 1.0: 降低亮度
            - = 1.0: 保持不变
            
    Returns:
        np.ndarray: 亮度调整后的图像
        
    算法:
        output = image * factor
        
    注意:
        - 结果会被裁剪到有效像素范围
        - 保持原始数据类型
        
    示例:
        >>> bright_img = adjust_brightness(image, factor=1.5)
    """
```

#### 对比度调整

```python
def adjust_contrast(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    调整图像对比度
    
    Args:
        image (np.ndarray): 输入图像
        factor (float): 对比度因子
            - > 1.0: 增加对比度
            - < 1.0: 降低对比度
            - = 1.0: 保持不变
            
    Returns:
        np.ndarray: 对比度调整后的图像
        
    算法:
        mean = image.mean()
        output = (image - mean) * factor + mean
        
    特性:
        - 保持图像平均亮度不变
        - 扩展或压缩像素值分布
        
    示例:
        >>> high_contrast = adjust_contrast(image, factor=2.0)
    """
```

#### 伽马校正

```python
def gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    应用伽马校正
    
    Args:
        image (np.ndarray): 输入图像
        gamma (float): 伽马值
            - > 1.0: 图像变暗
            - < 1.0: 图像变亮
            - = 1.0: 保持不变
            
    Returns:
        np.ndarray: 伽马校正后的图像
        
    算法:
        output = 255 * (image / 255) ** gamma
        
    应用场景:
        - 显示器伽马校正
        - 图像增强
        - 色彩空间转换
        
    示例:
        >>> corrected = gamma_correction(image, gamma=0.5)
    """
```

---

## run_point_transform.py

### 函数定义

#### 点变换函数

```python
def apply_point_transform(image: np.ndarray, transform_type: str, **kwargs) -> np.ndarray:
    """
    对图像应用点变换操作
    
    Args:
        image (np.ndarray): 输入图像
        transform_type (str): 变换类型
            - 'threshold': 二值化
            - 'negative': 图像反转
            - 'log': 对数变换
            - 'power': 幂函数变换
        **kwargs: 变换参数
        
    Returns:
        np.ndarray: 变换后的图像
        
    点变换特性:
        - 每个像素独立变换
        - 不依赖邻域像素
        - 计算效率高
        - 保持图像空间结构
        
    示例:
        >>> binary = apply_point_transform(img, 'threshold', threshold=128)
        >>> negative = apply_point_transform(img, 'negative')
    """
```

#### 阈值化

```python
def threshold_transform(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    二值化阈值变换
    
    Args:
        image (np.ndarray): 输入灰度图像
        threshold (int): 阈值 (0-255)
        
    Returns:
        np.ndarray: 二值化图像 (0或255)
        
    算法:
        output = 255 if pixel >= threshold else 0
        
    应用:
        - 图像分割
        - 特征提取
        - 文档图像处理
        
    示例:
        >>> binary = threshold_transform(gray_image, threshold=100)
    """
```

#### 图像反转

```python
def negative_transform(image: np.ndarray) -> np.ndarray:
    """
    图像负片变换
    
    Args:
        image (np.ndarray): 输入图像
        
    Returns:
        np.ndarray: 反转后的图像
        
    算法:
        output = 255 - image
        
    效果:
        - 黑色变白色
        - 白色变黑色
        - 保持图像结构
        
    示例:
        >>> negative = negative_transform(image)
    """
```

#### 对数变换

```python
def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    对数变换增强
    
    Args:
        image (np.ndarray): 输入图像
        c (float): 尺度常数
        
    Returns:
        np.ndarray: 对数变换后的图像
        
    算法:
        output = c * log(1 + image)
        
    特性:
        - 扩展暗部细节
        - 压缩亮部范围
        - 增强低灰度区域对比度
        
    应用:
        - 医学图像增强
        - 傅里叶频谱显示
        
    示例:
        >>> enhanced = log_transform(image, c=50)
    """
```

---

## 数据结构

### 张量形状说明

```python
# 图像张量形状
input_batch: torch.Tensor    # (N, C, H, W)
# N: batch size
# C: 通道数 (1=灰度, 3=RGB)  
# H: 图像高度
# W: 图像宽度

# MNIST特定形状
mnist_image: torch.Tensor    # (1, 28, 28) 单张图像
mnist_batch: torch.Tensor    # (64, 1, 28, 28) 批量图像
mnist_output: torch.Tensor   # (64, 10) 分类输出
```

### 数据类型

```python
# NumPy数组类型
image_uint8: np.ndarray     # dtype=uint8, range=[0, 255]
image_float32: np.ndarray   # dtype=float32, range=[0.0, 1.0]

# PyTorch张量类型  
tensor_float32: torch.Tensor  # dtype=torch.float32
tensor_long: torch.Tensor     # dtype=torch.long (for labels)
```

---

## 错误处理

### 常见异常类型

```python
# 设备相关错误
RuntimeError: "CUDA out of memory"
# 解决: 减少batch_size或使用CPU

# 数据形状错误
ValueError: "Expected 4D tensor, got 3D"
# 解决: 检查输入张量维度

# 文件I/O错误
FileNotFoundError: "Model file not found"
# 解决: 检查文件路径是否正确

# 依赖包错误
ImportError: "No module named 'torch'"
# 解决: 安装缺失的依赖包
```

### 错误处理最佳实践

```python
def safe_model_training():
    """安全的模型训练示例"""
    try:
        # 检查设备可用性
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据
        train_loader, test_loader = load_mnist_data()
        
        # 创建模型
        model = CNN_MNIST().to(device)
        
        # 训练模型
        train_model(model, train_loader, test_loader, device)
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("GPU内存不足，切换到CPU训练")
            device = torch.device('cpu')
            model = model.to(device)
        else:
            raise e
            
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        return None
```

---

## 最佳实践

### 1. 模型训练建议

```python
# 设备检测和设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 设置随机种子确保可重现性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 模型初始化
model = CNN_MNIST()
model.to(device)

# 优化器和学习率调度
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

### 2. 数据处理建议

```python
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST统计值
])

# 数据加载器设置
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,  # 并行数据加载
    pin_memory=True  # 固定内存以加速GPU传输
)
```

### 3. 模型评估建议

```python
def comprehensive_evaluation(model, test_loader, device):
    """全面的模型评估"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # 计算各种评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

### 4. 内存管理建议

```python
# 清理GPU内存
def cleanup_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 在训练循环中定期清理
for epoch in range(epochs):
    # 训练代码...
    
    if epoch % 5 == 0:
        cleanup_memory()
```

### 5. 模型保存和版本管理

```python
def save_model_with_metadata(model, filepath, metadata=None):
    """保存带元数据的模型"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    torch.save(save_dict, filepath)

def load_model_with_validation(filepath, device):
    """加载模型并验证"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # 验证模型架构
    if checkpoint.get('model_architecture') != 'CNN_MNIST':
        raise ValueError("模型架构不匹配")
    
    # 创建并加载模型
    model = CNN_MNIST()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('metadata', {})
```

这份API文档提供了项目中所有主要函数和类的详细说明，包括参数类型、返回值、使用示例和最佳实践。开发者可以参考此文档来正确使用和扩展项目功能。