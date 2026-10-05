"""
ESCNet (ICCV 2025) 白盒维度探针脚本
功能：绕过预训练权重下载，验证网络前向传播的张量维度流转
"""
import torch
from models.ESCNet import ESCNet
from config import load_config

print("1. 读取配置与组装模型...")
# 加载 yaml 配置
config = load_config('config.yaml')
# 核心技巧：强制关闭 pretrained，避免下载庞大的预训练权重，实现秒级启动
model = ESCNet(config, pretrained=False)

print("2. 发射探针 (Batch=2, RGB, 416x416)...")
# 论文设定的标准输入尺寸为 416x416
dummy_input = torch.randn(2, 3, 416, 416)
outputs = model(dummy_input)

print("\n3. 探针捕获成功！解析 MFMM 级联输出的嵌套结构：")
def print_shapes(obj, level=0):
    """递归雷达：用于解析嵌套的 tuple/list 张量树"""
    indent = "  " * level
    if isinstance(obj, torch.Tensor):
        print(f"{indent}👉 Tensor 尺寸: {obj.shape}")
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}📦 包含 {len(obj)} 个元素的 {type(obj).__name__}:")
        for item in obj:
            print_shapes(item, level + 1)

print_shapes(outputs)
