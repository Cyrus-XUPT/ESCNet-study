import torch
from models.ESCNet import ESCNet
from config import load_config

print("1. 读取配置与组装模型...")
config = load_config('config.yaml')
model = ESCNet(config, pretrained=False)

print("2. 发射 416x416 探针...")
dummy_input = torch.randn(2, 3, 416, 416)
outputs = model(dummy_input)

print("\n3. 探针捕获成功！解析嵌套输出结构：")
def print_shapes(obj, level=0):
    indent = "  " * level
    if isinstance(obj, torch.Tensor):
        print(f"{indent}👉 Tensor 尺寸: {obj.shape}")
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}📦 包含 {len(obj)} 个元素的 {type(obj).__name__}:")
        for item in obj:
            print_shapes(item, level + 1)

print_shapes(outputs)
