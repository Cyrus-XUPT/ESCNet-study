# ESCNet (ICCV 2025) 架构拆解与白盒验证

本项目记录了对 ICCV 2025 伪装目标检测顶会论文 **ESCNet (Edge-Semantic Collaborative Network)** 的源码拆解、环境排雷以及物理张量维度的探针验证过程。

## 🧠 核心架构原理解析
传统 COD 方法在深层网络中容易出现“边缘衰减”（Edge Fading）。ESCNet 的物理流水线通过三大核心模块解决此问题：
1. **PVT-v2 引擎**：利用 Transformer 建立全局语义认知。
2. **AETP (自适应边缘-纹理感知器)**：利用**可变形卷积**打造“弹性渔网”，动态对齐物体边缘，生成清晰边缘蓝图。
3. **DSFA & MFMM (动态增强与特征调制)**：提取原图高频细节 (PatchRef)，并结合 Mask、Edge、Laplacian 先验，执行由粗到细 (Coarse-to-Fine) 的级联打磨。

## 🛠️ 环境部署排雷指南 (CPU 轻量版)
官方现代深度学习工程依赖较新，若物理磁盘受限或无 GPU，可参考此方案：
```bash
conda create -n escnet python=3.9 -y
conda activate escnet
# 1. 安装纯 CPU 版 PyTorch，避免下载数 GB 的 CUDA 驱动撑爆磁盘
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
# 2. 安装现代视觉依赖与校验库
pip install einops kornia timm pydantic
sed -i 's#/path/to/datasets/COD/Train#.#g' config.yaml
sed -i 's#/path/to/datasets/COD/Test/NC4K#.#g' config.yaml
sed -i 's#/path/to/weights/pvt_v2_b2.pth#README.md#g' config.yaml
sed -i 's#/path/to/weights/pvt_v2_b4.pth#README.md#g' config.yaml
sed -i 's#/path/to/weights/pvt_v2_b5_22k.pth#README.md#g' config.yaml
