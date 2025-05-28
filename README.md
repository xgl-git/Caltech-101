# 🧠 Caltech-101 分类任务实验

该项目使用预训练的 ResNet-18 模型进行 Caltech-101 数据集的图像分类任务，包含从零初始化与微调的对比实验。所有的实验代码与模型权重已上传至 GitHub 和网盘，方便用户进行训练与测试。


---

## 📦 环境依赖

- torch==1.13.0
- torchvision==0.14.0
- numpy==1.21.4 
- matplotlib==3.5.0 
- tensorboard==2.10.0

##📁 数据准备
请访问以下链接手动下载 Caltech-101 数据集：

./data/101_ObjectCategories/

├── accordion/

├── airplanes/

├── ...

在下载并解压数据集后，脚本会自动加载并对图像进行预处理。图像将被缩放为 224x224，并进行标准化。
##🚀 如何训练模型
运行主程序即可开始训练：
```
python 1_finetune.py
```
训练随机初始化模型
```
python 2_scrtach.py
```
##🧪 不同参数的影响

```
python 4_hyperparam_search.py
```

