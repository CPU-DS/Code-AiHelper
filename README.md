# Code-AiHelper 

本项目展示了如何基于 transformers、peft 等框架，使用 Qwen2.5-7B-Instruct 模型在数据集上进行Lora微调训练，同时使用 SwanLab 监控训练过程与评估模型效果，以实现高效的参数调整和定制化应用。

## 目录  
- [文件结构](#文件结构)  
- [快速开始](#快速开始)  
  - [环境安装](#环境安装)  
  - [准备数据](#准备数据)
  - [加载模型](#加载模型)  
  - [开始微调](#开始微调)  
  - [推理测试](#推理测试)  
- [模型效果](#模型效果)  
- [许可证](#许可证)    

## 文件结构  
```
|-- README.md                 # 项目说明文档  
|-- data/                     # 训练和验证数据目录  
    |-- train.json            # 训练数据集  
    |-- val.json              # 验证数据集  
|-- scripts/                  # 项目脚本目录  
    |-- train.py              # LoRA 微调训练脚本  
    |-- infer.py              # 推理脚本  
|-- model/                    # 模型存储目录  
    |-- base_model/           # 基础模型存放路径  
    |-- lora_adapter/         # 微调后模型权重存放路径  
|-- docs/                     # 文档存储目录  
    |-- experiment_report.md  # 实验报告  
|-- examples/                 # 示例存储目录  
    |-- sample_input.txt      # 输入示例  
    |-- sample_output.txt     # 输出示例  
```

## 快速开始  

### 环境安装  
实验基础环境如下：
```
Ubuntu 22.04
python 3.10
cuda 12.2
pytorch 2.4.0
```
使用以下命令安装依赖：  
```bash
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.20.0
pip install transformers==4.46.2
pip install accelerate==0.34.2
pip install peft==0.11.1
pip install datasets==2.21.0
pip install swanlab==0.3.23
```

### 准备数据
将训练数据放置在 data/ 目录下，并确保格式符合模型输入要求。数据格式如下：
```json
{
  "instruction": "你是一位经验丰富的Python编程专家和技术顾问，擅长分析Python题目和学生编写的代码。你的任务是理解题目要求和测试样例，分析学生代码，找出潜在的语法或逻辑错误，提供具体的错误位置和修复建议，并用专业且易懂的方式帮助学生改进代码。请以markdown格式返回你的答案。",
  "input": "## 题目描述：{此处填入题目描述}\n\n## 测试样例：{此处依次写出若干个测试样例}\n\n## 错误代码：{此处给出相应的Python错误代码}",
  "output": "## 分析：{此处给出对Python错误代码的具体分析}\n\n## 修改后的代码：{此处给出具体的修改代码}"
}
```
### 加载模型
使用modelscope中的snapshot_download下载模型，然后加载到 Transformers 中进行训练：
```python
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct", cache_dir="/root/autodl-tmp", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("base_model/Qwen/Qwen2.5-Coder-7B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("base_model/Qwen/Qwen2.5-Coder-7B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
```

### 开始微调
1. 下载并加载Qwen2.5-7B-Coder-Instruct模型
3. 加载数据集，取前5000条数据参与训练，5条数据进行主观评测
4. 配置Lora，参数为r=64, lora_alpha=16, lora_dropout=0.1
5. 使用SwanLab记录训练过程，包括超参数、指标和每个epoch的模型输出结果
6. 训练3个epoch
   
运行以下命令以开始 LoRA 微调：
```bash
python scripts/train.py
```

注意：首次使用SwanLab，需要先在官网注册一个账号，然后在用户设置页面复制你的API Key，然后在训练开始提示登录时粘贴即可，后续无需再次登录。

### 推理测试
使用微调后的模型进行推理：
```bash
python scripts/infer.py
```

## 模型效果
在微调后的模型上，在纠正代码错误上的准确率提升了26.17%，达到了81.76%。

## 许可证
本项目基于 MIT License 发布。详情请参阅 LICENSE。

## 联系方式
如果你对本项目有任何疑问或建议，欢迎通过以下方式联系：
- Email: your_email@example.com
- GitHub Issues: Issues 页面
欢迎 Star 和 Fork！😊

