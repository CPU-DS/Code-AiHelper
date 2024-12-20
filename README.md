# Code-AiHelper 

本项目展示了如何基于 transformers、peft 等框架，使用 Qwen2.5-Coder-7B-Instruct 模型在代码数据集上进行 LoRA 微调训练，以实现高效的参数调整和定制化应用。

## 📁 文件结构  
```
|-- README.md                 # 项目说明文档  
|-- data/                     # 训练和测试数据目录
    |-- minHash_de.py         # minHash 去重
    |-- train.json            # 训练数据集  
    |-- test.json             # 测试数据集  
|-- scripts/                  # 脚本目录  
    |-- train.py              # LoRA 微调训练脚本  
    |-- infer.py              # 推理脚本
    |-- api.py                # 模型部署脚本
    |-- request.py            # 调用api脚本
|-- model/                    # 模型存储目录  
    |-- base_model/           # 基础模型存放路径  
    |-- lora_adapter/         # 微调后模型权重存放路径  
```

## 🚀 快速开始  

### ⚙️ 环境安装  
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

### 📊 准备数据
1. 由于数据来自学生对 Python 题目的回答，存在数据相似度过高的问题。实验通过 minHash 方法来评估代码之间的相似度。对于相似度阈值大于 0.7 的代码记录，仅保留一条记录；而对于相似度不高于 0.7 的记录，则全部保留。
```bash
python data/minHash_de.py
```
2. 采用随机采样策略，将保留数据的 40% 作为后续实验的总数据集，按照 9 : 1 的比例将该数据集划分为训练集和测试集。
3. 采用合成数据的方法来构造微调数据集，即通过调用商用大模型的 API ，来生成对训练集中的每一条记录的错误代码的详细分析及修正后的代码版本，将“修改后的代码”会被提交到 CodeRunner 进行评测。若能通过评测，则认为作答正确，该记录得以保留，生成最终的训练集。
4. 将训练数据放置在 data/ 目录下，并确保格式符合模型输入要求。数据格式如下：
```json
{
  "instruction": "你是一位经验丰富的Python编程专家和技术顾问，擅长分析Python题目和学生编写的代码。你的任务是理解题目要求和测试样例，分析学生代码，找出潜在的语法或逻辑错误，提供具体的错误位置和修复建议，并用专业且易懂的方式帮助学生改进代码。请以markdown格式返回你的答案。",
  "input": "## 题目描述：{此处填入题目描述}\n\n## 测试样例：{此处依次写出若干个测试样例}\n\n## 错误代码：{此处给出相应的Python错误代码}",
  "output": "## 分析：{此处给出对Python错误代码的具体分析}\n\n## 修改后的代码：{此处给出具体的修改代码}"
}
```

### 📦 加载模型
使用 modelscope 中的 snapshot_download 下载模型，然后加载到 Transformers 中进行训练：
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

### 🎛️ 开始微调
1. 下载并加载 Qwen2.5-7B-Coder-Instruct 模型
3. 加载数据集，取前 3 条数据进行主观评测
4. 配置 Lora，参数为 r=64, lora_alpha=16, lora_dropout=0.1
5. 使用 SwanLab 记录训练过程，包括超参数、指标和每个 epoch 的模型输出结果
6. 训练 3 个 epoch
   
运行以下命令以开始 LoRA 微调：
```bash
python scripts/train.py
```

注意：首次使用 SwanLab，需要先在官网注册一个账号并在用户设置页面复制 API Key，然后在训练开始提示登录时粘贴，后续无需再次登录。

### 🧪 推理测试
使用微调后的模型进行推理：
```bash
python scripts/infer.py
```

### 🌐 部署模型
设置访问令牌，在命令行中输入：
```bash
export API_KEY="your_api_key_here"
```
在命令行输入以下命令启动api服务：
```bash
python scripts/api.py
```
加载完毕后出现如下信息说明成功。
```
INFO:     Started server process [583718]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5040 (Press CTRL+C to quit)
```
使用 python 中的 requests 库进行调用:
```bash
python scripts/request.py
```

## 🌟 模型效果
微调前、后的模型正确率分别为 55.59% 和 81.76% 。

微调后的模型参数见 [huggingface](https://huggingface.co/monidew/Code-AiHelper)
