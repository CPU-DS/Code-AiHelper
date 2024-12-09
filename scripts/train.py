import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import swanlab
        
        
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 4000
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct", cache_dir="../model/base_model", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("../model/base_model/Qwen/Qwen2.5-Coder-7B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../model/base_model/Qwen/Qwen2.5-Coder-7B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集
train_jsonl_path = "../data/train.jsonl"
train_df = pd.read_json(train_jsonl_path, lines=True)[3:]
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
test_df = pd.read_json(train_jsonl_path, lines=True)[:3]

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph
    lora_dropout=0.1,  # Dropout 比例
)

peft_model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="../model/lora_adapter/Qwen2.5-Coder-7b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=100,
    num_train_epochs=3,
    save_steps=1000,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

class HuanhuanSwanLabCallback(SwanLabCallback):   
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
            
        print("训练开始")
        print("未开始微调，先取3条主观评测：")
        test_text_list = []
        for index, row in test_df[:].iterrows():
            instruction = row["instruction"]
            input_value = row["input"]

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]

            response = predict(messages, peft_model, tokenizer)
            messages.append({"role": "assistant", "content": f"{response}"})
                
            result_text = f"【Q】{messages[1]['content']}\n【LLM】{messages[2]['content']}\n"
            print(result_text)
            
            test_text_list.append(swanlab.Text(result_text, caption=response))

        swanlab.log({"Prediction": test_text_list}, step=0)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # ===================测试阶段======================
        test_text_list = []
        for index, row in test_df.iterrows():
            instruction = row["instruction"]
            input_value = row["input"]
            ground_truth = row["output"]

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]

            response = predict(messages, peft_model, tokenizer)
            messages.append({"role": "assistant", "content": f"{response}"})
            
            if index == 0:
                print("epoch", round(state.epoch), "主观评测：")
                
            result_text = f"【Q】{messages[1]['content']}\n【LLM】{messages[2]['content']}\n【GT】 {ground_truth}"
            print(result_text)
            
            test_text_list.append(swanlab.Text(result_text, caption=response))

        swanlab.log({"Prediction": test_text_list}, step=round(state.epoch))
        
        
swanlab_callback = HuanhuanSwanLabCallback(
    project="Qwen2.5-Coder-LoRA-Law",
    experiment_name="7b",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2.5-Coder-7B-Instruct",
        "system_prompt": "你是一位经验丰富的Python编程专家和技术顾问，擅长分析Python题目和学生编写的代码。你的任务是理解题目要求和测试样例，分析学生代码，找出潜在的语法或逻辑错误，提供具体的错误位置和修复建议，并用专业且易懂的方式帮助学生改进代码。请以markdown格式返回你的答案。",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()
