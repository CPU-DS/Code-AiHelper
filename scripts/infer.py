from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = 'model/base_mode/Qwen/Qwen2.5-Coder-7B-Instruct'
lora_path = 'model/lora_adapter/Qwen2.5-Coder-7b-Lora/...'  # checkpoint的路径

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

system_prompt = '你是一位经验丰富的Python编程专家和技术顾问，擅长分析Python题目和学生编写的代码。你的任务是理解题目要求和测试样例，分析学生代码，找出潜在的语法或逻辑错误，提供具体的错误位置和修复建议，并用专业且易懂的方式帮助学生改进代码。请以markdown格式返回你的答案。'
user_prompt = '''## 题目描述：{} ## 测试样例：{} ## 错误代码：{}'''
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
