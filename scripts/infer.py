from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = '../model/base_mode/Qwen/Qwen2.5-Coder-7B-Instruct'
lora_path = '../model/lora_adapter/Qwen2.5-Coder-7b-Lora/'  # checkpoint的路径

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

system_prompt = '你是一位经验丰富的Python编程专家和技术顾问，擅长分析Python题目和学生编写的代码。你的任务是理解题目要求和测试样例，分析学生代码，找出潜在的语法或逻辑错误，提供具体的错误位置和修复建议，并用专业且易懂的方式帮助学生改进代码。请以markdown格式返回你的答案。'

user_prompt = '''## 题目描述：
约翰的 N （ 1 ≤N ≤109 ）只奶牛要出发去探索牧场四周的土地，她们将沿着一条路走，一直卡到三岔路口（可以认为所有的路口都是这样的）。这时候，这一群奶牛可能会分成两群，分别沿着接下来的两条路继续走。如果她们再次走到三岔路口，那么仍有可能继续分裂成两群继续走。奶牛的分裂方式十分古怪：如果这一群奶牛可以精确地分成两部分（大小不能为零），这两部分的牛数恰好相差 K（ 1 ≤ K ≤1000），那么在三岔路口牛群就会分裂。否则，牛群不会分裂，她们都将在这里待下去，平静地吃草。请计算，最终将会有多少群奶牛在平静地吃草。输入格式两个整数 N ， K。输出格式一个整数，表示最终奶牛会分成的群数。

## 测试样例：
### Sample_Input_1
6 2
### Sample_Output_1
3
### Sample_Input_2
8 3
### Sample_Output_2
1
### Sample_Input_3
20 4
### Sample_Output_3
7

## 错误代码：
```python
def count_groups(n, k):
    # Base case: if the group size is less than or equal to k, it cannot be split further
    if n <= k:
        return 1
    # Recursive case: split the group into two as evenly as possible
    left = n // 2
    right = n - left
    # If the difference between two groups is less than k, no further split is possible
    if abs(left - right) < k:
        return 1
    # Otherwise, count the number of groups recursively
    return count_groups(left, k) + count_groups(right, k)

# Read input
n, k = map(int, input().split())

# Calculate the result and print
result = count_groups(n, k)
print(result)
```
'''
messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": prompt}
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
