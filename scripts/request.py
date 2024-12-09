import requests
import json


def stream_chat(response):
    for line in response.iter_lines():
        if line:
            info = line.decode('utf-8')
            decoded_text = json.loads(info)
            yield decoded_text


def get_completion(system_prompt, user_prompt):
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {API_KEY}'
    }
    messages={
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    data = json.dumps(messages, ensure_ascii=False)
    response = requests.post(url='http://xx.x.x.xxx:5040/code/api', headers=headers, data=data)

    # 检查响应状态码
    if response.status_code == 200:
        # 处理流式响应
        answer=''
        for value in stream_chat(response):
            for _ in value["response"]:
                answer += _
        return answer
    else:
        return {"error": "Failed to get response", "status_code": response.status_code}
            

if __name__ == '__main__':
    system_prompt = '你是一位经验丰富的Python编程专家和技术顾问，擅长分析Python题目和学生编写的代码。你的任务是理解题目要求和测试样例，分析学生代码，找出潜在的语法或逻辑错误，提供具体的错误位置和修复建议，并用专业且易懂的方式帮助学生改进代码。请以markdown格式返回你的答案。'
    user_prompt = '''## 题目描述：{} ## 测试样例：{} ## 错误代码：{}'''
    response = get_completion(system_prompt, user_prompt)
    print(response)
