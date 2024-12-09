import os
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from fastapi import FastAPI, Depends, HTTPException, status, Header, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import datetime
import torch
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from peft import PeftModel


API_KEY = os.getenv("API_KEY")

# Clear GPU Memory Function
def torch_gc(cuda_device):
    if torch.cuda.is_available():  
        with torch.cuda.device(cuda_device):  
            torch.cuda.empty_cache()  
            torch.cuda.ipc_collect()           

# Define the request body model
class PromptRequest(BaseModel):
    system_prompt: str
    user_prompt: str

# Extract API key from request header
def api_key_header(authorization: str = Header(..., convert_underscores=False)):
    if authorization.startswith("Bearer "):
        return authorization.split("Bearer ")[1]
    return None

# Verify the API key
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=[{"type":"unauthorized","loc":["header","authorization"],"msg":"Invalid API Key","input":api_key}],
            headers={"WWW-Authenticate": "Bearer"},
        )


# Create a FastAPI application
app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "response": None,
        },
    )
    
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": exc.errors(),
            "status_code": 422,
            "response": None,
        },
    )
        
# Endpoint for handling POST requests
@app.post("/code/api")
async def create_item(request: PromptRequest, api_key_valid: bool = Depends(verify_api_key)):
    # 构建日志信息
    now = datetime.datetime.now()  
    time = now.strftime("%Y-%m-%d %H:%M:%S")  
    log = "[" + time + "]"
    print(log)  
    global model, tokenizer 
    system_prompt = request.system_prompt
    user_prompt = request.user_prompt
    response_data = generate(system_prompt, user_prompt)
    torch_gc("cuda:0")  
    torch_gc("cuda:1")  
    return {
        "error": None,
        "status_code": 200,
        "response": response_data,
    }

@spaces.GPU
def generate(system_prompt: str,user_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,  
        tokenize=False, 
        add_generation_prompt=True,
        max_length=1024
    )
    
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model.generate(model_inputs.input_ids, max_new_tokens=1024, streamer=streamer)
 
    for text in streamer:
        yield text
        
if __name__ == '__main__':
    model_path = 'model/base_mode/Qwen/Qwen2.5-Coder-7B-Instruct'
    lora_path = 'model/lora_adapter/Qwen2.5-Coder-7b-Lora/...'  # checkpoint的路径
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    model.eval() 
    uvicorn.run(app, host='0.0.0.0', port=5040, workers=1) 
