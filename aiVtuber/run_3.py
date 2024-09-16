import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback, AdamW, get_scheduler
from datasets import Dataset
import torch
import requests
from urllib.parse import urlencode
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
import wave
import json
import threading
import pyvts
import aiofiles
import asyncio
import websockets
import speech_recognition as sr
from multiprocessing import Process, Queue, Pool
from torch.utils.data import DataLoader

async def connect_auth(myvts):
    ''' functions to get authenticated '''
    await myvts.connect()
    await myvts.request_authenticate_token()
    await myvts.request_authenticate()
    await myvts.close()

async def trigger(myvts):
    ''' function to trigger hotkey '''
    await myvts.connect()
    await myvts.request_authenticate()
    response_data = await myvts.request(myvts.vts_request.requestHotKeyList())
    print(response_data)
    hotkey_list = []
    for hotkey in response_data["data"]["availableHotkeys"]:
        hotkey_list.append(hotkey["name"])
    send_hotkey_request = myvts.vts_request.requestTriggerHotKey(hotkey_list[0])
    await myvts.request(send_hotkey_request)  # send request to play 'My Animation 1'
    await myvts.close()

# Step 1: Load and prepare the models and tokenizers
gpt2_model_name = 'gpt2-medium'  # Change to gpt2-medium
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# Set pad token as eos_token for GPT-2
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

def text_to_speech(text):
    """使用 HTTP 請求呼叫 GPT-SoVITS API 來將文本轉換為語音"""
    
    # API 的 URL 和端口
    url = "http://192.168.1.106:9880/"

    # 構建請求參數
    params = {
        'refer_wav_path': 'C:/Users/MINAMI/Desktop/aiVtuber/GPT-SoVITS/output/slicer_opt/firefly.wav_0001213120_0001383360.wav',
        'prompt_text': 'I understand. Article 4 of Glamoth military regulations.',
        'prompt_language': 'en',
        'text': text,  # 使用輸入參數
        'text_language': 'en'
    }

    try:
        # 發送 POST 請求到 API 端點
        response = requests.post(url, json=params)
        
        # 確保請求成功
        response.raise_for_status()

        # 音頻資料通常以二進制流的形式返回
        audio_path = "output.wav"
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        print(f"Synthesis successful! Audio saved at: {audio_path}")
        return audio_path

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def load_and_clean_text_files(file_paths):
    """Load and clean text data from specified file paths."""
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [' '.join(word for word in line.strip().lower().split() if word.isalpha()) for line in lines if line.strip()]
            texts.extend(lines)
    return {'text': texts}

def tokenize_function(examples):
    """Tokenize input texts."""
    return gpt2_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def generate_text(model, tokenizer, input_text, max_length=50, num_return_sequences=1):
    """General text generation function."""
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )

    return [tokenizer.decode(o, skip_special_tokens=True) for o in output]

async def change_expression(api, expression_name):
    """修改角色的表情"""
    expression_data = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "changeExpression",
        "messageType": "ExpressionStateRequest",
        "data": {
            "expressionFile": expression_name,
            "active": True
        }
    }
    await api.request(json.dumps(expression_data))

# 播放音频的函数
async def async_play_audio(audio_path):
    """使用 asyncio 進行異步播放音頻文件"""
    chunk = 1024  # 每次读取的音频块大小

    wf = wave.open(audio_path, 'rb')
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
        await asyncio.sleep(0)  # 确保其他任务也有机会运行

    stream.stop_stream()
    stream.close()
    p.terminate()
    
def play_audio_non_blocking(audio_path):
    """在新线程中播放音频"""
    # 这里需要改为调用同步函数，并在新线程中播放音频
    def play_audio_thread(audio_path):
        asyncio.run(async_play_audio(audio_path))

    audio_thread = threading.Thread(target=play_audio_thread, args=(audio_path,))
    audio_thread.daemon = True  # 使用 daemon 线程，确保程序可以在退出时强制关闭线程
    audio_thread.start()

# 初始化語音識別器
recognizer = sr.Recognizer()

def recognize_speech_from_microphone(queue, device_index=None):
    """使用 SpeechRecognition 從麥克風進行語音識別並將結果放入隊列"""
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening... (Multiprocessing)")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        queue.put(text)  # 把结果放入队列中
    except sr.UnknownValueError:
        print("Could not understand audio")
        queue.put("")  # 无法识别时返回空字符串
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        queue.put("")

# 定義音頻播放的工作函數
def play_audio_worker(audio_path):
    asyncio.run(async_play_audio(audio_path))

def generate_response_with_gemini(prompt):
    """使用 Gemini API 生成回應"""
    
    # Google Gemini API URL
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    
    # 你的 API 密鑰
    api_key = "AIzaSyAflYuFDHUCfVSZ4bZoPFxNEGdMIzK39lo"  # 將此處替換為你的 Google API 密鑰
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 構建請求的 payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(f"{api_url}?key={api_key}", headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 檢查是否有 HTTP 錯誤
        
        # 解析回應內容
        response_json = response.json()
        candidates = response_json.get('candidates', [])
        if candidates:
            ai_response = candidates[0]['content']['parts'][0]['text'].strip()
            print(f"Gemini AI: {ai_response}")
            return ai_response
        else:
            print("No valid candidates returned from API")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return None

async def chat_with_ai(api):
    """啟動與AI的互動模式，讓AI讀取並回覆用戶語音輸入"""
    print("Entering interactive mode. Say 'exit' to quit.")

    while True:
        try:
            device_index = 1  # 設置你麥克風的設備索引
            queue = Queue()
            speech_process = Process(target=recognize_speech_from_microphone, args=(queue, device_index))
            speech_process.start()

            # 等待語音識別進程完成
            speech_process.join()  # 使用 join() 來等待進程完成

            user_input = queue.get() if not queue.empty() else ''

            if user_input.lower() == 'exit':
                print("Exiting interactive mode.")
                break

            if user_input:
                # 使用 Gemini AI 生成回應
                ai_response = generate_response_with_gemini(user_input)
                print(f"Gemini AI: {ai_response}")

                # 將回應轉換為語音
                audio_path = text_to_speech(ai_response)
                if audio_path:
                    print(f"Audio response saved at: {audio_path}")
                    play_audio_non_blocking(audio_path)

                    if "happy" in ai_response.lower():
                        await change_expression(api, "Happy")
                    elif "sad" in ai_response.lower():
                        await change_expression(api, "Sad")
        except Exception as e:
            print(f"An error occurred during AI interaction: {e}")

async def play_audio_and_sync(api, audio_path):
    """播放音頻並同步角色嘴巴動畫。"""
    audio = AudioSegment.from_wav(audio_path)
    samples = np.array(audio.get_array_of_samples())

    segment_length = 100  # 分析每段音頻的長度，例如 100 毫秒

    for i in range(0, len(samples), int(audio.frame_rate * segment_length / 1000)):
        segment = samples[i:i + int(audio.frame_rate * segment_length / 1000)]
        amplitude = (np.abs(segment).mean() / 32768) * 10  # 放大振幅以便明顯控制嘴巴動作
        if amplitude > 1.0:
            amplitude = 1.0  # 限制振幅最大值為1

        print(f"Calculated amplitude: {amplitude}")

        # 檢查 WebSocket 是否仍然開啟
        if api.websocket and not api.websocket.closed:
            try:
                await animate_mouth(api, amplitude)  # 這裡的 animate_mouth 也要是異步的
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed. Reconnecting...")
                await api.connect()  # 如果連接關閉，嘗試重新連接
                await animate_mouth(api, amplitude)
        else:
            print("WebSocket is closed. Reconnecting...")
            await api.connect()  # 如果連接關閉，嘗試重新連接
            await animate_mouth(api, amplitude)

        play(audio[i:i + int(audio.frame_rate * segment_length / 1000)])
        await asyncio.sleep(segment_length / 1000)

async def animate_mouth(api, amplitude):
    """
    根據音頻振幅調整角色嘴巴動畫。
    """
    mouth_data = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "mouthAnimation",
        "messageType": "ParameterValueRequest",
        "data": {
            "parameter": "MouthOpen",
            "value": amplitude
        }
    }

    json_payload = json.dumps(mouth_data)
    print("Sending JSON payload:", json_payload)
    
    try:
        response = await api.request(mouth_data)
        print("API response:", response)
    except json.JSONDecodeError as e:
        print("Invalid JSON format:", e)
    except AttributeError as e:
        print("Error sending request:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)

def generate_and_synthesize_text(input_text):
    """Generate text using GPT-2 model and synthesize it using GPT-SoVITS."""
    # Generate text
    generated_texts = generate_text(gpt2_model, gpt2_tokenizer, input_text, max_length=50, num_return_sequences=1)
    
    # For each generated text, synthesize it to speech using GPT-SoVITS
    for text in generated_texts:
        print(f"Generated Text: {text}")

        # 使用 text_to_speech 函數合成語音
        audio_path = text_to_speech(text)
        
        if audio_path:
            print(f"Synthesis successful! Audio saved at: {audio_path}")

# Step 3: Data preparation
file_paths = [
    'C:/Users/MINAMI/Desktop/aiVtuber/data/dialogue_data.txt',
]

data = load_and_clean_text_files(file_paths)
df = pd.DataFrame(data)

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)


# Step 4: Training configuration
data_collator = DataCollatorForLanguageModeling(tokenizer=gpt2_tokenizer, mlm=False)

training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Increase batch size due to smaller model size
    gradient_accumulation_steps=8,  # Adjust accumulation steps to balance memory usage
    num_train_epochs=10,
    learning_rate=2e-5,
    lr_scheduler_type='linear',
    warmup_steps=500,
    load_best_model_at_end=True,
    save_total_limit=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    output_dir='./results',
    save_steps=1000,
    evaluation_strategy='steps',
    save_strategy='steps',
    fp16=torch.cuda.is_available()  # Enable mixed precision training for faster training on GPUs
)

optimizer = AdamW(gpt2_model.parameters(), lr=5e-5)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(tokenized_train_datasets) * training_args.num_train_epochs,
)

class LossLRPerplexityLogger(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.perplexities = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.losses.append(logs["loss"])
                self.perplexities.append(np.exp(logs["loss"]))
            if "learning_rate" in logs:
                self.learning_rates.append(logs["learning_rate"])

loss_lr_perplexity_logger = LossLRPerplexityLogger()

trainer = Trainer(
    model=gpt2_model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_val_datasets,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    callbacks=[loss_lr_perplexity_logger]
)

# Step 5: Main execution logic based on command-line arguments
async def main():
    parser = argparse.ArgumentParser(description="Train or generate text using gpt2-medium model.")
    parser.add_argument('--train', action='store_true', help='Flag to train the model.')
    parser.add_argument('--generate', action='store_true', help='Flag to generate text.')
    parser.add_argument('--input_text', type=str, default="Hello, I'm a language model", help='Input text for text generation.')
    parser.add_argument('--api', action='store_true', help='Flag to run VTube Studio API interaction.')
    parser.add_argument('--vts_control', action='store_true', help='Flag to control VTube Studio interaction.')
    args = parser.parse_args()

    if args.api:
        myvts = pyvts.vts()
        await connect_auth(myvts)  # 使用 await 来执行非同步函数
    else:
        myvts = None  # 在不需要API的情况下将其设为None

    if args.train:
        print("Starting training...")
        trainer.train()
        # Plot training loss over time
        plt.figure(figsize=(10, 5))
        plt.plot(loss_lr_perplexity_logger.losses, label="Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.show()

        # Plot learning rate over time
        plt.figure(figsize=(10, 5))
        plt.plot(loss_lr_perplexity_logger.learning_rates, label="Learning Rate", color='orange')
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Over Time")
        plt.legend()
        plt.show()

        # Plot perplexity over time
        plt.figure(figsize=(10, 5))
        plt.plot(loss_lr_perplexity_logger.perplexities, label="Perplexity", color='green')
        plt.xlabel("Training Steps")
        plt.ylabel("Perplexity")
        plt.title("Perplexity Over Time")
        plt.legend()
        plt.show()

    elif args.generate:
        print("Generating text...")
        input_text_zora = args.input_text
        generate_and_synthesize_text(input_text_zora)
    elif args.api:
        # 保持連接，然後在所有交互完成後再關閉連接
        await chat_with_ai(myvts)

        await myvts.close()  # 完成所有交互後關閉連接

if __name__ == "__main__":
    asyncio.run(main())
