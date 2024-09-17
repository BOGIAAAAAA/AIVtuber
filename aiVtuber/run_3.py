import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback, AdamW, get_scheduler,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
from datasets import Dataset
import torch
import requests
from urllib.parse import urlencode
from pydub import AudioSegment
import pyaudio
import aiohttp
import wave
import json
import threading
from groq import AsyncGroq
import pyvts
import asyncio
import websockets
import speech_recognition as sr
from torch.utils.data import DataLoader
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 GPT-2 模型和分词器
gpt2_model_name = 'gpt2-medium'
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)

# 设置 pad_token 为 eos_token，以避免警告
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# 加载 DistilBERT 模型和分词器
bert_model_name = 'distilbert-base-uncased'
bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_name).to(device)

# 初始化 Groq API
async def get_groq_response(prompt):
    """异步调用 Groq API 生成响应"""
    try:
        client = AsyncGroq()

        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        # 处理异常，记录错误信息
        logger.error(f"Groq API error: {e}")
        return "抱歉，我现在无法生成回复。"


# 异步文本生成函数
async def generate_text(model, tokenizer, input_text, max_length=50, num_return_sequences=1):
    """使用 GPT-2 模型生成文本"""
    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

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
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        return [""]

# 结合 Groq 和 GPT-2 的响应
def combine_responses(groq_response, gpt2_responses):
    """组合 Groq 和 GPT-2 的生成文本"""
    gpt2_response = gpt2_responses[0] if gpt2_responses else ""
    combined_response = f"{groq_response}"
    return combined_response

# 语音合成函数
async def text_to_speech(text):
    """使用 HTTP 请求调用 GPT-SoVITS API 将文本转换为语音"""
    url = "http://192.168.1.106:9880/"  # 请确保此 URL 正确
    params = {
        'refer_wav_path': 'C:/Users/MINAMI/Desktop/aiVtuber/GPT-SoVITS/output/slicer_opt/firefly.wav_0001213120_0001383360.wav',
        'prompt_text': 'I understand. Article 4 of Glamoth military regulations.',
        'prompt_language': 'en',
        'text': text,
        'text_language': 'en'
    }

    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        audio_path = "output.wav"
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during text-to-speech conversion: {e}")
        return None

# 异步播放音频函数
async def async_play_audio(audio_path):
    """异步播放音频文件，确保正确释放资源"""
    p = None
    wf = None
    try:
        chunk = 1024
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
            await asyncio.sleep(0)  # 确保其他任务有机会运行

        # 关闭音频流
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        logger.error(f"Error during audio playback: {e}")
    finally:
        if p:
            p.terminate()
        if wf:
            wf.close()

# 语音识别函数
async def recognize_speech_from_microphone(device_index=None):
    """使用 SpeechRecognition 从麦克风进行语音识别"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(device_index=device_index) as source:
            recognizer.adjust_for_ambient_noise(source)
            logger.info("Listening... (Async)")
            # 使用线程池执行阻塞的 listen 函数
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, recognizer.listen, source, 5, 5)

        # 使用线程池执行阻塞的 recognize_google 函数
        text = await loop.run_in_executor(None, recognizer.recognize_google, audio)
        logger.info(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        logger.warning("Could not understand audio")
        return None
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        logger.error(f"Error during speech recognition: {e}")
        return None

# VTube Studio 认证连接
async def connect_auth(myvts):
    """进行 VTube Studio 认证"""
    await myvts.connect()
    await myvts.request_authenticate_token()
    await myvts.request_authenticate()

# 修改角色表情
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

# 主交互函数
async def chat_with_ai(myvts=None, device_index=None):
    """启动与 AI 的交互模式，AI 读取并回复用户语音输入"""
    logger.info("Entering interactive mode. Say 'exit' to quit.")
    while True:
        try:
            user_input = await recognize_speech_from_microphone(device_index)
            if user_input and user_input.lower() == 'exit':
                logger.info("Exiting interactive mode.")
                break

            if user_input:
                # 生成 AI 回复并合成语音
                generated_text, audio_path = await handle_generation_and_synthesis(user_input)
                if generated_text:
                    logger.info(f"AI Response: {generated_text}")
                    # 已在 handle_generation_and_synthesis 中处理，无需重复处理
        except Exception as e:
            logger.error(f"An error occurred during AI interaction: {e}")

# 处理文本生成和语音合成
async def handle_generation_and_synthesis(input_text):
    """串行处理文本生成与语音合成，并播放生成的语音"""
    try:
        # 1. 使用 GPT-2 生成文本
        gpt2_responses = await generate_text(gpt2_model, gpt2_tokenizer, input_text)

        # 2. 使用 Groq API 生成文本
        groq_response = await get_groq_response(input_text)

        # 3. 结合两者的回复
        combined_response = combine_responses(groq_response, gpt2_responses)
        logger.info(f"Combined Response: {combined_response}")

        # 4. 语音合成
        audio_path = await text_to_speech(combined_response)

        # 5. 播放生成的音频
        if audio_path:
            logger.info(f"Audio saved at: {audio_path}")
            await async_play_audio(audio_path)

        return combined_response, audio_path

    except Exception as e:
        logger.error(f"Error during generation and synthesis: {e}")
        return "", None

# 数据加载和清理
def load_and_clean_text_files(file_paths):
    """加载并清理指定路径的文本数据"""
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [
                ' '.join(word for word in line.strip().lower().split() if word.isalpha())
                for line in lines if line.strip()
            ]
            texts.extend(lines)
    return {'text': texts}

def tokenize_function(examples):
    """对输入文本进行分词"""
    return gpt2_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# 自定义 TrainerCallback，用于记录损失、学习率和困惑度
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

# 主函数
async def main():
    parser = argparse.ArgumentParser(description="Train or generate text using gpt2-medium model.")
    parser.add_argument('--train', action='store_true', help='Flag to train the model.')
    parser.add_argument('--generate', action='store_true', help='Flag to generate text.')
    parser.add_argument('--classify', action='store_true', help='Flag to classify text.')
    parser.add_argument('--input_text', type=str, default="Hello, I'm a language model", help='Input text for text generation.')
    parser.add_argument('--api', action='store_true', help='Flag to run VTube Studio API interaction.')
    parser.add_argument('--vts_control', action='store_true', help='Flag to control VTube Studio interaction.')
    parser.add_argument('--device_index', type=int, default=None, help='Microphone device index for speech recognition.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the training data files.')
    parser.add_argument('--model_save_path', type=str, default='saved_model', help='Path to save or load the model.')
    args = parser.parse_args()

    myvts = None
    if args.api or args.vts_control:
        logger.info("Connecting to VTube Studio API...")
        myvts = pyvts.vts()
        await connect_auth(myvts)

    if args.train:
        logger.info("Starting training...")

        # 数据准备
        file_paths = [
            os.path.join(args.data_dir, 'dialogue_data_proficiency.txt'),
            os.path.join(args.data_dir, 'dialogue_data.txt'),
            os.path.join(args.data_dir, 'personality_data.txt'),
            os.path.join(args.data_dir, 'zora.txt')
        ]
        data = load_and_clean_text_files(file_paths)
        df = pd.DataFrame(data)

        # 数据集拆分
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
        tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)

        # 训练配置
        data_collator = DataCollatorForLanguageModeling(tokenizer=gpt2_tokenizer, mlm=False)
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
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
            fp16=torch.cuda.is_available()
        )

        optimizer = AdamW(gpt2_model.parameters(), lr=5e-5)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(tokenized_train_datasets) * training_args.num_train_epochs,
        )

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

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model(args.model_save_path)
        gpt2_tokenizer.save_pretrained(args.model_save_path)

        # 绘制训练损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(loss_lr_perplexity_logger.losses, label="Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.show()

        # 绘制学习率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(loss_lr_perplexity_logger.learning_rates, label="Learning Rate", color='orange')
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Over Time")
        plt.legend()
        plt.show()

        # 绘制困惑度曲线
        plt.figure(figsize=(10, 5))
        plt.plot(loss_lr_perplexity_logger.perplexities, label="Perplexity", color='green')
        plt.xlabel("Training Steps")
        plt.ylabel("Perplexity")
        plt.title("Perplexity Over Time")
        plt.legend()
        plt.show()

    elif args.generate:
        logger.info("Generating text...")
        input_text_zora = args.input_text

        # 加载已保存的模型
        gpt2_model = GPT2LMHeadModel.from_pretrained(args.model_save_path).to(device)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.model_save_path)

        # 使用 GPT-2 生成回复
        gpt2_responses = await generate_text(gpt2_model, gpt2_tokenizer, input_text_zora, max_length=150)

        # 使用 Groq API 生成回复
        groq_response = await get_groq_response(input_text_zora)

        # 结合两个回复
        combined_response = combine_responses(groq_response, gpt2_responses)
        logger.info(combined_response)

    elif args.classify:
        logger.info("Classifying text...")
        input_text_zora = args.input_text

        # 使用 DistilBERT 进行文本分类
        inputs = bert_tokenizer(input_text_zora, return_tensors="pt").to(device)
        outputs = bert_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1)

        # 根据预测类别改变 Vtuber 表情
        if predicted_class.item() == 0:
            logger.info("预测的情感为负面。")
            if myvts:
                await change_expression(myvts, "Sad")
        else:
            logger.info("预测的情感为正面。")
            if myvts:
                await change_expression(myvts, "Happy")

    elif args.api or args.vts_control:
        # 启动与 AI 的交互模式
        await chat_with_ai(myvts, device_index=args.device_index)
        # 完成所有交互后关闭连接
        if myvts:
            await myvts.close()
    else:
        # 默认启动与 AI 的交互模式
        await chat_with_ai(myvts, device_index=args.device_index)

if __name__ == "__main__":
    asyncio.run(main())