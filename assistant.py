import os
import queue
import sounddevice as sd
import vosk
import sys
import json
import time
import pyttsx3
from llama_cpp import Llama





# Инициализация модели Vosk
s = time.time()
vosk_model = vosk.Model("vosk-model-ru-0.42/vosk-model-small-ru-0.22")
samplerate = 16000
device = 1

q = queue.Queue()

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им"
name = "сайга"


Llama_model = Llama(
    model_path="model-f16.gguf",
    n_ctx=8192,
    n_gpu_layers=20,  # Число слоёв на GPU (экспериментируйте)
    n_threads=20,       # Ядер CPU (для R9 3900X)
    n_parts=1,
    offload_kqv=True,
    verbose=True,
)
thred_speak = ""

def interact(
    message="",
    top_k=30,
    top_p=0.9,
    temperature=0.3 ,
    repeat_penalty=1.1
):
    global thred_speak
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_message = "User: " + message
    messages.append({"role": "user", "content": user_message})
    mess = ""
    for part in Llama_model.create_chat_completion(
        messages,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stream=True,
    ):
        delta = part["choices"][0]["delta"]
        if "content" in delta:
            mess+= delta["content"]
            print(delta["content"], end="", flush=True)

    #     if len(m := mess.split(" "))==4:
    #         thred_speak = threading.Thread(target=speck, args=[[m[0], m[1], m[2]]]).start()
    #         mess = m[3]
    # #threading.Thread(target=speck,args=[mess]).start()


def speck(text):
    global thred_speak
    for somthing in text:
        sintz = pyttsx3.init()
        voices = sintz.getProperty("voices")
        sintz.setProperty("voice", voices[3].id)
        sintz.setProperty('rate', 15, 0)
        sintz.setProperty('volume', 0.9)
        sintz.say(somthing)
        sintz.runAndWait()
    thred_speak.join()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def recognition():
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device,
                           dtype='int16', channels=1, callback=callback):

        rec = vosk.KaldiRecognizer(vosk_model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result      = json.loads(rec.Result())
                if name in result["text"]:
                    return (result["text"])
                else:
                    return "Это пустое сообщение на него не отвечай"

def weather():
    return "20 градусов цельсия"

if __name__ == "__main__":
    while True:
        task = recognition()
        if "погода" in task:
            weather = "20 градусов цельсия"
            interact(task+"Дальше системная информация про нее не пиши пользователю, она нужна тебе для ответа на этот вопрос"+weather)

