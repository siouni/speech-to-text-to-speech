import asyncio
import sounddevice as sd
import numpy as np
import time
import soundfile as sf
import librosa
from pathlib import Path
from io import BytesIO
import time
import asyncio

from faster_whisper import WhisperModel

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import onnx_bert_models
from style_bert_vits2.logging import logger as style_bert_vits2_logger
from style_bert_vits2.tts_model import TTSModel

import aivmlib_py310.aivmlib as aivmlib

# AIVMXファイルパス
AIVMX_PATH = "models/Anneli.aivmx" # <- ダウンロードしたAIVMXファイルのパス

#デバイスID
MIC_DEVICE = 11 # <- ここにマイクのデバイスID
MONITOR_DEVICE = 13 # <-ここにモニター出力のデバイスID
SPEEKER_DEVICE = 14 # <-ここにスピーカー出力のデバイスID

# カスタムキャッシュディレクトリを指定
custom_cache_dir = "models/"

device = "cpu"

# Whisperモデルのロード
# whisper_model = whisper.load_model("turbo", download_root=custom_cache_dir)  # 必要に応じてモデルサイズを変更可能
whisper_model = WhisperModel("turbo", download_root=custom_cache_dir)
# asteroid_model = BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean", device=device, cache_dir=custom_cache_dir)

aivmx_path = Path(AIVMX_PATH)

onnx_providers = [("CPUExecutionProvider", {
    "arena_extend_strategy": "kSameAsRequested",
})]

# 録音設定
SAMPLERATE = 44100  # サンプリングレート
CHANNELS = 1  # モノラル録音
BLOCKSIZE = 1024  # フレームサイズ
THRESHOLD = 0.01  # 録音開始のボリューム閾値
SILENCE_DURATION = 2.0  # 無音とみなす継続時間（秒）
TARGET_VOLUME = 0.1  # 正規化後のターゲットボリューム（RMS値）
# 最大音量閾値を設定（例: RMS値が0.5以上の場合、無視する）
MAX_VOLUME_THRESHOLD = 0.5

# 信頼スコアのしきい値
AVG_LOGPROB_THRESHOLD = -1.0
NO_SPEECH_PROB_THRESHOLD = 0.6

# 状態管理
recording = False  # 録音中かどうか
audio_buffer = []  # 録音された音声データ
silence_counter = 0  # 無音フレームのカウンタ

# 処理用のキュー
audio_queue = asyncio.Queue()
segment_queue = asyncio.Queue()
sound_queue = asyncio.Queue()

# モデルとトークナイザーをロード
onnx_bert_models.load_model(
    language=Languages.JP,
    pretrained_model_name_or_path="tsukumijima/deberta-v2-large-japanese-char-wwm-onnx",
    onnx_providers=onnx_providers,
    cache_dir=str(custom_cache_dir),
    revision="d701ec67708287b20d2063270f6b535e6eed09ab",
)
onnx_bert_models.load_tokenizer(
    language=Languages.JP,
    pretrained_model_name_or_path="tsukumijima/deberta-v2-large-japanese-char-wwm-onnx",
    cache_dir=str(custom_cache_dir),
    revision="d701ec67708287b20d2063270f6b535e6eed09ab",
)
print("bert_model, bert_tokenizer loaded")

with open(aivmx_path, mode="rb") as file:
    aivm_metadata = aivmlib.read_aivmx_metadata(file)

hyper_parameters = HyperParameters.model_validate(
    aivm_metadata.hyper_parameters.model_dump()
)

style_vectors = np.load(BytesIO(aivm_metadata.style_vectors))

aivm_manifest = aivm_metadata.manifest

print("aivm metadata loaded:", aivm_manifest.uuid)

tts_model = TTSModel(
    # 音声合成モデルのパスとして、AIVMX ファイル (ONNX 互換) のパスを指定
    model_path=aivmx_path,
    # config_path とあるが、HyperParameters の Pydantic モデルを直接指定できる
    config_path=hyper_parameters,
    # style_vec_path とあるが、style_vectors の NDArray を直接指定できる
    style_vec_path=style_vectors,
    # ONNX 推論で利用する ExecutionProvider を指定
    onnx_providers=onnx_providers,
)

tts_model.load()

# 録音コールバック
def audio_callback(indata, frames, time, status):
    global recording, audio_buffer, silence_counter

    if status:
        print(f"Stream status: {status}")

    # 入力音声の音量を計算
    rms = np.sqrt(np.mean(indata**2))
    if recording:
        audio_buffer.append(indata.copy())
        if rms < THRESHOLD:
            silence_counter += 1
            if silence_counter * BLOCKSIZE / SAMPLERATE > SILENCE_DURATION:
                recording = False
                print("Silence detected. Stopping recording.")
        else:
            silence_counter = 0
    else:
        if rms >= THRESHOLD:
            recording = True
            audio_buffer = [indata.copy()]
            silence_counter = 0
            print("Sound detected. Starting recording.")

# 非同期で音声処理
async def process_audio_task():
    while True:
        if not audio_queue.empty():
            start_time = time.time()
            audio_data = await audio_queue.get()
            print("Processing audio...")

            # STT処理
            # ここでfaster-whisperを使用したSTT処理を行います
            await process_whisper(audio_data)

            exec_time = time.time() - start_time
            print("Process audio exec time:", exec_time)

        await asyncio.sleep(0.1)  # キューが空の場合の待機

async def process_segment_task():
    while True:
        if not segment_queue.empty():
            start_time = time.time()
            text, delay = await segment_queue.get()
            print("Processing segment...")

            # TTS処理
            # ここでStyleBertVITS2を使用したSTT処理を行います
            await process_tts(text, delay)

            exec_time = time.time() - start_time
            print("Process segment exec time:", exec_time)

        await asyncio.sleep(0.1)  # キューが空の場合の待機

async def process_sound_task():
    while True:
        if not sound_queue.empty():
            start_time = time.time()
            audio, sr, delay = await sound_queue.get()
            print("Processing sound...")

            # サウンド処理
            # ここでサウンド処理を行います
            await asyncio.sleep(delay) # セグメント間の「間」を取る。
            await process_sound(audio, sr)

            exec_time = time.time() - start_time
            print("Process sound exec time:", exec_time)

        await asyncio.sleep(0.1)  # キューが空の場合の待機

# メイン処理
async def main():
    global audio_buffer

    # 音声入力ストリームを非同期で監視
    print("Listening...")
    with sd.InputStream(device=MIC_DEVICE, samplerate=SAMPLERATE, channels=CHANNELS, callback=audio_callback, blocksize=BLOCKSIZE, dtype='float32'):
        try:
            # 録音処理と並行して音声処理を実行
            _ = asyncio.create_task(process_audio_task())
            # 録音処理と並行してセグメント処理を実行
            _ = asyncio.create_task(process_segment_task())
            # 録音処理と並行してサウンド処理を実行
            _ = asyncio.create_task(process_sound_task())

            while True:
                if not recording and len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer, axis=0).flatten()
                    await audio_queue.put(audio_data)  # キューにデータを追加
                    audio_buffer = []  # バッファをリセット
                    print("Listening...")

                await asyncio.sleep(0.1)  # メインループの待機
        except KeyboardInterrupt:
            print("Stopped by user.")

# STT処理関数
async def process_whisper(audio_data):
    """音声データを文字列セグメントに変換するSTT処理"""
    # Whisperで文字起こし
    print("Transcribing the loudest source with Whisper...")
    resampled_audio = librosa.resample(audio_data, orig_sr=SAMPLERATE, target_sr=16000)

    segments, _ = whisper_model.transcribe(resampled_audio, language="ja")
    # フィルタリング処理
    filtered_segments = []
    for segment in segments:
        if segment.avg_logprob >= AVG_LOGPROB_THRESHOLD and segment.no_speech_prob <= NO_SPEECH_PROB_THRESHOLD:
            filtered_segments.append(segment)
    
    text = ""
    delay_end = 0
    for segment in filtered_segments:
        print("[avg_logprob: %.2f, no_speech_prob: %.2f] %s" % (segment.avg_logprob, segment.no_speech_prob, segment.text))
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        delay = delay_end - segment.start
        delay_end = segment.end
        await segment_queue.put((segment.text, delay))
        text += segment.text
    print("Transcription:", text)

# TTS処理関数
async def process_tts(text, delay):
    """文字列を音声に変換するTTS処理"""
    print("Generateing Voice:", text)
    sr, audio = tts_model.infer(
        text=text,
        style=aivm_manifest.speakers[0].styles[0].name,
    )
    print("Generated Voice")
    await sound_queue.put((audio, sr, delay))

async def play_on_device(audio, sr, device_id):
    """指定したデバイスで音声を再生"""
    sd.play(audio, samplerate=sr, device=device_id)
    sd.wait()

# サウンド処理関数
async def process_sound(audio, sr):
    """音声を再生するサウンド処理"""
    print("Playing Sound...")
    # 再生タスクを作成
    tasks = [
        play_on_device(audio, sr, MONITOR_DEVICE),
        play_on_device(audio, sr, SPEEKER_DEVICE),
    ]
    # 並行して再生タスクを実行
    await asyncio.gather(*tasks)

# 実行
asyncio.run(main())
