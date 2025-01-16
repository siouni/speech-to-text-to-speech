# speech-to-text-to-speech
 音声認識->音声合成ソフト

 Whisper（[faster-whisper-onnx](https://github.com/benniekiss/faster-whisper-onnx)）とStyle-Bert-VITS2（[Style-Bert-VITS2](https://github.com/tsukumijima/Style-Bert-VITS2)）を組み合わせて、音声認識からテキストの文字起こしを行い、そのテキストから音声合成を行うプログラムです。

 同様のソフトとして、[ゆかりねっと](http://www.okayulu.moe/)や[ゆかりねっとコネクターNEO（ゆかコネ）](https://nmori.github.io/yncneo-Docs/)などがあります。
 
 現状、あくまで自分用に作っている為、機能的な面や安定性では先程紹介したソフトのほうがいいと思います。

## なんで作っったの？

ゆかりねっとは（詳しく追いかけてないけど）最終更新から結構時間が経っています。
また、ゆかコネについては目的が若干異なります。

また、ブラウザの音声認識機能を使う方式は色々と手間がかかるので、あまり好きになれませんでした。（UDトーク方式はもっと面倒くさい）

ゆかコネは色々できるので、色々やりたい方はそちらをおすすめします。

基本的にはシンプルに音声認識と音声合成を行うソフトを目指しています。

## 今できること
- 音声認識
- 音声合成

## これから追加する機能
- GUI
- 辞書変換

## そのうちやるかも
- 立ち絵表示機能
- モデルマージ・モデル変換（Style-Bert-VITS2に元からあるはず）

## 使い方
現状、Windows11でしか確認してません。
### インストール
GPUを使用する場合、CUDA 12.1が必要です。

初期設定では、WhisperをGPUで処理、音声合成をCPUで処理するように設定していますが、CUDAがなければ自動でWhisperはCPUに切り替わるはずです。

[CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-1-download-archive) こちらのページからインストーラーをダウンロードしてインストールしてください。
（なお、複数バージョンの同時インストールなどについては、自分で調べてください。）

```cli
git clone https://github.com/siouni/speech-to-text-to-speech.git
cd speech-to-text-to-speech
python -m venv venv
venv\Scripts\activate

#CUDAを使用する場合
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

#CUDAを使用しない場合
pip install torch torchvision torchaudio

pip install -r requirements.txt
```
### デバイスIDチェック
```cli
python device_check.py
```
利用できるデバイスとそのIDが一覧で表示されます。
多分MMEのものを選んでおいて問題ないと思います。（違いがよくわからない）

### モデルのダウンロード
音声合成モデルは事前にダウンロードして、modelsフォルダに入れておいてください。

[AivisHub](https://hub.aivis-project.com/)

**注意**：使用できるファイルは現在のところaivmx形式のみです。

コード修正（main.py）
```python
# AIVMXファイルパス
AIVMX_PATH = "models/Anneli.aivmx" # <- ダウンロードしたAIVMXファイルのパス

#デバイスID
MIC_DEVICE = 11 # <- ここにマイクのデバイスID
MONITOR_DEVICE = 13 # <-ここにモニター出力のデバイスID
SPEEKER_DEVICE = 14 # <-ここにスピーカー出力のデバイスID
```
モニター出力とスピーカー出力を分けているのは、[VB-AUDIO](https://vb-audio.com/)の[VB-CABLE Virtual Audio Device](https://vb-audio.com/Cable/index.htm)などの使用を想定しています。

どちらのデバイスにも同じ音声が再生されます。（そのうち設定機能でON/OFFもできるようにします。）

実行
```cli
python main.py
```
初回の実行時に必要なモデルが自動ダウンロードされます。
相当な量になるので、ディスクの空き容量を2～30GB程度は確保しておいてください。

## 注意事項

aivmlib_py310は、[aivmlib](https://github.com/Aivis-Project/aivmlib)が、Python3.11以上を必要としつつ、Windows環境で[Onnx](https://github.com/onnx/onnx)がPython3.11では動作しなかったため、Python3.10向けに少し修正を加えたものとなります。

おそらく動作には問題ないと思いますが、修正がどのように影響しているか不明ですので、流用する場合は自己責任にてお願いします。

また、該当ソースのライセンスについては、**aivmlib**のライセンスに従ってください。
（MITだからそれで行けるはずですが、問題あれば教えて下さい。）
