# 安装PyTorch（根据CUDA版本选择，CPU版也可）
pip install torch torchaudio

# 安装FunASR和modelscope
pip install funasr
pip install modelscope
from funasr import AutoModel
import os

# 加载Paraformer中文模型（首次运行会自动下载，约1.3GB）
model = AutoModel.from_pretrained(
    "paraformer-zh",  # 中文模型
    model_revision="v2.0.4",
    disable_update=True
)

# 识别单个音频文件
def recognize_audio(audio_path):
    result = model.generate(
        input=audio_path,
        batch_size=1,
        disable_pbar=True
    )
    # 返回识别文本
    return result[0]["text"]

# 批量识别示例
audio_folder = "./synthesized_audio"
results = {}
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):
        path = os.path.join(audio_folder, filename)
        text = recognize_audio(path)
        results[filename] = text
        print(f"{filename}: {text}")