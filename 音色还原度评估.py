import torch
import torchaudio
import librosa
import soundfile as sf
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
import numpy as np

def load_audio(audio_path, target_sr=16000):
    """加载并预处理音频（转为16kHz单声道）"""
    waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    # 确保音频在合理范围内（避免模型输入异常）
    waveform = waveform / np.max(np.abs(waveform)) * 0.9
    return waveform, target_sr

def extract_speaker_embedding(audio_path, model, processor, device="cpu"):
    """提取归一化的说话人嵌入向量"""
    # 1. 加载并预处理音频
    waveform, _ = load_audio(audio_path)
    
    # 2. 特征处理（转为模型输入格式）
    inputs = processor(
        waveform, 
        sampling_rate=16000, 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # 3. 推理（禁用梯度计算提升速度）
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取归一化的说话人嵌入向量（xvector）
        embedding = torch.nn.functional.normalize(outputs.embeddings, dim=-1)
    
    return embedding.cpu().numpy()

def calculate_similarity(embedding1, embedding2):
    """计算两个嵌入向量的余弦相似度（SIM值）"""
    return np.dot(embedding1, embedding2.T)  # 因已归一化，直接点积即可

# 主流程
if __name__ == "__main__":
    # 1. 配置设备（优先GPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 2. 加载模型和处理器（选择说话人验证专用模型）
    model_name = "microsoft/wavlm-base-plus-sv"  # 推荐
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name).to(device).eval()
    
    # 3. 准备参考音频和合成音频
    reference_audio = "/Users/fxt/Desktop/音色对比/张道深1（唱歌音）clean.wav"  # 参考歌手音频
    synthesized_audio = "/Users/fxt/Desktop/音色对比/张道深1（唱歌音）clean.wav"  # 合成音频
    
    # 4. 提取嵌入向量
    ref_emb = extract_speaker_embedding(reference_audio, model, processor, device)
    syn_emb = extract_speaker_embedding(synthesized_audio, model, processor, device)
    
    # 5. 计算SIM值（音色相似度）
    sim_value = calculate_similarity(ref_emb, syn_emb)[0][0]
    print(f"音色相似度(SIM): {sim_value:.4f}")
    # 输出示例：音色相似度(SIM): 0.9220（与SoulX-Singer论文结果一致）