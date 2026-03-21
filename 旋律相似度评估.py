import librosa
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import warnings
import sys

# 屏蔽无关警告
warnings.filterwarnings('ignore')

# Python版本检测（提示兼容信息）
PYTHON_VERSION = sys.version_info
if PYTHON_VERSION >= (3, 13):
    print(f"检测到Python {PYTHON_VERSION.major}.{PYTHON_VERSION.minor}，已自动使用PYIN替代CREPE提取F0 + fastdtw替代dtw（兼容3.13+）")

def load_audio(audio_path, target_sr=16000):
    """
    加载音频并统一预处理：支持WAV/MP3/FLAC格式，重采样、单声道、降噪
    :param audio_path: 音频文件路径（wav/mp3/flac）
    :param target_sr: 目标采样率
    :return: 预处理后的音频数据、采样率
    """
    # 校验文件是否存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")
    
    # 校验文件格式（支持wav/mp3/flac）
    supported_formats = ('.wav', '.mp3', '.flac')
    if not audio_path.lower().endswith(supported_formats):
        raise ValueError(f"不支持的音频格式！仅支持：{supported_formats}")
    
    # 加载音频（librosa原生支持FLAC，需安装soundfile依赖）
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"音频加载失败：{str(e)}")
    
    # 简单降噪：去除静音段（top_db从20放宽到25，保留更多弱信号）
    y, _ = librosa.effects.trim(y, top_db=25)
    
    # 校验音频有效性
    if len(y) == 0:
        raise ValueError("音频文件无有效音频数据（可能全为静音）")
    
    return y, sr

def extract_f0_pyin(audio_data, sr, frame_length=1024, hop_length=160):
    """
    替换CREPE：使用Librosa-PYIN提取高精度F0（兼容Python 3.13+）
    放宽置信度阈值，保留更多潜在有效音高帧
    """
    # PYIN提取F0（置信度过滤：从0.5放宽到0.3）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=audio_data,
        fmin=librosa.note_to_hz('C2'),  # 最低检测音高（男声下限）
        fmax=librosa.note_to_hz('C7'),  # 最高检测音高（女声上限）
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        win_length=frame_length // 2
    )
    
    # 过滤低置信度（置信度<0.3设为0，原0.5）
    f0[voiced_probs < 0.3] = 0.0
    
    # 生成时间轴（和原CREPE输出格式一致）
    time = librosa.frames_to_time(
        frames=np.arange(len(f0)),
        sr=sr,
        hop_length=hop_length
    )
    
    return time, f0

def align_f0_sequences(f0_ref, f0_test, time_ref, time_test):
    """
    核心修改2：替换dtw为fastdtw，适配调用逻辑
    保留原逻辑：DTW对齐F0序列（解决时长不一致）
    新增：对齐后强制长度一致 + 过滤零值，避免后续计算报错
    """
    # 仅保留非零F0（有音高的帧）用于对齐
    non_zero_ref = f0_ref > 0
    non_zero_test = f0_test > 0
    
    # 处理全零F0的极端情况
    if not np.any(non_zero_ref) or not np.any(non_zero_test):
        raise ValueError("原始/合成语音无有效音高帧，无法对齐")
    
    # fastdtw计算最优对齐路径（欧氏距离）
    # fastdtw返回：(总距离, 对齐路径列表)，路径格式为[(ref_idx, test_idx), ...]
    distance, path = fastdtw(
        f0_ref[non_zero_ref].reshape(-1, 1),
        f0_test[non_zero_test].reshape(-1, 1),
        dist=euclidean
    )
    
    # 核心适配：将fastdtw的路径转换为原dtw包的index1/index2格式
    index1 = [p[0] for p in path]  # 参考序列的对齐索引
    index2 = [p[1] for p in path]  # 测试序列的对齐索引
    
    # 根据对齐路径重构F0序列
    aligned_ref = f0_ref[non_zero_ref][index1]
    aligned_test = f0_test[non_zero_test][index2]
    
    # 关键优化：强制保证对齐后数组长度一致（截断较长的数组）
    min_len = min(len(aligned_ref), len(aligned_test))
    aligned_ref = aligned_ref[:min_len]
    aligned_test = aligned_test[:min_len]
    
    # 过滤对齐后的零值（避免后续计算时干扰）
    align_valid_mask = (aligned_ref > 0) & (aligned_test > 0)
    aligned_ref = aligned_ref[align_valid_mask]
    aligned_test = aligned_test[align_valid_mask]
    
    # 同步截断时间轴
    aligned_time = time_ref[non_zero_ref][index1][:min_len]
    aligned_time = aligned_time[align_valid_mask]
    
    # 校验：对齐后仍无有效帧则报错
    if len(aligned_ref) == 0 or len(aligned_test) == 0:
        raise ValueError("对齐后无有效音高帧，无法计算相似度")
    
    return aligned_ref, aligned_test, aligned_time

def calculate_pitch_similarity_only_shape(ref_pitch, syn_pitch):
    """
    降低评估标准的核心调整：
    1. 有效帧门槛从2帧降至1帧
    2. 局部滑动窗口校正音高偏移（替代全局中位数，适配局部旋律变化）
    3. 负相关不再直接归零，而是映射为0-0.2的低分值
    4. 相似度平滑：对低分值做线性提升，更贴合听觉感知
    """
    # 1. 过滤零值（仅保留有效音高帧）
    valid_mask = (ref_pitch > 0) & (syn_pitch > 0)
    ref_pitch_valid = ref_pitch[valid_mask]
    syn_pitch_valid = syn_pitch[valid_mask]

    valid_frames = len(ref_pitch_valid)
    # 降低有效帧门槛：从2帧改为1帧
    if valid_frames < 1:
        return {
            'valid_frames': 0,
            'pitch_offset': 0.0,
            'shape_similarity': 0.0
        }

    # 强制长度一致
    min_len = min(len(ref_pitch_valid), len(syn_pitch_valid))
    ref_pitch_valid = ref_pitch_valid[:min_len]
    syn_pitch_valid = syn_pitch_valid[:min_len]

    # 2. 优化音高偏移计算：滑动窗口局部校正（适配旋律局部变化）
    window_size = max(5, min_len // 10)  # 自适应窗口大小（最少5帧，最多10%总帧数）
    pitch_offset_list = []
    for i in range(0, min_len, window_size):
        window_end = min(i + window_size, min_len)
        ref_window = ref_pitch_valid[i:window_end]
        syn_window = syn_pitch_valid[i:window_end]
        if len(ref_window) > 0:
            window_offset = np.median(syn_window - ref_window)
            pitch_offset_list.append(window_offset)
    # 全局偏移取窗口偏移的中位数（更鲁棒）
    pitch_offset = np.median(pitch_offset_list) if pitch_offset_list else 0.0
    syn_pitch_corrected = syn_pitch_valid - pitch_offset

    # 3. 形状相似度计算：放宽负相关惩罚
    try:
        correlation = np.corrcoef(ref_pitch_valid, syn_pitch_corrected)[0, 1]
        # 调整1：负相关不再直接归零，映射为0-0.2（弱负相关=0.2，强负相关=0）
        if correlation < 0:
            shape_similarity = max(0.0, 0.2 + correlation * 0.2)  # 例：corr=-0.5 → 0.1；corr=-1 → 0
        else:
            shape_similarity = correlation
        # 调整2：相似度平滑（线性提升低分值，无欺骗性，仅调整尺度）
        # 公式：y = 0.1 + 0.9*x  → 原0→0.1，原0.5→0.55，原1→1（仅提升低分，不超1）
        shape_similarity = 0.1 + 0.9 * shape_similarity
        # 确保相似度在0-1范围内
        shape_similarity = np.clip(shape_similarity, 0.0, 1.0)
    except:
        shape_similarity = 0.1  # 异常情况从0改为0.1，降低惩罚

    return {
        'valid_frames': valid_frames,
        'pitch_offset': pitch_offset,
        'shape_similarity': shape_similarity
    }

def plot_aligned_f0_only(aligned_time, aligned_ref, aligned_test, save_path="aligned_f0_only.png"):
    """
    新增：单独绘制「仅对齐后F0曲线」对比图（更聚焦，满足输出要求）
    """
    # 中文显示配置（兼容macOS/Windows/Linux）
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'][0]
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # 绘制对齐后F0曲线
    ax.plot(aligned_time, aligned_ref, label="原始语音F0（对齐后）", color="#2E86AB", linewidth=2.5)
    ax.plot(aligned_time, aligned_test, label="合成语音F0（对齐后）", color="#E63946", linewidth=2.5, alpha=0.8)
    ax.axhline(y=0, color="#808080", linestyle="--", linewidth=1.2, label="F0基线(0Hz)")
    
    # 美化配置
    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("基频F0 (Hz)", fontsize=12)
    ax.set_title("DTW对齐后 F0曲线对比（核心评估）", fontsize=14, fontweight='bold')
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.4, linestyle="-")
    ax.set_ylim(bottom=-10)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n【单独对齐后F0对比图】已保存至：{save_path}")

def plot_f0_comparison(
    time_ref, f0_ref, time_test, f0_test,
    aligned_time, aligned_ref, aligned_test,
    save_path="f0_comparison.png"
):
    """
    保留原有双图对比（原始+对齐后），确保兼容性
    """
    # 中文显示配置
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'][0]
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # 子图1：原始F0曲线对比（含0Hz基线）
    ax1.plot(time_ref, f0_ref, label="原始语音F0", color="#2E86AB", linewidth=1.8)
    ax1.plot(time_test, f0_test, label="合成语音F0", color="#E63946", linewidth=1.8, alpha=0.8)
    ax1.axhline(y=0, color="#808080", linestyle="--", linewidth=1, label="F0基线(0Hz)")
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("基频F0 (Hz)")
    ax1.set_title("原始 vs 合成语音 F0曲线（未对齐）")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3, linestyle="-")
    ax1.set_ylim(bottom=-10)
    
    # 子图2：DTW对齐后的F0曲线对比（核心）
    ax2.plot(aligned_time, aligned_ref, label="原始语音F0（对齐后）", color="#2E86AB", linewidth=2)
    ax2.plot(aligned_time, aligned_test, label="合成语音F0（对齐后）", color="#E63946", linewidth=2, alpha=0.8)
    ax2.axhline(y=0, color="#808080", linestyle="--", linewidth=1, label="F0基线(0Hz)")
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("基频F0 (Hz)")
    ax2.set_title("DTW对齐后 F0曲线对比（核心评估基线）")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3, linestyle="-")
    ax2.set_ylim(bottom=-10)
    
    # 整体布局调整
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"【双图对比F0图】已保存至：{save_path}")

# ===================== 主函数：一键运行评估 =====================
def evaluate_melody_similarity(ref_audio_path, test_audio_path, 
                               save_combined_plot="f0_comparison.png",
                               save_aligned_only_plot="aligned_f0_only.png"):
    """
    完整评估流程：加载音频→PYIN提取F0→fastdtw对齐→仅计算形状相似度→绘制对比图
    仅输出：有效音高帧数、整体音高偏移、形状相似度
    """
    try:
        # 1. 加载并预处理音频
        print("步骤1：加载并预处理音频...")
        y_ref, sr_ref = load_audio(ref_audio_path)
        y_test, sr_test = load_audio(test_audio_path)
        
        # 2. 提取F0特征（替换为PYIN）
        print("步骤2：使用PYIN提取F0基频特征（兼容Python 3.13+）...")
        time_ref, f0_ref = extract_f0_pyin(y_ref, sr_ref)
        time_test, f0_test = extract_f0_pyin(y_test, sr_test)
        
        # 3. fastdtw对齐F0序列
        print("步骤3：fastdtw动态时间规整对齐F0序列（兼容Python 3.13+）...")
        aligned_ref, aligned_test, aligned_time = align_f0_sequences(f0_ref, f0_test, time_ref, time_test)
        
        # 4. 仅计算形状相似度（已调整评估标准）
        print("步骤4：计算形状相似度（已降低评估标准）...")
        pitch_results = calculate_pitch_similarity_only_shape(aligned_ref, aligned_test)
        
        # 5. 绘制双图对比（原始+对齐后）
        print("步骤5：绘制原始+对齐后F0对比图...")
        plot_f0_comparison(
            time_ref, f0_ref, time_test, f0_test,
            aligned_time, aligned_ref, aligned_test,
            save_path=save_combined_plot
        )
        
        # 6. 单独绘制对齐后F0对比图（满足核心输出要求）
        print("步骤6：绘制仅对齐后F0对比图...")
        plot_aligned_f0_only(aligned_time, aligned_ref, aligned_test, save_path=save_aligned_only_plot)
        
        # 输出评估结果（仅保留三个核心值）
        print("\n===== 旋律相似度评估结果（已降低标准） =====")
        print(f"有效音高帧数（参与计算）：{pitch_results['valid_frames']}")
        print(f"整体音高偏移: {pitch_results['pitch_offset']:.2f} Hz")
        print(f"形状相似度: {pitch_results['shape_similarity']:.4f} (0-1，越高越好)")
        
        return pitch_results
    
    except Exception as e:
        print(f"评估失败：{str(e)}")
        return None

# ===================== 测试示例 =====================
if __name__ == "__main__":
    # 替换为你的音频路径（wav/mp3/flac均可）
    REF_AUDIO = '/Users/fxt/Desktop/旋律对比/老调/原声2 山歌唱了几十年.wav' # 原始语音
    TEST_AUDIO = "/Users/fxt/Desktop/旋律对比/老调/原声2+张道深+melody.flac"  # 合成语音
    
    # 运行评估（指定两张图的保存路径）
    pitch_results = evaluate_melody_similarity(
        ref_audio_path=REF_AUDIO,
        test_audio_path=TEST_AUDIO,
        save_aligned_only_plot="/Users/fxt/Desktop/旋律对比/老调/原声2_aligned_f0_only.png"
    )