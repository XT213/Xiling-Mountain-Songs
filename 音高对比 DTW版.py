"""
F0 RMSE 评估脚本（DTW对齐版）- 批量处理版
在脚本末尾直接配置音频路径对，自动计算并绘图。
校正后的 RMSE 默认使用 DTW 对齐，排除节奏差异干扰。
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_f0(audio_path, sr=16000, hop_length=160,
               fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')):
    y, sr = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin, fmax=fmax,
                                                 sr=sr, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced_flag = f0 > 0
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    return f0, voiced_flag, times

def hz_to_cents(f0_hz, ref_hz=440.0):
    f0_hz = np.where(f0_hz > 0, f0_hz, np.nan)
    return 1200 * np.log2(f0_hz / ref_hz)

def align_to_min_len(*arrays):
    min_len = min(len(arr) for arr in arrays)
    return [arr[:min_len] for arr in arrays]

def estimate_global_shift_cents(f0_ref, f0_syn, voiced_ref, voiced_syn):
    f0_ref, f0_syn, voiced_ref, voiced_syn = align_to_min_len(f0_ref, f0_syn,
                                                               voiced_ref, voiced_syn)
    mask = voiced_ref & voiced_syn
    if np.sum(mask) < 5:
        return 0.0
    ref_cents = hz_to_cents(f0_ref[mask])
    syn_cents = hz_to_cents(f0_syn[mask])
    diff_cents = syn_cents - ref_cents
    return np.median(diff_cents)

def apply_shift_cents(f0_hz, shift_cents):
    f0_cents = hz_to_cents(f0_hz)
    f0_cents_shifted = f0_cents - shift_cents
    f0_shifted_hz = 440.0 * 2**(f0_cents_shifted / 1200)
    f0_shifted_hz[f0_hz == 0] = 0
    return f0_shifted_hz

def compute_rmse_cents(f0_ref_hz, f0_syn_hz, voiced_ref, voiced_syn):
    """
    校正前的 RMSE：无 DTW 对齐，直接逐帧对比（仅浊音帧）。
    """
    f0_ref_hz, f0_syn_hz, voiced_ref, voiced_syn = align_to_min_len(f0_ref_hz, f0_syn_hz,
                                                                     voiced_ref, voiced_syn)
    mask = voiced_ref & voiced_syn
    if np.sum(mask) == 0:
        return np.nan
    ref_cents = hz_to_cents(f0_ref_hz[mask])
    syn_cents = hz_to_cents(f0_syn_hz[mask])
    mse = np.mean((syn_cents - ref_cents)**2)
    return np.sqrt(mse)

def compute_rmse_cents_dtw(f0_ref_hz, f0_syn_hz, voiced_ref, voiced_syn):
    """
    校正后的 RMSE：DTW 对齐完整序列后，在浊音帧上计算 RMSE。
    """
    f0_ref, f0_syn, v_ref, v_syn = align_to_min_len(f0_ref_hz, f0_syn_hz, voiced_ref, voiced_syn)

    ref_feat = np.atleast_2d(f0_ref)
    syn_feat = np.atleast_2d(f0_syn)

    D, wp = librosa.sequence.dtw(ref_feat, syn_feat)

    ref_indices = wp[:, 0]
    syn_indices = wp[:, 1]
    both_voiced = v_ref[ref_indices] & v_syn[syn_indices]

    if np.sum(both_voiced) == 0:
        return np.nan

    ref_aligned = f0_ref[ref_indices[both_voiced]]
    syn_aligned = f0_syn[syn_indices[both_voiced]]

    ref_cents = hz_to_cents(ref_aligned)
    syn_cents = hz_to_cents(syn_aligned)

    valid = ~np.isnan(ref_cents) & ~np.isnan(syn_cents)
    if np.sum(valid) == 0:
        return np.nan

    mse = np.mean((syn_cents[valid] - ref_cents[valid])**2)
    return np.sqrt(mse)

def interpolate_nan(y):
    n = len(y)
    x = np.arange(n)
    mask = np.isnan(y)
    if np.all(mask):
        return y
    if np.any(mask):
        y[mask] = np.interp(x[mask], x[~mask], y[~mask])
    return y

def plot_f0_comparison(ref_times, f0_ref, syn_times, f0_syn,
                       voiced_ref=None, voiced_syn=None,
                       f0_syn_corrected=None,
                       save_path=None):
    plt.figure(figsize=(10, 5))

    def prepare_clean_series(times, f0, voiced):
        clean_f0 = np.where(voiced, f0, np.nan)
        return times, clean_f0

    if voiced_ref is not None:
        t_ref, f_ref_clean = prepare_clean_series(ref_times, f0_ref, voiced_ref)
    else:
        f_ref_clean = np.where(f0_ref > 0, f0_ref, np.nan)
        t_ref = ref_times
    plt.plot(t_ref, f_ref_clean, color='blue', linewidth=0.8, alpha=0.6, label='参考音频 F0')
    f_ref_interp = interpolate_nan(f_ref_clean.copy())
    if len(f_ref_interp) > 31:
        trend_ref = savgol_filter(f_ref_interp, window_length=31, polyorder=2)
        plt.plot(t_ref, trend_ref, color='darkblue', linestyle='--', linewidth=1.2,
                 label='参考音频趋势线')

    if voiced_syn is not None:
        t_syn, f_syn_clean = prepare_clean_series(syn_times, f0_syn, voiced_syn)
    else:
        f_syn_clean = np.where(f0_syn > 0, f0_syn, np.nan)
        t_syn = syn_times
    plt.plot(t_syn, f_syn_clean, color='red', linewidth=0.8, alpha=0.6, label='合成音频 F0')
    f_syn_interp = interpolate_nan(f_syn_clean.copy())
    if len(f_syn_interp) > 31:
        trend_syn = savgol_filter(f_syn_interp, window_length=31, polyorder=2)
        plt.plot(t_syn, trend_syn, color='darkred', linestyle='--', linewidth=1.2,
                 label='合成音频趋势线')

    if f0_syn_corrected is not None:
        f_syn_corr_clean = np.where(f0_syn_corrected > 0, f0_syn_corrected, np.nan)
        plt.plot(t_syn, f_syn_corr_clean, color='darkorange', linewidth=1.2, alpha=0.85,
                 linestyle='--', label='音高校正后合成 F0')

    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.title('F0 对比：参考音频 vs. 合成音频')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"音高对比图已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def main(ref_audio_path, syn_audio_path, sr=16000, plot=True, save_plot_path=None):
    f0_ref, voiced_ref, times_ref = extract_f0(ref_audio_path, sr=sr)
    f0_syn, voiced_syn, times_syn = extract_f0(syn_audio_path, sr=sr)

    # 估计整体移调
    global_shift = estimate_global_shift_cents(f0_ref, f0_syn, voiced_ref, voiced_syn)
    print(f"估计整体移调量: {global_shift:.2f} 音分")

    # 校正前 RMSE（无 DTW，保留时间错位信息）
    rmse_raw = compute_rmse_cents(f0_ref, f0_syn, voiced_ref, voiced_syn)
    print(f"校正前 F0 RMSE: {rmse_raw:.2f} 音分" if not np.isnan(rmse_raw) else "校正前 F0 RMSE: 无法计算")

    # 移调校正
    f0_syn_corrected = apply_shift_cents(f0_syn, global_shift)

    # 校正后 RMSE（DTW 对齐，排除节奏差异）
    rmse_corrected = compute_rmse_cents_dtw(f0_ref, f0_syn_corrected, voiced_ref, voiced_syn)
    print(f"校正后 F0 RMSE: {rmse_corrected:.2f} 音分" if not np.isnan(rmse_corrected) else "校正后 F0 RMSE: 无法计算")

    if plot:
        plot_f0_comparison(times_ref, f0_ref, times_syn, f0_syn,
                           voiced_ref=voiced_ref, voiced_syn=voiced_syn,
                           f0_syn_corrected=f0_syn_corrected,
                           save_path=save_plot_path)

    return {
        'global_shift_cents': global_shift,
        'rmse_raw_cents': rmse_raw,
        'rmse_corrected_cents': rmse_corrected
    }

if __name__ == "__main__":
    # ========== 在此处配置你的音频文件对 ==========
    audio_tasks = [
        ('/Users/fxt/Desktop/原声2 山歌唱了几十年.m4a',
         '/Users/fxt/Desktop/合成原声2.m4a',
         '/Users/fxt/Desktop/【新】旋律对比/老调/原声2_对比.png'),
    ]

    for ref, syn, img in audio_tasks:
        print(f"\n>>> 正在处理: {ref}  vs  {syn}")
        main(ref, syn, plot=True, save_plot_path=img)