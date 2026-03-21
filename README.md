# Xiling-Mountain-Songs
# 西岭山歌数字化再生客观评估代码

本仓库包含论文《基于零样本歌声合成的口头非遗数字化再生研究——以国家级非遗西岭山歌为例》的客观声学评估代码，用于计算：

- 音色相似度（SIM）——基于 WavLM 的说话人嵌入余弦相似度
- 旋律相似度（形状相似度）——基于 PYIN + FastDTW + 滑动窗口偏移校正
- 歌词准确率（CER）——基于 Paraformer 的字符错误率

所有评估均在本地运行，无需上传音频至云端。

---

## 环境配置

### 系统要求
- Python 3.9 ~ 3.12（Python 3.13 部分库可能不兼容）
- 推荐使用虚拟环境（conda 或 venv）

### 安装依赖
```bash
pip install -r requirements.txt
