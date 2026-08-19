# DAIC-WOZ 抑郁症检测实验平台

基于 DAIC-WOZ 数据集的多模态抑郁症检测研究项目，整合语音声学特征、视觉特征（CLNF AU）、转写时序特征与 LLM 文本观察特征，通过逻辑回归与 Stacking 集成进行 PHQ-8 二分类预测。

## 项目结构

```
├── config.py               # 路径与常量（单一来源）
├── llm_providers.py        # 多提供商 LLM 接入层（Anthropic / OpenAI 兼容）
├── build_features.py       # 特征构建流水线
├── calibrate_and_fuse.py   # 校准与特征融合
├── eval_utils.py           # 评估工具（bootstrap CI 等）
├── app_ui.py               # Streamlit 可视化界面
├── export_tool.py          # 结果导出工具
├── features/               # 各模态特征提取模块
│   ├── acoustic.py
│   ├── visual.py
│   └── transcript_timing.py
├── skills/                 # LLM Skill 定义（Prompt 模板）
│   ├── text_observation/
│   └── d1_direct_scoring/
├── data/
│   ├── labels/             # AVEC2017 官方划分 CSV（需单独获取）
│   └── skill_models.json   # ⚠️ API 配置，不进仓库
├── outputs/
│   ├── features/           # 提取完毕的特征 CSV
│   └── results/            # 实验结果 JSON
└── tests/
```

## 数据集获取

原始数据集（DAIC-WOZ）需从 [USC ICT](https://dcapswoz.ict.usc.edu/) 申请授权后下载，无法通过本仓库获取。

下载完成后，在项目根目录创建 `local_paths.json`，填写 zip 文件所在目录：

```json
{
  "zip_dir": "你的数据集路径"
}
```

也可以用环境变量代替：

```bash
set DAIC_ZIP_DIR=你的数据集路径
```

## 环境安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## API 配置

复制以下模板创建 `data/skill_models.json`（此文件含明文密钥，**不进仓库**）：

```json
{
  "text_observation": {
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "api_key": "sk-ant-...",
    "base_url": null,
    "reasoning": "off"
  },
  "d1_direct_scoring": {
    "provider": "openai_compatible",
    "model": "deepseek-chat",
    "api_key": "sk-...",
    "base_url": "https://api.deepseek.com/v1",
    "reasoning": "off"
  }
}
```

也可以用环境变量 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` 代替配置文件中的 `api_key` 字段。

## 运行

```bash
# 从 zip 提取数据
python extract_from_zips.py

# 构建特征
python build_features.py

# 启动可视化界面
streamlit run app_ui.py
```

## 不进仓库的内容

| 路径 | 原因 |
|---|---|
| `data/extracted/` | 原始数据集（12 GB） |
| `data/skill_models.json` | 含明文 API key |
| `data/anthropic_key.txt` | API key 文件 |
| `local_paths.json` | 机器本地路径 |
| `outputs/skill_cache/` | LLM 调用缓存（含 API 响应） |
| `.venv/` | 虚拟环境 |

## 数据集引用

```
Gratch, J. et al. (2014). The Distress Analysis Interview Corpus of human and
computer interviews. LREC 2014.
```
