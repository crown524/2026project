"""DAIC-WOZ 项目路径与常量配置。

设计原则：所有路径集中此处，样本数量无关。
2 个 participant 可跑通流水线验证，189 个可出真实指标，代码不需改动。
"""
import json
import os
from pathlib import Path

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"


def _resolve_zip_dir() -> Path:
    """原始 zip 存放处。跨机器换电脑时不改代码，按三级优先级解析：

    1. 环境变量 DAIC_ZIP_DIR（临时覆盖，适合命令行一次性指定）
    2. local_paths.json 的 zip_dir（setup_and_run.py 写入，跟机器绑定）
    3. 本机桌面下的 daic数据集（默认猜测）

    local_paths.json 属于机器本地状态，不进便携包，因此每台机器各自持有
    自己的数据路径，同一份代码可在多台电脑间直接拷贝。
    """
    env = os.environ.get("DAIC_ZIP_DIR")
    if env:
        return Path(env).expanduser()

    cfg = PROJECT_ROOT / "local_paths.json"
    if cfg.exists():
        try:
            saved = json.loads(cfg.read_text(encoding="utf-8")).get("zip_dir")
            if saved:
                return Path(saved).expanduser()
        except (OSError, json.JSONDecodeError):
            pass  # 配置损坏时静默退回默认猜测，不阻断导入

    return Path.home() / "Desktop" / "daic数据集"


RAW_ZIP_DIR = _resolve_zip_dir()

# 按需抽取后的工作目录
EXTRACTED_DIR = DATA_ROOT / "extracted"
TRANSCRIPT_DIR = EXTRACTED_DIR / "transcripts"
COVAREP_DIR = EXTRACTED_DIR / "covarep"
FORMANT_DIR = EXTRACTED_DIR / "formant"
CLNF_DIR = EXTRACTED_DIR / "clnf"

# 标签与官方划分文件（需从 USC 单独下载，不在 *_P.zip 内）
LABEL_DIR = DATA_ROOT / "labels"
TRAIN_SPLIT = LABEL_DIR / "train_split_Depression_AVEC2017.csv"
DEV_SPLIT = LABEL_DIR / "dev_split_Depression_AVEC2017.csv"
TEST_SPLIT = LABEL_DIR / "full_test_split.csv"

# 产出
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
RESULT_DIR = OUTPUT_DIR / "results"

for d in (TRANSCRIPT_DIR, COVAREP_DIR, FORMANT_DIR, CLNF_DIR,
          LABEL_DIR, FEATURE_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- 数据格式常量（均为实测核实，勿凭猜测修改）----------

# 转写文件虽名为 .csv，实际是 TAB 分隔
TRANSCRIPT_SEP = "\t"

# 转写列名
COL_START, COL_STOP, COL_SPEAKER, COL_VALUE = "start_time", "stop_time", "speaker", "value"

# 说话人标识
SPEAKER_PARTICIPANT = "Participant"
SPEAKER_INTERVIEWER = "Ellie"

# 采样率
COVAREP_HZ = 100
CLNF_FPS = 30

# COVAREP 74 维，第 1 列为 F0，值为 0 表示未发声帧，必须过滤
COVAREP_N_DIM = 74
COVAREP_F0_COL = 0

# CLNF 质量过滤阈值
CLNF_MIN_CONFIDENCE = 0.90
CLNF_SUCCESS_VALUE = 1

# 需要从 zip 抽取的文件后缀（HOG 体积巨大且不可解释，明确排除）
WANTED_SUFFIXES = (
    "_TRANSCRIPT.csv",
    "_COVAREP.csv",
    "_FORMANT.csv",
    "_CLNF_AUs.txt",
    "_CLNF_gaze.txt",
    "_CLNF_pose.txt",
)
EXCLUDED_KEYWORDS = ("hog", "features3D", "features.txt", "AUDIO.wav")

# ---------- 标签 ----------
# PHQ-8 总分 0-24；AVEC2017 官方二分类阈值为 >=10
PHQ8_BINARY_THRESHOLD = 10
COL_PARTICIPANT_ID = "Participant_ID"
COL_PHQ8_SCORE = "PHQ8_Score"
COL_PHQ8_BINARY = "PHQ8_Binary"

# ---------- 实验 ----------
RANDOM_SEED = 42
N_BOOTSTRAP = 2000  # bootstrap 置信区间重采样次数
