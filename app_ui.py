"""实验观察台（Streamlit）：包在现有脚本外面的薄壳 UI。

定位：观察与迭代工具，不是最终产品界面。四个页面：
  1. 数据状态       —— 就绪自检、数据集路径登记、split 归属、缺包清单、标签上传
  2. Skill 与模型配置 —— 每个 Skill 独立选 provider/model/API key，连通测试
  3. 观察查看器     —— 转写原文 + 高亮命中 quote + 被剔除的幻觉引用
  4. 实验面板       —— 选特征路组合，一键跑消融，查看历史结果

启动：
    .venv/Scripts/streamlit run app_ui.py
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

import config as C
import eval_utils as E
import llm_providers as LP
from features import transcript_timing as tt

st.set_page_config(page_title="DAIC 多模态实验观察台", layout="wide")

VENV_PY = str(C.PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
CACHE_DIR = C.OUTPUT_DIR / "skill_cache"


# ---------- 通用 ----------

def run_script(args: list[str]) -> str:
    proc = subprocess.run(
        [VENV_PY] + args, cwd=str(C.PROJECT_ROOT),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        timeout=3600)
    return (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")


def local_pids() -> list[int]:
    return sorted(int(p.name.split("_")[0])
                  for p in C.TRANSCRIPT_DIR.glob("*_TRANSCRIPT.csv"))


# ---------- 页面 1：数据状态 ----------

def page_data_status():
    st.header("数据状态")
    pids = local_pids()
    ids = E.split_membership()
    have = set(pids)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("本地转写", f"{len(pids)} / 189")
    c2.metric("train 已有", f"{len(have & ids['train'])} / {len(ids['train']) or '?'}")
    c3.metric("dev 已有", f"{len(have & ids['dev'])} / {len(ids['dev']) or '?'}")
    c4.metric("test 已有", f"{len(have & ids['test'])} / {len(ids['test']) or '?'}")

    if not (C.TRAIN_SPLIT.exists() and C.DEV_SPLIT.exists()):
        st.error("缺少标签文件（train/dev split CSV），下方拖拽上传。")
    if have and ids["test"] and have <= ids["test"]:
        st.warning("本地样本全部属于 **test 集**：只能用于流水线调试与最终评估，"
                   "禁止用于提示词开发或调参（方案 §7.3）。")

    missing_tr = sorted(ids["train"] - have)
    missing_dv = sorted(ids["dev"] - have)
    if missing_tr or missing_dv:
        with st.expander(f"缺包清单：train 缺 {len(missing_tr)}、dev 缺 {len(missing_dv)}"):
            st.write("**train 缺：**", ", ".join(map(str, missing_tr)) or "无")
            st.write("**dev 缺：**", ", ".join(map(str, missing_dv)) or "无")

    st.subheader("上传标签 CSV（拖拽）")
    up = st.file_uploader("train/dev/test split 文件",
                          type="csv", accept_multiple_files=True)
    if up:
        for f in up:
            (C.LABEL_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"已保存 {len(up)} 个文件到 {C.LABEL_DIR}")
        st.rerun()

    st.subheader("数据集所在文件夹")
    st.caption("填 *_P.zip 所在目录。数据集几百 GB，浏览器上传不现实，"
               "这里只登记路径，抽取直接从本机磁盘读。保存后写入 "
               "`local_paths.json`（跟机器绑定，不进便携包）。")
    cur = str(C.RAW_ZIP_DIR)
    newp = st.text_input("路径", value=cur, key="zip_dir_input")
    n_zip = len(list(Path(newp).glob("*_P.zip"))) if Path(newp).is_dir() else -1
    if n_zip < 0:
        st.warning("该目录不存在。")
    else:
        st.success(f"目录存在，找到 {n_zip} 个 *_P.zip。")
    if st.button("保存路径", disabled=(newp == cur)):
        cfg = C.PROJECT_ROOT / "local_paths.json"
        d = {}
        if cfg.exists():
            try:
                d = json.loads(cfg.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        d["zip_dir"] = newp
        cfg.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        st.success("已保存。config 在进程启动时解析路径，请重启界面生效。")

    st.subheader("扫描新数据包并抽取")
    st.caption(f"当前抽取源 `{C.RAW_ZIP_DIR}`。抽取自动跳过 HOG（省数百 GB）。")
    col_a, col_b = st.columns(2)
    if col_a.button("抽取全部（转写+声学+视觉）"):
        with st.spinner("抽取中…"):
            st.code(run_script(["extract_from_zips.py"]) or "(无输出)")
    if col_b.button("抽取后重建三路特征"):
        with st.spinner("build_features 运行中…"):
            st.code(run_script(["build_features.py"]) or "(无输出)")

    feat_files = sorted(C.FEATURE_DIR.glob("*.csv"))
    if feat_files:
        st.subheader("已生成的特征文件")
        st.table(pd.DataFrame(
            [{"文件": f.name,
              "行数": sum(1 for _ in open(f, encoding="utf-8-sig")) - 1}
             for f in feat_files]))


# ---------- 页面 2：Skill 与模型配置 ----------

def page_skill_config():
    st.header("Skill 与模型配置")
    st.caption("每个 Skill 独立配置。key 明文存本地 data/skill_models.json，"
               "勿提交到任何仓库。中转端点若静默换模型，审计字段会记录实际模型名。")
    cfg = LP.load_config()

    for skill in sorted(cfg):
        with st.expander(f"**{skill}**", expanded=True):
            item = cfg[skill]
            c1, c2 = st.columns(2)
            item["provider"] = c1.selectbox(
                "provider", ["anthropic", "openai_compatible"],
                index=0 if item.get("provider") == "anthropic" else 1,
                key=f"prov_{skill}")
            item["model"] = c2.text_input("model", item.get("model", ""),
                                          key=f"model_{skill}")
            item["api_key"] = c1.text_input("api_key", item.get("api_key", ""),
                                            type="password", key=f"key_{skill}")
            item["base_url"] = c2.text_input(
                "base_url（官方端点留空；中转站填域名即可，/v1 自动补全）",
                item.get("base_url") or "", key=f"url_{skill}") or None

            level = item.get("reasoning", "off")
            item["reasoning"] = c1.selectbox(
                "推理强度",
                list(LP.REASONING_LEVELS),
                index=list(LP.REASONING_LEVELS).index(
                    level if level in LP.REASONING_LEVELS else "off"),
                key=f"reason_{skill}",
                help="off=关闭。openai_compatible 映射为 reasoning_effort"
                     "（OpenAI API 上限是 high，选 max 自动按 high 发送）；"
                     "anthropic 映射为 thinking 预算(low=2k/medium=8k/"
                     "high=16k/max=32k)。注意：Anthropic 开 thinking 后强制 "
                     "temperature=1，输出不再逐字确定——观察类 Skill 建议 off，"
                     "D1 打分实验可开。用「检测实效」确认端点没有丢弃该参数。")

            b1, b2, b3 = st.columns(3)
            if b1.button("测试连通", key=f"ping_{skill}"):
                with st.spinner("调用中…"):
                    ok, msg = LP.ping(item)
                if ok:
                    st.success(f"连通正常，实际模型: {msg}")
                    if msg != item["model"]:
                        st.warning(f"端点返回的模型 ≠ 请求的模型（{item['model']}），"
                                   "复现实验时以审计字段 model_actual 为准。")
                else:
                    st.error(msg)
            if b2.button("拉取模型列表", key=f"models_{skill}"):
                try:
                    ids = LP.list_models(item)
                    st.code("\n".join(ids) or "(空)")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
            if b3.button("检测推理强度实效", key=f"probe_{skill}"):
                with st.spinner("对比 off 与最高档的实际用量…"):
                    eff, detail = LP.probe_reasoning(item)
                if eff is True:
                    st.success(detail)
                elif eff is False:
                    st.error(detail)
                else:
                    st.warning(detail)

    if st.button("保存全部配置", type="primary"):
        LP.save_config(cfg)
        st.success(f"已保存到 {LP.CONFIG_PATH}")


# ---------- 页面 3：观察查看器 ----------

def _highlight(text: str, quotes: list[str], color: str) -> str:
    out = html.escape(text)
    for q in sorted(set(quotes), key=len, reverse=True):
        if not q:
            continue
        pat = re.compile(re.escape(html.escape(q)), re.IGNORECASE)
        out = pat.sub(lambda m: f"<mark style='background:{color}'>{m.group(0)}</mark>",
                      out)
    return out


def page_observation_viewer():
    st.header("观察查看器")
    st.caption("检查文本 Skill 的每条观察是否有据可依。绿色=通过逐字核验，"
               "红色=幻觉引用（已被剔除，不进特征）。")

    caches = sorted(CACHE_DIR.glob("*_text_observation_*.json"))
    if not caches:
        st.info("还没有 Skill 运行结果。先在「实验面板」或命令行运行 "
                "run_text_skill.py。")
        return

    ids = E.split_membership()
    entries = []
    for p in caches:
        pid = int(p.name.split("_")[0])
        split = ("train" if pid in ids["train"] else
                 "dev" if pid in ids["dev"] else "test")
        entries.append({"pid": pid, "split": split, "path": p})

    show_test = st.checkbox("显示 test 样本（仅最终评估时勾选）", value=False)
    pool = [e for e in entries if show_test or e["split"] != "test"]
    if not pool:
        st.warning("当前只有 test 样本的结果。提示词迭代请以 train 样本为准；"
                   "如确需查看请勾选上面的复选框。")
        return

    sel = st.selectbox(
        "选择会话",
        pool, format_func=lambda e: f"{e['pid']}（{e['split']}）{e['path'].name}")
    result = json.loads(sel["path"].read_text(encoding="utf-8"))
    meta = result.get("_meta", {})

    c1, c2, c3, c4 = st.columns(4)

    # 兼容新旧schema：v2使用items字段，v1使用observations字段
    items = result.get("items", [])
    observations = result.get("observations", [])

    # 从items中提取所有evidence_for作为观察
    obs_list = []
    if items:
        for item in items:
            for ev in item.get("evidence_for", []):
                obs_list.append({
                    "dimension": item.get("dimension"),
                    "quote": ev.get("quote"),
                    "turn_index": ev.get("turn_index"),
                    "strength": ev.get("strength"),
                    "is_quoted_or_reported": ev.get("is_quoted_or_reported", False)
                })
    else:
        obs_list = observations

    c1.metric("观察数", len(obs_list))
    c2.metric("剔除幻觉", result.get("quote_verification_failed", 0))
    c3.metric("弃答", "是" if result.get("abstained") else "否")
    c4.metric("模型", str(meta.get("model_actual", "?")))
    st.caption(f"prompt_hash={meta.get('prompt_hash', '?')[:12]}  "
               f"provider={meta.get('provider')}  "
               f"latency={meta.get('latency_s')}s")

    sf = result.get("safety_flags", {})
    if sf.get("self_harm_explicit") or sf.get("self_harm_ideation_possible"):
        st.error(f"安全标志命中: {sf}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("转写（高亮命中）")
        df = tt.load_transcript(C.TRANSCRIPT_DIR / f"{sel['pid']}_TRANSCRIPT.csv")
        good = [o.get("quote", "") for o in obs_list]
        bad = [o.get("quote", "") for o in result.get("rejected_observations", [])]
        blocks = []
        for _, row in df.iterrows():
            spk = row[C.COL_SPEAKER]
            txt = str(row[C.COL_VALUE])
            if spk == C.SPEAKER_PARTICIPANT:
                h = _highlight(txt, good, "#b6f0b6")
                h = _highlight(h, bad, "#f5b5b5") if bad else h
                blocks.append(f"<b>P:</b> {h}")
            else:
                blocks.append(f"<span style='color:#888'>E: "
                              f"{html.escape(txt)}</span>")
        st.markdown("<div style='max-height:520px;overflow-y:auto;"
                    "line-height:1.7'>" + "<br>".join(blocks) + "</div>",
                    unsafe_allow_html=True)
    with right:
        st.subheader("观察列表")
        for o in obs_list:
            flags = []
            if o.get("is_quoted_or_reported"):
                flags.append("quoted")
            _dim = o.get("dimension") or ""
            _str = o.get("strength") or ""
            _flags = (", " + "/".join(flags)) if flags else ""
            _quote = o.get("quote") or ""
            _turn = o.get("turn_index") or ""
            st.markdown(f"- **{_dim}** ({_str}{_flags}) - \"{_quote}\" `P{_turn}`")
        if result.get("rejected_observations"):
            st.subheader("被剔除（幻觉引用）")
            for o in result["rejected_observations"]:
                _dim = o.get("dimension") or ""
                _quote = o.get("quote") or ""
                st.markdown(f"- ~~{_dim}: \"{_quote}\"~~")


# ---------- 页面 4：实验面板 ----------

EXP_PRESETS = {
    "A1 仅转写时序": "tt",
    "A2 仅文本观察": "obs",
    "A3 仅声学": "ac",
    "A5 仅视觉": "vi",
    "B1 时序+文本": "tt,obs",
    "B3 三模态": "obs,ac,vi",
    "B4/D2 全配置": "tt,obs,ac,vi",
    "D1 LLM直接打分": "d1",
}


def page_experiments():
    st.header("实验面板")

    st.subheader("运行 Skill（先于 obs/d1 相关实验）")
    c1, c2, c3 = st.columns([2, 1, 1])
    skill = c1.selectbox("skill", ["text_observation", "d1_direct_scoring"])
    limit = c2.number_input("limit（0=全部）", 0, 200, 0)
    if c3.button("运行 Skill"):
        args = ["run_text_skill.py", "--skill", skill]
        if limit:
            args += ["--limit", str(int(limit))]
        with st.spinner("调用 LLM 中…（有缓存，重复运行只跑新增样本）"):
            st.code(run_script(args) or "(无输出)")

    st.subheader("消融实验")
    preset = st.selectbox("实验组", list(EXP_PRESETS))
    lanes = st.text_input("特征路（可手改）", EXP_PRESETS[preset])
    model_kind = st.radio("回归器", ["gbdt", "ridge", "both"], horizontal=True)
    if st.button("运行实验", type="primary"):
        with st.spinner("训练评测中…"):
            st.code(run_script(["run_experiment.py", "--lanes", lanes,
                                "--model", model_kind]) or "(无输出)")

    st.subheader("历史结果")
    results = sorted(C.RESULT_DIR.glob("exp_*.json"), reverse=True)
    if not results:
        st.info("暂无结果文件。")
        return
    sel = st.selectbox("结果文件", results, format_func=lambda p: p.name)
    data = json.loads(sel.read_text(encoding="utf-8"))
    st.caption(f"lanes={data.get('lanes')}  n={data.get('n_samples')}  "
               f"time={data.get('timestamp')}")
    rows = []
    for r in data.get("results", []):
        rows.append({
            "配置": r.get("label"), "n": r.get("n"),
            "Spearman": r.get("spearman"),
            "Spearman 95%CI": str(r.get("spearman_ci95")),
            "RMSE": r.get("rmse"), "MAE": r.get("mae"),
            "PR-AUC": r.get("pr_auc"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------- 入口 ----------

PAGES = {
    "数据状态": page_data_status,
    "Skill 与模型配置": page_skill_config,
    "观察查看器": page_observation_viewer,
    "实验面板": page_experiments,
}

page = st.sidebar.radio("页面", list(PAGES))
st.sidebar.caption(f"项目: {C.PROJECT_ROOT}")
PAGES[page]()
