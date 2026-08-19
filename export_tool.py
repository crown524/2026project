"""数据集导出工具（GUI）：把原始 zip 变成可随身携带的小体积产物。

在存放完整数据集的那台电脑上运行，一次点击完成四步：
    按需抽取（跳过 HOG）→ 计算确定性三路特征 → 生成文本观察与 D1 打分
    → 收集产物到指定文件夹

第三步要调 API，所以单独一个勾选项。不勾的话导出包里 obs/d1 两路
沿用旧数据，人数会落后于另外三路，实验时会被内连接砍到旧的人数。

产物分两级，默认只导出第一级：
    可外传（约 0.5MB）  特征宽表 + 标签。纯统计数字，无法反推原文
    受限（约 7MB）      转写原文 + Skill 缓存。含逐字引文，受 DAIC-WOZ
                        协议约束，只能在已签协议的成员间用 U 盘流转

用法：
    python export_tool.py        # 或双击 导出工具.bat
"""
from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
_v = HERE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PY = str(_v) if _v.exists() else sys.executable
LOCAL_CFG = HERE / "local_paths.json"

# (项目内相对路径, 是否受协议约束, 说明)
ARTIFACTS = [
    ("outputs/features", False, "特征宽表"),
    ("data/labels", False, "标签与官方划分"),
    ("data/extracted/transcripts", True, "转写原文"),
    ("outputs/skill_cache", True, "Skill 缓存"),
]

# 五路特征表。导出包里各表人数必须一致，否则实验时内连接会静默取最小值，
# 而不是报错——所以人数写进说明文件，目标机能一眼看出哪路没跟上。
LANE_FILES = [
    ("a1_transcript_timing.csv", "tt", "转写时序"),
    ("a3_acoustic.csv", "ac", "声学"),
    ("a5_visual.csv", "vi", "视觉"),
    ("a2_text_observations.csv", "obs", "文本观察"),
    ("d1_direct_scores.csv", "d1", "LLM 直接打分"),
]


class ExportApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("DAIC 数据集导出工具")
        root.geometry("760x560")
        self.q: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False

        self.src = tk.StringVar(value=self._saved_src())
        self.dst = tk.StringVar(value=str(Path.home() / "Desktop" / "daic_导出包"))
        self.do_extract = tk.BooleanVar(value=True)
        self.do_features = tk.BooleanVar(value=True)
        self.keep_audio = tk.BooleanVar(value=False)
        self.with_restricted = tk.BooleanVar(value=False)
        # 文本观察表也属于"实验台要的数据集"，但它靠调 API 生成而非纯计算，
        # 所以单独一项：数据机上勾一次，逐字引文留在本机，导出包只带计数。
        self.do_skills = tk.BooleanVar(value=False)

        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="① 数据集文件夹（放着 300_P.zip 这类文件）"
                  ).grid(row=0, column=0, columnspan=3, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.src, width=64).grid(
            row=1, column=0, columnspan=2, sticky="we", **pad)
        ttk.Button(frm, text="浏览…", command=self._pick_src).grid(
            row=1, column=2, **pad)
        self.src_info = ttk.Label(frm, text="", foreground="#555")
        self.src_info.grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        ttk.Label(frm, text="② 导出到（这个文件夹拷走即可）").grid(
            row=3, column=0, columnspan=3, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.dst, width=64).grid(
            row=4, column=0, columnspan=2, sticky="we", **pad)
        ttk.Button(frm, text="浏览…", command=self._pick_dst).grid(
            row=4, column=2, **pad)

        box = ttk.LabelFrame(frm, text="③ 选项", padding=8)
        box.grid(row=5, column=0, columnspan=3, sticky="we", **pad)
        ttk.Checkbutton(box, text="抽取 zip（已抽过可关掉，跳过最慢的一步）",
                        variable=self.do_extract).grid(row=0, sticky="w")
        ttk.Checkbutton(box, text="重算特征（改过特征代码就要勾）",
                        variable=self.do_features).grid(row=1, sticky="w")
        ttk.Checkbutton(box, text="生成文本观察与 D1 打分（要调 API，obs/d1 两路必需）",
                        variable=self.do_skills).grid(row=2, sticky="w")
        ttk.Checkbutton(box, text="另存原始音频到抽取层（A4 备用，不进导出包，很占空间）",
                        variable=self.keep_audio).grid(row=3, sticky="w")
        ttk.Checkbutton(box, text="含转写原文与 Skill 缓存（受协议约束，仅 U 盘流转）",
                        variable=self.with_restricted,
                        command=self._warn_restricted).grid(row=4, sticky="w")

        bar = ttk.Frame(frm)
        bar.grid(row=6, column=0, columnspan=3, sticky="we", **pad)
        self.btn = ttk.Button(bar, text="开始导出", command=self._start)
        self.btn.pack(side="left")
        ttk.Button(bar, text="打开导出文件夹", command=self._open_dst).pack(
            side="left", padx=8)
        self.prog = ttk.Progressbar(bar, length=380, mode="determinate")
        self.prog.pack(side="right")

        self.log = tk.Text(frm, height=15, wrap="word", font=("Consolas", 9))
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew", **pad)
        ttk.Scrollbar(frm, command=self.log.yview).grid(
            row=7, column=3, sticky="ns")
        frm.rowconfigure(7, weight=1)
        frm.columnconfigure(0, weight=1)

        self._refresh_src()
        self._pump()

    # ---------- 路径 ----------

    def _saved_src(self) -> str:
        if LOCAL_CFG.exists():
            try:
                v = json.loads(LOCAL_CFG.read_text(encoding="utf-8")).get("zip_dir")
                if v:
                    return v
            except (OSError, json.JSONDecodeError):
                pass
        return str(Path.home() / "Desktop" / "daic数据集")

    def _pick_src(self) -> None:
        d = filedialog.askdirectory(title="选择放着 *_P.zip 的文件夹")
        if d:
            self.src.set(d)
            self._refresh_src()

    def _pick_dst(self) -> None:
        d = filedialog.askdirectory(title="选择导出目标文件夹")
        if d:
            self.dst.set(d)

    def _refresh_src(self) -> None:
        """显示 zip 数量，并把路径锁进 local_paths.json 供其他脚本复用。"""
        p = Path(self.src.get())
        if not p.is_dir():
            self.src_info.config(text="路径不存在", foreground="#c00")
            return
        n = len(list(p.glob("*_P.zip")))
        col = "#080" if n else "#c00"
        self.src_info.config(
            text=f"找到 {n} 个 *_P.zip" + ("" if n else "（这个文件夹里没有数据包）"),
            foreground=col)
        if n:
            cfg = {}
            if LOCAL_CFG.exists():
                try:
                    cfg = json.loads(LOCAL_CFG.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cfg = {}
            cfg["zip_dir"] = str(p)
            LOCAL_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    def _open_dst(self) -> None:
        d = Path(self.dst.get())
        if not d.is_dir():
            messagebox.showinfo("还没有", "导出文件夹尚未生成。")
            return
        if os.name == "nt":
            os.startfile(d)  # noqa: S606
        else:
            subprocess.run(["xdg-open" if sys.platform != "darwin" else "open",
                            str(d)], check=False)

    def _warn_restricted(self) -> None:
        if self.with_restricted.get():
            messagebox.showwarning(
                "受协议约束",
                "转写原文与 Skill 缓存含访谈逐字内容。\n\n"
                "DAIC-WOZ 协议禁止再分发：不要放百度网盘、Google Drive 等\n"
                "第三方存储，也不要发给未签协议的人。仅限已签协议的组内成员\n"
                "用 U 盘或本地网络传递。")

    # ---------- 日志 ----------

    def _say(self, msg: str) -> None:
        self.q.put(("log", msg))

    def _pump(self) -> None:
        """主线程轮询队列刷新界面，工作线程不直接碰 tkinter。"""
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", str(val) + "\n")
                    self.log.see("end")
                elif kind == "prog":
                    self.prog.config(value=float(val))
                elif kind == "done":
                    self.busy = False
                    self.btn.config(state="normal", text="开始导出")
        except queue.Empty:
            pass
        self.root.after(120, self._pump)

    # ---------- 执行 ----------

    def _start(self) -> None:
        if self.busy:
            return
        src, dst = Path(self.src.get()), Path(self.dst.get())
        if not src.is_dir() or not list(src.glob("*_P.zip")):
            messagebox.showerror("路径不对", "数据集文件夹里没有 *_P.zip。")
            return
        if dst.resolve() == src.resolve():
            messagebox.showerror("路径冲突", "导出文件夹不能和数据集是同一个。")
            return
        self.busy = True
        self.btn.config(state="disabled", text="进行中…")
        self.log.delete("1.0", "end")
        threading.Thread(target=self._pipeline, args=(src, dst),
                         daemon=True).start()

    def _run(self, script: str, *args: str, env_src: Path) -> bool:
        """跑子进程并把输出实时喂进日志。抽取逻辑复用现有脚本，不重写。"""
        cmd = [PY, script, *args]
        self._say(f"$ {' '.join(cmd[1:])}")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8",
               "PYTHONUTF8": "1", "DAIC_ZIP_DIR": str(env_src)}
        try:
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=1)
        except OSError as e:
            self._say(f"  ! 启动失败: {e}")
            return False
        assert p.stdout is not None
        for raw in p.stdout:
            self._say("  " + raw.decode("utf-8", errors="replace").rstrip())
        return p.wait() == 0

    def _save_audio(self, src: Path) -> None:
        """把 *_AUDIO.wav 单独抽到抽取层。

        抽取器的排除名单里有 AUDIO.wav，若之后用 --delete-zip 删了原包，
        音频就永久丢失、A4(wav2vec) 再也做不了。这一步是保险。
        """
        out = HERE / "data" / "extracted" / "audio"
        out.mkdir(parents=True, exist_ok=True)
        zips = sorted(src.glob("*_P.zip"))
        got = 0
        for i, zp in enumerate(zips, 1):
            try:
                with zipfile.ZipFile(zp) as zf:
                    for info in zf.infolist():
                        name = info.filename.split("/")[-1]
                        if not name.upper().endswith("_AUDIO.WAV"):
                            continue
                        dest = out / name
                        if dest.exists() and dest.stat().st_size == info.file_size:
                            continue
                        with zf.open(info) as s, open(dest, "wb") as o:
                            shutil.copyfileobj(s, o, length=1 << 20)
                        got += 1
            except (zipfile.BadZipFile, OSError) as e:
                self._say(f"  ! {zp.name}: {e}")
            self.q.put(("prog", 100 * i / len(zips)))
        total = sum(f.stat().st_size for f in out.glob("*.wav"))
        self._say(f"  音频已留存 {got} 个（累计 {total / 1e9:.1f} GB）→ {out}")

    def _collect(self, dst: Path) -> tuple[int, int]:
        """把产物复制到导出文件夹。返回 (文件数, 字节数)。"""
        n = nb = 0
        for rel, restricted, desc in ARTIFACTS:
            if restricted and not self.with_restricted.get():
                continue
            srcd = HERE / rel
            if not srcd.is_dir():
                self._say(f"  跳过 {desc}：{rel} 不存在")
                continue
            cnt = 0
            for f in srcd.rglob("*"):
                if not f.is_file() or "__pycache__" in f.parts:
                    continue
                tgt = dst / rel / f.relative_to(srcd)
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, tgt)
                cnt += 1
                nb += f.stat().st_size
            n += cnt
            flag = "（受协议约束）" if restricted else ""
            self._say(f"  {desc}: {cnt} 个文件{flag}")
        return n, nb

    @staticmethod
    def _rows(path: Path) -> int:
        """数据行数（不含表头）；-1 表示文件不在包里。

        用 csv 解析而不是数换行：字段内若含换行，行数会算多。
        """
        if not path.exists():
            return -1
        try:
            with path.open(encoding="utf-8", newline="") as f:
                return max(sum(1 for _ in csv.reader(f)) - 1, 0)
        except OSError:
            return -1

    def _coverage(self, dst: Path) -> tuple[str, str]:
        """各路人数清单 + 人数不齐时的提示。"""
        counts, lines = [], []
        # 先排 ASCII 路名再排人数，中文放最后：中文是双宽字符，用 str 宽度
        # 补空格永远对不齐，把它挪到行尾就不需要对齐了
        for fn, code, name in LANE_FILES:
            r = self._rows(dst / "outputs/features" / fn)
            counts.append(r)
            lines.append(f"  {code:<5}{'缺失' if r < 0 else f'{r} 人':<8}{name}")

        got = [v for v in counts if v > 0]
        warn = ""
        if got and max(got) != min(got):
            warn = (f"\n⚠ 各路人数不一致（{min(got)}~{max(got)}）。实验脚本按 "
                    "participant_id 内连接，\n"
                    f"  多路组合时会被砍到 {min(got)} 人且不报错。要补齐就在导出工具里\n"
                    "  勾上「生成文本观察与 D1 打分」重跑一次。\n")
        return "\n".join(lines), warn

    def _manifest(self, dst: Path, src: Path, n: int, nb: int) -> None:
        zips = len(list(src.glob("*_P.zip")))
        tx = len(list((HERE / "data/extracted/transcripts").glob("*")))
        cov, warn = self._coverage(dst)
        (dst / "导出说明.txt").write_text(
            # 相邻字面量先拼接再乘，会把标题一起复制 40 遍；必须显式加号断开
            "DAIC 项目导出包\n"
            + "=" * 40 + "\n"
            + f"来源数据集: {zips} 个 *_P.zip\n"
            f"已抽取转写: {tx} 个\n"
            f"包含文件  : {n} 个，{nb / 1024:.0f} KB\n"
            f"含受限内容: {'是（转写原文，禁止再分发）' if self.with_restricted.get() else '否（仅统计特征，可自由传输）'}\n"
            + f"\n各路特征人数\n{cov}\n" + warn
            + "\n用法：把 outputs/ 与 data/ 两个文件夹合并进目标机的\n"
            "daic_project 同名位置，然后直接跑 run_experiment.py。\n"
            "\n注意：本包不含原始音视频，也不含 .venv。目标机需自行\n"
            "执行 setup_and_run.bat 建环境。\n",
            encoding="utf-8")
        self._say(f"  已写入 导出说明.txt")

    def _pipeline(self, src: Path, dst: Path) -> None:
        try:
            self._say(f"数据集: {src}")
            self._say(f"导出到: {dst}\n")

            if self.do_extract.get():
                self._say("[1] 按需抽取（跳过 HOG，这步最慢）")
                if not self._run("extract_from_zips.py", env_src=src):
                    self._say("  ! 抽取失败，后面的步骤已停止")
                    return
            else:
                self._say("[1] 跳过抽取")

            if self.keep_audio.get():
                self._say("\n[1b] 另存原始音频")
                self._save_audio(src)
            self.q.put(("prog", 0))

            if self.do_features.get():
                self._say("\n[2] 计算三路特征")
                if not self._run("build_features.py", env_src=src):
                    self._say("  ! 特征计算失败，后面的步骤已停止")
                    return
            else:
                self._say("\n[2] 跳过特征计算")

            if self.do_skills.get():
                self._say("\n[2b] 生成文本观察与 D1 打分（调 API，最花时间）")
                for sk in ("text_observation", "d1_direct_scoring"):
                    self._say(f"  → {sk}")
                    if not self._run("run_text_skill.py", "--skill", sk,
                                     env_src=src):
                        self._say(f"  ! {sk} 失败；obs/d1 两路的数据会不全，"
                                  "但确定性三路不受影响，继续导出")
            else:
                self._say("\n[2b] 跳过文本观察（obs/d1 两路将沿用已有数据）")

            self._say("\n[3] 收集产物")
            dst.mkdir(parents=True, exist_ok=True)
            n, nb = self._collect(dst)
            self._manifest(dst, src, n, nb)

            self._say(f"\n完成：{n} 个文件，{nb / 1024:.0f} KB")
            self._say(f"把整个 {dst.name} 文件夹拷走即可。")
            self.q.put(("prog", 100))
        except Exception as e:  # 后台线程异常必须捞住，否则界面静默卡死
            self._say(f"\n! 出错: {type(e).__name__}: {e}")
        finally:
            self.q.put(("done", None))


def main() -> None:
    root = tk.Tk()
    ExportApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
