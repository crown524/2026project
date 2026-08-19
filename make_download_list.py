"""生成 DAIC-WOZ 下载清单（按优先级分批），写入数据集文件夹。"""
import pandas as pd

import config as C

OUT = C.RAW_ZIP_DIR / "下载清单.txt"

tr = sorted(pd.read_csv(C.TRAIN_SPLIT)["Participant_ID"].astype(int))
dv = sorted(pd.read_csv(C.DEV_SPLIT)["Participant_ID"].astype(int))
te_df = pd.read_csv(C.TEST_SPLIT)
te = sorted(te_df["Participant_ID"].astype(int))

have = sorted(int(p.name.split("_")[0])
              for p in C.RAW_ZIP_DIR.glob("*_P.zip"))


def fmt(ids, per_line=6):
    lines = []
    for i in range(0, len(ids), per_line):
        lines.append("  " + "  ".join(f"{p}_P.zip" for p in ids[i:i + per_line]))
    return lines


lines = ["DAIC-WOZ 下载清单（只下 编号_P.zip，放入本文件夹即可）",
         "=" * 55, ""]
first = [p for p in tr[:30] if p not in have]
lines.append(f"【第一批 · train 前 30 个】凑齐即可出第一批真实指标")
lines += fmt(first)
rest = [p for p in tr[30:] if p not in have]
lines += ["", f"【第二批 · train 其余 {len(rest)} 个】"]
lines += fmt(rest)
dv_need = [p for p in dv if p not in have]
lines += ["", f"【第三批 · dev 全部 {len(dv_need)} 个】正式评测需要"]
lines += fmt(dv_need)
te_need = [p for p in te if p not in have]
lines += ["", f"【最后 · test 其余 {len(te_need)} 个】只在最终评估用一次，最后再下"]
lines += fmt(te_need)
lines += ["",
          f"已有 ({len(have)}): " + "  ".join(f"{p}_P.zip" for p in have),
          "",
          "不用下载: index.php",
          "已就位: 4 个 split 标签 csv、util.zip",
          "可选: Documentation PDF、documents.zip（官方文档，建议看但不阻塞）",
          "",
          "注意: zip 里包含巨大的 HOG 文件，无法单独不下（在包里），",
          "      但抽取脚本会自动跳过它，不会占用你的磁盘。",
          "      zip 不要手动解压，放进来点 UI 的抽取按钮即可。"]

OUT.write_text("\n".join(lines), encoding="utf-8-sig")
print(f"train={len(tr)} dev={len(dv)} test={len(te)} 已有={have}")
print(f"清单已写入: {OUT}")
print()
print("第一批前 12 个:")
print("  " + "  ".join(f"{p}_P.zip" for p in first[:12]))
