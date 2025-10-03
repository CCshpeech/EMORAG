# -*- coding: utf-8 -*-
# Parallel (SLURM array) utterance-level embedding extractor for emotion2vec
# Input : /scratch/yiyach/emotion2vec_out/wav.scp
# Output: /scratch/yiyach/emotion2vec_out/utter_embeddings.part{ID}.csv

import os, csv, math
from funasr import AutoModel

WAV_SCP = "/scratch/yiyach/emotion2vec_out/wav.scp"
OUT_DIR = "/scratch/yiyach/emotion2vec_out"
MODEL_ID = "emotion2vec/emotion2vec_base"  # HF 正确ID

# 读取 SLURM 数组信息
task_id  = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
task_cnt = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

os.makedirs(OUT_DIR, exist_ok=True)
part_csv = os.path.join(OUT_DIR, f"utter_embeddings.part{task_id:03d}.csv") if task_cnt > 1 \
           else os.path.join(OUT_DIR, "utter_embeddings.csv")

print(f"[INFO] WAV_SCP={WAV_SCP}")
print(f"[INFO] OUT_DIR={OUT_DIR}")
print(f"[INFO] MODEL_ID={MODEL_ID}")
print(f"[INFO] ARRAY {task_id+1}/{task_cnt} -> {part_csv}")

# 读取清单并切片
with open(WAV_SCP, "r") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

n = len(lines)
per = math.ceil(n / task_cnt)
start = task_id * per
end   = min((task_id + 1) * per, n)
subset = lines[start:end]
print(f"[INFO] total {n} items; this task handles [{start}:{end}) = {len(subset)}")

# 加载模型
model = AutoModel(model=MODEL_ID, hub="hf")

# 提取
with open(part_csv, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    header = ["utt_id", "path"] + [f"f{i+1}" for i in range(768)]
    writer.writerow(header)

    for i, line in enumerate(subset, 1):
        pieces = line.split(maxsplit=1)
        if len(pieces) == 2 and not pieces[0].endswith(".wav"):
            uid, wav = pieces[0], pieces[1]
        else:
            uid, wav = f"utt_{start+i:08d}", pieces[-1]

        try:
            rec = model.generate(
                wav,
                granularity="utterance",
                extract_embedding=True
            )
            # ---- 兼容列表或字典两种返回 ----
            if isinstance(rec, list) and rec and isinstance(rec[0], dict) and "feats" in rec[0]:
                vec = rec[0]["feats"]
            elif isinstance(rec, dict) and "feats" in rec:
                vec = rec["feats"]
            else:
                raise ValueError(f"Unexpected return from generate(): {type(rec)} / keys={getattr(rec, 'keys', lambda: [])()}")

            row = [uid, wav] + [float(x) for x in vec]
        except Exception as e:
            print(f"[WARN] fail {uid}: {e}")
            row = [uid, wav] + [""] * 768

        writer.writerow(row)

        if i % 50 == 0 or i == len(subset):
            print(f"[{i}/{len(subset)}] processed")

print(f"[INFO] Done: {part_csv}")
