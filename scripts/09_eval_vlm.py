#!/usr/bin/env python3
"""Đo ACCURACY thật của VLM so với nhãn tay, không phải "CLEAR rate".

Vì sao cần: 08_ab_prompt.py chỉ có 1 frame ground-truth âm nên nó chấm sai —
prompt nào nói CLEAR nhiều là "thắng", kể cả khi bỏ sót cột điện giữa đường.
Với thiết bị dẫn đường cho người khiếm thị thì bỏ sót (false negative) nguy
hiểm hơn báo thừa (false positive), nên phải tách hai loại lỗi ra mà đo.

Chấm 4 mức, khắt khe dần:
  1. recall      = trong các frame CÓ vật cản, model báo được bao nhiêu %
  2. specificity = trong các frame KHÔNG có vật cản, model nói CLEAR bao nhiêu %
  3. type acc    = báo đúng loại vật (tính trên frame đã báo đúng là có hazard)
  4. pos acc     = báo đúng hướng

balanced accuracy = (recall + specificity) / 2 — dùng vì bộ frame lệch 12/2.

Dùng:
    python scripts/09_eval_vlm.py
    python scripts/09_eval_vlm.py --prompts pipe_v3 pipe_v5 --repeat 3
    python scripts/09_eval_vlm.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvp_config as cfg  # noqa: E402
import vlm_prompts as vp  # noqa: E402

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"
GT_PATH = os.path.join(FRAMES_DIR, "ground_truth.json")


def load_ground_truth() -> dict:
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)["frames"]
    return {k: v for k, v in gt.items() if not v.get("excluded")}


def load_frames(gt: dict, long_edge: int):
    from PIL import Image

    out = []
    for name in sorted(gt):
        path = os.path.join(FRAMES_DIR, name)
        if not os.path.exists(path):
            print(f"  (thiếu file {name}, bỏ qua)")
            continue
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > long_edge:
            s = long_edge / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        # Round-trip JPEG: đúng thứ server nhận từ WebSocket (§13)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=cfg.JPEG_QUALITY)
        out.append((name, Image.open(buf).convert("RGB")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.VLM_MODEL)
    ap.add_argument("--prompts", nargs="+", default=list(vp.PROMPTS))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--long-edge", type=int, default=cfg.FRAME_LONG_EDGE)
    ap.add_argument("--dump", default=None, help="ghi kết quả thô ra file JSON")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    gt = load_ground_truth()
    frames = load_frames(gt, args.long_edge)
    n_pos = sum(1 for n, _ in frames if gt[n]["hazard"])
    n_neg = len(frames) - n_pos

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation=cfg.ATTN_IMPL,
        device_map="cuda:0",
    )
    model.eval()

    print(f"Model : {args.model} | fp16 | {args.long_edge}px")
    print(f"Frames: {len(frames)} dùng được ({n_pos} có vật cản / {n_neg} trống)")
    print(f"Prompt: {', '.join(args.prompts)} | repeat={args.repeat}\n")

    def infer(img, prompt, max_new):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Frame:"}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        plen = inputs["input_ids"].shape[-1]
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        g = out[0][plen:]
        return processor.decode(g, skip_special_tokens=True).strip(), ms, len(g)

    infer(frames[0][1], vp.PROMPTS[args.prompts[0]], 24)  # warmup

    summary, dump = [], {}
    for pname in args.prompts:
        prompt = vp.PROMPTS[pname]
        max_new = vp.MAX_NEW_TOKENS.get(pname, cfg.VLM_MAX_NEW_TOKENS)
        four = pname in vp.FOUR_FIELD
        parse = cfg.parse_pipe4 if four else cfg.parse_pipe

        tp = fn = tn = fp = bad = 0
        hard_tn = hard_fp = 0
        type_ok = pos_ok = type_n = 0
        lat, toks, rows = [], [], []
        triad: dict[str, str] = {}

        for _ in range(args.repeat):
            for fname, img in frames:
                raw, ms, ntok = infer(img, prompt, max_new)
                lat.append(ms)
                toks.append(ntok)
                parsed = parse(vp.split_v5(raw))
                truth = gt[fname]["hazard"]
                is_hard = gt[fname].get("hard", False)

                if parsed is None:
                    bad += 1
                    # Sai schema = Decision Engine bỏ frame = im lặng.
                    # Trên frame CÓ vật cản thì đó là bỏ sót, phải tính là FN.
                    if truth:
                        fn += 1
                    else:
                        tn += 1
                        hard_tn += is_hard
                    rows.append((fname, truth, None, f"SAI:{raw[:30]}", ms))
                    continue

                # Với schema 4 trường, "có cảnh báo hay không" là quyết định của
                # Decision Engine chứ không phải của VLM: chỉ báo khi NEAR (§16).
                if four:
                    said = parsed["hazard"] and parsed.get("distance") == "near"
                else:
                    said = parsed["hazard"]

                if truth and said:
                    tp += 1
                    type_n += 1
                    type_ok += parsed["type"] in gt[fname]["types"]
                    pos_ok += parsed["position"] in gt[fname]["positions"]
                elif truth and not said:
                    fn += 1
                elif not truth and said:
                    fp += 1
                    hard_fp += is_hard
                else:
                    tn += 1
                    hard_tn += is_hard
                out = "|".join(str(parsed[k]) for k in
                               (("type", "position", "distance", "action") if four
                                else ("type", "position", "action")))
                rows.append((fname, truth, said, out, ms))
                triad.setdefault(fname, out)

        recall = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        hard_spec = hard_tn / (hard_tn + hard_fp) if hard_tn + hard_fp else 0.0
        bal = (recall + spec) / 2
        t_acc = type_ok / type_n if type_n else 0.0
        p_acc = pos_ok / type_n if type_n else 0.0
        med = statistics.median(lat)
        summary.append((pname, recall, spec, bal, t_acc, p_acc, med,
                        statistics.mean(toks), bad, hard_spec))

        print(f"--- {pname} ---")
        print(f"  recall {recall:.0%} ({tp}/{tp + fn})  "
              f"specificity {spec:.0%} ({tn}/{tn + fp})  "
              f"balanced {bal:.0%}")
        print(f"  specificity trên {hard_tn + hard_fp} frame ÂM KHÓ (vật ở xa / lệch bên, §16): "
              f"{hard_spec:.0%} ({hard_tn}/{hard_tn + hard_fp})")
        print(f"  type {t_acc:.0%}  position {p_acc:.0%}  "
              f"| {med:.0f} ms  {statistics.mean(toks):.1f} tok  sai schema {bad}")

        # Cặp đối chứng: cùng một cây cột, chỉ khác khoảng cách / vị trí ngang.
        trio = ["synthnear_pole.jpg", "synthfar_pole.jpg", "synthside_pole.jpg"]
        if all(t in triad for t in trio):
            near, far, side = (triad[t] for t in trio)
            if four:
                # Chỉ so trường khoảng cách — đó mới là thứ đang được kiểm tra.
                dn, df = near.split("|")[2], far.split("|")[2]
                verdict = (f"phân biệt được xa/gần (near={dn} far={df})" if dn != df
                           else f"KHÔNG phân biệt được xa/gần (cả hai đều {dn})")
            else:
                verdict = ("phân biệt được" if near != far and near != side
                           else "KHÔNG phân biệt được khoảng cách/vị trí")
            print(f"  cùng-1-cây-cột: gần={near} xa={far} lệch-bên={side}\n"
                  f"                  -> {verdict}")

        seen = set()
        for fname, truth, said, txt, _ms in rows:
            if fname in seen:
                continue
            seen.add(fname)
            if said is None:
                mark = "!!"
            elif truth and not said:
                mark = "BỎ SÓT"
            elif not truth and said:
                mark = "BÁO THỪA"
            else:
                mark = "ok"
            print(f"    {'HAZ' if truth else 'CLR'} {fname:<26} {txt:<34} {mark}")
        print()
        dump[pname] = [{"frame": f, "truth_hazard": t, "said_hazard": s,
                        "output": o, "ms": round(m)} for f, t, s, o, m in rows]

    print("=== Tổng hợp (sắp theo balanced accuracy) ===")
    hdr = (f"{'prompt':<10}{'recall':>8}{'specif':>8}{'sp.khó':>8}{'balanc':>8}"
           f"{'type':>7}{'pos':>7}{'ms':>8}{'tok':>7}")
    print(hdr)
    for s in sorted(summary, key=lambda s: -s[3]):
        print(f"{s[0]:<10}{s[1]:7.0%}{s[2]:7.0%}{s[9]:7.0%}{s[3]:7.0%}{s[4]:6.0%}"
              f"{s[5]:6.0%}{s[6]:7.0f}{s[7]:7.1f}")

    best = max(summary, key=lambda s: (s[3], -s[6]))
    print(f"\nTốt nhất: {best[0]} — balanced {best[3]:.0%}, recall {best[1]:.0%}, "
          f"specificity {best[2]:.0%} (âm khó {best[9]:.0%}), {best[6]:.0f} ms")
    if best[1] < 0.9:
        print(f"  CẢNH BÁO: recall {best[1]:.0%} nghĩa là cứ 10 vật cản thì bỏ sót "
              f"{round((1 - best[1]) * 10)}. Với thiết bị dẫn đường thì KHÔNG dùng được "
              "một mình — §16 cần thêm detector/depth, hoặc đổi model lớn hơn.")
    if best[9] < 0.5:
        print(f"  CẢNH BÁO: specificity trên frame âm khó chỉ {best[9]:.0%} — model báo cả "
              "vật ở xa và vật lệch hẳn sang bên. Đúng nguy cơ 'overload thông tin' "
              "mà §30 lo. Decision Engine (§14/§16) phải tự lọc theo khoảng cách, "
              "không tin được VLM ở khoản này.")
    print(f"\nLƯU Ý: chỉ {n_neg} frame ground-truth âm nên specificity có bước nhảy "
          f"{1 / max(n_neg, 1):.0%}. §8 vẫn cần 100-500 frame thật có nhãn.")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "long_edge": args.long_edge,
                       "results": dump}, f, ensure_ascii=False, indent=2)
        print(f"Kết quả thô -> {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
