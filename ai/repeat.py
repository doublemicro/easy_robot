"""(v10) 반복 동작 감지: AI가 한 기술이나 같은 순서만 되풀이하는지 잰다.

- 판단할 때마다(4프레임) 내 캐릭터를 보고, 새로 시작한 기술만 기록한다 (버튼을 누른 횟수가 아니라 실제로 나간 기술).
  기술: 상단/정면/하단/원거리/패링/대쉬/강공격(기 모으기 시작)/대쉬 공격/점프
- 같은 라벨이라도 sf(상태 프레임)가 줄었으면 다시 시작한 것으로 센다.
- summarize(): 가장 많이 쓴 기술의 비율, 가장 많이 나온 3연속 순서의 비율 (+ 상위 5개씩),
  (v11) 4·5·6연속 순서 중 가장 많은 것의 비율 (long). 예: 원거리 > 대쉬 > 대쉬 공격 > 정면 > 정면 > 하단
- check(): 기준(MOVE_LIMIT, GRAM_LIMITS)을 넘었는지. 넘은 것 중 가장 긴 순서를 대표 패턴(flag)으로 고른다.
  긴 순서일수록 우연히 똑같이 나오기 어려워 기준이 낮다.
  기준값 참고 (상위 봇 상대, 3/4/5/6연속): 무작위 신경망 1/1/0/1%, 상위 봇 8/6/6/3%,
  정면↔하단을 반복하던 AI 22/13~15/10~11/7~8%.
  train.py는 이 값이 기준을 넘는 상태가 계속되면 학습을 멈춘다.
- record_observation(): 사람이 대전에서 본 패턴을 비율 없이 기록 (python -m tools.report_exploit "RANGED > DASH_ATK" --note ...)
- record(): 반복이 감지되면 밸런스 패치용 기록에 쌓는다 (계속 반복된다 = 그 기술/순서가 너무 세다는 신호).
    balance_reports/repeats.jsonl   기록 원본 (한 줄 = 한 건)
    balance_reports/사기기술.md      요약 (기록할 때마다 새로 만든다)
"""
import dataclasses
import json
import os
import time
from collections import Counter

from game import config as C

SKILLS = {C.LABELS.index(n): n for n in
          ("HIGH", "MID", "LOW", "RANGED", "PARRY", "DASH", "CHARGE", "DASH_ATK", "CROUCH")}
KOR = {"HIGH": "상단", "MID": "정면", "LOW": "하단", "RANGED": "원거리", "PARRY": "패링", "DASH": "대쉬",
       "CHARGE": "강공격", "DASH_ATK": "대쉬 공격", "JUMP": "점프", "CROUCH": "숙이기"}
MOVE_LIMIT = 0.40                                   # 기술 하나의 비율
GRAM_LIMITS = {3: 0.20, 4: 0.12, 5: 0.08, 6: 0.06}  # n연속 순서 하나의 비율
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "balance_reports")


class MoveTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None
        self.moves = []

    def update(self, f):
        cur = (f.label, f.sf, f.grounded)
        if self.prev is not None:
            pl, psf, pg = self.prev
            if f.label in SKILLS and (f.label != pl or f.sf < psf):
                self.moves.append(SKILLS[f.label])
            elif pg and not f.grounded and f.label == 0:
                self.moves.append("JUMP")
        self.prev = cur


def _grams(seqs, n):
    c = Counter()
    for s in seqs:
        c.update(tuple(s[i:i + n]) for i in range(len(s) - n + 1))
    return c, max(1, sum(c.values()))


def summarize(seqs, n=3, long=(4, 5, 6)):
    """seqs: 판마다의 기술 목록들. → dict (top_move, move_share, top_gram, gram_share, moves, top5_*, long)."""
    moves = Counter()
    for s in seqs:
        moves.update(s)
    grams, ng = _grams(seqs, n)
    nm = max(1, sum(moves.values()))
    tm, tmc = moves.most_common(1)[0] if moves else ("-", 0)
    tg, tgc = grams.most_common(1)[0] if grams else (("-",), 0)
    out = {"top_move": tm, "move_share": tmc / nm,
           "top_gram": " > ".join(tg), "gram_share": tgc / ng,
           "moves": sum(moves.values()),
           "top5_moves": [[k, round(c / nm, 3)] for k, c in moves.most_common(5)],
           "top5_grams": [[" > ".join(k), round(c / ng, 3)] for k, c in grams.most_common(5)],
           "long": {}}
    for k in long:
        g, t = _grams(seqs, k)
        if g:
            top, c = g.most_common(1)[0]
            out["long"][str(k)] = [" > ".join(top), round(c / t, 3)]
    return out


def check(rep, move_limit=MOVE_LIMIT, gram_limits=None):
    """기준을 넘었으면 대표 패턴 {"pattern", "len", "share"}, 아니면 None.
    넘은 순서 중 가장 긴 것을 고르고, 순서는 안 넘고 기술 하나만 넘었으면 그 기술."""
    lim = GRAM_LIMITS if gram_limits is None else gram_limits
    seqs = {3: [rep["top_gram"], rep["gram_share"]], **{int(k): v for k, v in rep.get("long", {}).items()}}
    over = [(n, g, sh) for n, (g, sh) in seqs.items() if n in lim and sh > lim[n]]
    if over:
        n, g, sh = max(over)
        return {"pattern": g, "len": n, "share": round(sh, 3)}
    if rep["move_share"] > move_limit:
        return {"pattern": rep["top_move"], "len": 1, "share": round(rep["move_share"], 3)}
    return None


def parse_limits(text):
    """"3:0.2,4:0.12" → {3: 0.2, 4: 0.12}"""
    return {int(k): float(v) for k, v in (x.split(":") for x in text.split(",") if x.strip())}


def by_opponent(tagged):
    """tagged: (상대 이름, 기술 목록) 들 → 상대별 summarize."""
    groups = {}
    for name, s in tagged:
        groups.setdefault(name, []).append(s)
    return {k: summarize(v) for k, v in groups.items()}


# ---------------------------------------------------------------------- 밸런스 패치용 기록
def _move_data(m):
    """기술 이름 → 지금 config의 수치 (패치 전후 비교용)."""
    kind = {"HIGH": C.Act.HIGH, "MID": C.Act.MID, "LOW": C.Act.LOW, "CHARGE": C.Act.HEAVY,
            "DASH_ATK": C.DASH_ATK}.get(m)
    if kind is not None:
        d = dataclasses.asdict(C.ATTACKS[kind])
        if m == "DASH_ATK":
            d["slide"] = dataclasses.asdict(C.DASH_ATTACK)
        if m == "CHARGE":
            d["stage_damage"] = list(C.HEAVY_DAMAGE)
        return d
    obj = {"DASH": C.DASH, "RANGED": C.RANGED, "PARRY": C.PARRY}.get(m)
    return dataclasses.asdict(obj) if obj is not None else {}


def record(source, rep, per_opp=None, winrates=None, ckpt=None, steps=None, stage=None, note=""):
    """반복 감지 한 건을 기록하고 요약을 다시 만든다. 기록한 항목을 돌려준다."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    flag = rep.get("flag")
    involved = set(rep["top_gram"].split(" > ")) | {rep["top_move"]}
    if flag:
        involved |= set(flag["pattern"].split(" > "))
    involved = sorted(involved - {"-"})
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, "ckpt": ckpt,
             "steps": steps, "stage": stage, "note": note,
             "top_move": rep["top_move"], "move_share": _r(rep["move_share"]),
             "top_gram": rep["top_gram"], "gram_share": _r(rep["gram_share"]),
             "top5_moves": rep.get("top5_moves"), "top5_grams": rep.get("top5_grams"),
             "long": rep.get("long"), "flag": flag,
             "per_opponent": {k: {"top_move": v["top_move"], "move_share": round(v["move_share"], 3),
                                  "top_gram": v["top_gram"], "gram_share": round(v["gram_share"], 3)}
                              for k, v in (per_opp or {}).items()},
             "winrates": winrates or {},
             "frame_data": {m: _move_data(m) for m in involved}}
    with open(os.path.join(REPORT_DIR, "repeats.jsonl"), "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    write_summary()
    return entry


def _r(x):
    return None if x is None else round(x, 3)


def _pct(x):
    return "사람 관찰" if x is None else f"{x:.0%}"


def _best(reps, key, share, val):
    xs = [r[share] for r in reps if r[key] == val and r[share] is not None]
    return f"{max(xs):.0%}" if xs else "사람 관찰"


def record_observation(sequence, note="", ckpt=None):
    """사람이 대전에서 본 AI의 패턴을 기록한다 (비율 없이). sequence = "RANGED > DASH > DASH_ATK > MID" 처럼."""
    moves = [m.strip() for m in sequence.split(">")]
    rep = {"top_move": moves[0],
           "move_share": None, "top_gram": " > ".join(moves), "gram_share": None,
           "flag": {"pattern": " > ".join(moves), "len": len(moves), "share": None}}
    return record("사람 관찰", rep, ckpt=ckpt, note=note)


def load_reports():
    path = os.path.join(REPORT_DIR, "repeats.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _k(m):
    return " > ".join(KOR.get(x, x) for x in m.split(" > "))


def write_summary():
    """repeats.jsonl → 사기기술.md (많이 걸린 기술·순서 순위 + 건별 상세)."""
    reps = load_reports()
    mv, gr = Counter(r["top_move"] for r in reps), Counter(r["top_gram"] for r in reps)
    L = ["# 반복 동작 기록 (밸런스 패치 후보)", "",
         "AI가 학습 중 같은 기술·같은 순서만 계속 쓰면 여기에 쌓인다. 많이 걸린 것일수록 너무 센 기술일 가능성이 크다.",
         "자동 생성 파일 — 원본은 `repeats.jsonl`. (`ai/repeat.py`)", "",
         f"기록 {len(reps)}건", "", "## 가장 많이 반복된 기술", "", "| 기술 | 걸린 횟수 | 최고 비율 |", "|---|---|---|"]
    for m, c in mv.most_common():
        L.append(f"| {KOR.get(m, m)} | {c} | {_best(reps, 'top_move', 'move_share', m)} |")
    fl = Counter(r["flag"]["pattern"] for r in reps if r.get("flag"))
    L += ["", "## 걸린 패턴 (기준을 넘은 가장 긴 순서)", "",
          "| 패턴 | 길이 | 걸린 횟수 | 최고 비율 |", "|---|---|---|---|"]
    for g, c in fl.most_common():
        xs = [r["flag"]["share"] for r in reps if r.get("flag") and r["flag"]["pattern"] == g]
        best = max((x for x in xs if x is not None), default=None)
        L.append(f"| {_k(g)} | {len(g.split(' > '))} | {c} | {_pct(best)} |")
    L += ["", "## 가장 많이 반복된 순서 (3연속)", "", "| 순서 | 걸린 횟수 | 최고 비율 |", "|---|---|---|"]
    for g, c in gr.most_common():
        L.append(f"| {_k(g)} | {c} | {_best(reps, 'top_gram', 'gram_share', g)} |")
    L += ["", "## 건별 기록 (최신이 위)", ""]
    for r in reversed(reps):
        L.append(f"### {r['time']} — {r['source']}")
        info = [f"체크포인트 `{r['ckpt']}`" if r.get("ckpt") else None,
                f"{r['steps']:,} 판단" if r.get("steps") else None,
                f"단계 {r['stage']}" if r.get("stage") else None]
        if any(info):
            L.append("- " + ", ".join(x for x in info if x))
        if r.get("note"):
            L.append(f"- 메모: {r['note']}")
        if r["gram_share"] is None:
            L.append(f"- 순서 **{_k(r['top_gram'])}** (사람이 대전에서 관찰)")
        else:
            L.append(f"- 기술 **{KOR.get(r['top_move'], r['top_move'])} {_pct(r['move_share'])}**, "
                     f"순서 **{_k(r['top_gram'])} {_pct(r['gram_share'])}**")
        f = r.get("flag")
        if f and f["share"] is not None:
            L.append(f"- 걸린 패턴: **{_k(f['pattern'])}** ({f['len']}연속, {_pct(f['share'])})")
        if r.get("long"):
            L.append("- 긴 순서: " + ", ".join(f"{n}연속 {_k(g)} {sh:.0%}" for n, (g, sh) in r["long"].items()))
        if r.get("top5_moves"):
            L.append("- 기술 상위: " + ", ".join(f"{KOR.get(m, m)} {s:.0%}" for m, s in r["top5_moves"]))
        if r.get("top5_grams"):
            L.append("- 순서 상위: " + ", ".join(f"{_k(g)} {s:.0%}" for g, s in r["top5_grams"]))
        if r.get("per_opponent"):
            L.append("- 상대별: " + ", ".join(f"{k} = {KOR.get(v['top_move'], v['top_move'])} {v['move_share']:.0%} / "
                                              f"{_k(v['top_gram'])} {v['gram_share']:.0%}"
                                              for k, v in r["per_opponent"].items()))
        if r.get("winrates"):
            L.append("- 그때 승률: " + ", ".join(f"{k} {v:.0%}" for k, v in r["winrates"].items()))
        for m, d in r.get("frame_data", {}).items():
            if d:
                keys = ("startup", "active", "recovery", "damage", "hitstun", "knockback", "cooldown", "duration")
                L.append(f"- {KOR.get(m, m)} 수치: " + ", ".join(f"{k} {d[k]}" for k in keys if k in d))
        L.append("")
    with open(os.path.join(REPORT_DIR, "사기기술.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(L))
