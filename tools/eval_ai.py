"""학습된 AI 체크포인트를 봇과 붙여 승률을 잰다 (v10, 1단계).

    python -m tools.eval_ai runs/stage1/latest.pt --n 200
    python -m tools.eval_ai runs/stage1/best_normal.pt --bots normal,hard --greedy

- n판 중 절반은 왼쪽(P1)·절반은 오른쪽(P2)에서 시작.
- 기본은 학습 때처럼 확률대로 행동을 고른다. --greedy면 가장 확률 높은 행동만.
- 학습 중 승률(최근 300판)과 달리 고정된 정책으로 따로 재므로 더 정확하다.
- (v10) 대쉬 공격: 판당 AI가 쓴 횟수 / 맞힌 비율 / 패링당한 비율 / 그 뒤 반격(상대 공격에 맞음)당한 비율.
  사람이 거리를 벌리고 헛친 대쉬 공격을 벌하면 AI가 지던 문제를 보는 지표.
- (v10) 반복 동작: 가장 많이 쓴 기술 / 3연속 순서의 비율 (ai/repeat.py).
  (v11) 4·5·6연속 순서도 본다. --record 를 주면 기준(ai/repeat.py의 MOVE_LIMIT / GRAM_LIMITS)을 넘을 때
  balance_reports/에 기록한다 (넘은 것 중 가장 긴 순서가 대표 패턴).

    python -m tools.eval_ai runs/stage1/latest.pt --record --note "사람이 거리 조절로 이김"
"""
import argparse
import time
from collections import Counter

from game import config as C
from ai.bots import BOTS
from ai.env import StickFightEnv
from ai.policy import PolicyBot, load_net
from ai.repeat import (GRAM_LIMITS, MOVE_LIMIT, MoveTracker, by_opponent, check, parse_limits, record,
                       summarize)


def duel(net, bot_name, n, greedy=False, seed=0, seqs=None):
    st = Counter()
    for side in (0, 1):
        ai = PolicyBot(net, greedy=greedy, seed=seed + side)
        env = StickFightEnv(opponent=BOTS[bot_name](seed=seed + 7 + side), me=side, seed=seed + side)
        for _ in range(n // 2):
            obs, _ = env.reset()
            done = False
            pending = None                    # 마지막 대쉬 공격: [남은 판단 수, 맞았나]
            tr = MoveTracker()
            while not done:
                obs, _, done, _, info = env.step(ai.act(obs))
                tr.update(env.world.f[side])
                for e in info["events"]:
                    if e[0] == "dash_attack" and e[1] == side:
                        st["da"] += 1
                        pending = [12, False]          # 약 0.8초 안에 무슨 일이 있었나
                    elif pending and e[0] == "hit" and e[1] == side and e[3] == C.DASH_ATK:
                        st["da_hit"] += 1
                        pending[1] = True
                    elif pending and e[0] == "parry" and e[2] == side:
                        st["da_parried"] += 1
                    elif pending and not pending[1] and e[0] == "hit" and e[2] == side:
                        st["da_punished"] += 1
                        pending = None
                if pending:
                    pending[0] -= 1
                    if pending[0] <= 0:
                        pending = None
            if seqs is not None:
                seqs.append((bot_name, tr.moves))
            w = info["winner"]
            st["win" if w == side else "loss" if w == 1 - side else "draw"] += 1
            st["timeout"] += int(all(f.hp > 0 for f in env.world.f))
            st["hp"] += env.world.f[side].hp - env.world.f[1 - side].hp
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bots", default="easy,normal,hard")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--record", action="store_true", help="반복이 기준을 넘으면 balance_reports/에 기록")
    ap.add_argument("--repeat-move", type=float, default=MOVE_LIMIT)
    ap.add_argument("--repeat-grams", default=",".join(f"{k}:{v}" for k, v in GRAM_LIMITS.items()))
    ap.add_argument("--note", default="", help="기록에 남길 메모")
    a = ap.parse_args()
    net, ck = load_net(a.ckpt)
    print(f"{a.ckpt}: {ck.get('steps', 0):,} 판단 학습, 단계 {ck.get('stage', 0) + 1}, 저장 {ck.get('time', '?')}")
    tagged, wins = [], {}
    for name in a.bots.split(","):
        t = time.time()
        st = duel(net, name, a.n, a.greedy, a.seed, tagged)
        wins[name] = round(st["win"] / a.n, 3)
        print(f"  vs {name:7s}: 승 {st['win'] / a.n:4.0%}  패 {st['loss'] / a.n:4.0%}  무 {st['draw']}  "
              f"(시간끝 {st['timeout']}, 평균 체력차 {st['hp'] / a.n:+.0f}, {time.time() - t:.0f}초)", flush=True)
        if st["da"]:
            print(f"             대쉬 공격 판당 {st['da'] / a.n:.1f}회: 맞힘 {st['da_hit'] / st['da']:.0%}  "
                  f"패링당함 {st['da_parried'] / st['da']:.0%}  헛치고 반격당함 {st['da_punished'] / st['da']:.0%}")

    rep = summarize([q for _, q in tagged])
    lim = parse_limits(a.repeat_grams)
    rep["flag"] = flag = check(rep, a.repeat_move, lim)
    over = flag is not None
    print(f"  반복: 기술 {rep['top_move']} {rep['move_share']:.0%} (기준 {a.repeat_move:.0%}), "
          f"3연속 {rep['top_gram']} {rep['gram_share']:.0%} (기준 {lim.get(3, 0):.0%})")
    for n, (g, sh) in rep["long"].items():
        print(f"        {n}연속 {g} {sh:.0%} (기준 {lim.get(int(n), 1):.0%})")
    if flag:
        print(f"  ← 걸림: {flag['pattern']} ({flag['len']}연속 {flag['share']:.0%})")
    if a.record and over:
        record(f"eval_ai{' --greedy' if a.greedy else ''} vs {a.bots}", rep, by_opponent(tagged), wins,
               a.ckpt, ck.get("steps"), ck.get("stage", 0) + 1, a.note)
        print("  → balance_reports/사기기술.md 에 기록")


if __name__ == "__main__":
    main()
