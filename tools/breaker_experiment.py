"""콤보 브레이커 실험: 공격형 봇끼리 대전시켜 설정별 통계를 낸다.

    python -m tools.breaker_experiment --n 200

설정을 바꿀 때 game/config.py 의 BREAKER 를 실행 중에 바꿔 끼운다 (파일은 안 바뀜).
"""
import argparse
import random
import time
from collections import Counter

from game import config as C
from game.config import Act
from ai.bots import AggroBot
from ai.env import StickFightEnv
from ai.perception import SeenWorld


def play(n, seed, break_rate=0.6):
    st = Counter()
    combo_lens = []
    env = StickFightEnv(relative_actions=False, opponent=AggroBot(seed=seed, break_rate=break_rate), seed=seed)
    me = AggroBot(seed=seed + 1, break_rate=break_rate)
    for ep in range(n):
        obs, _ = env.reset()
        done = False
        worst = [0, 0]          # 각자 가장 크게 뒤졌던 체력 차
        cur = [0, 0]            # 지금 이어지는 콤보 타수 (때린 쪽 기준)
        while not done:
            obs, r, done, _, info = env.step(me.act(obs, SeenWorld(env.world, env.eyes[0]), 0))
            w = env.world
            for e in info["events"]:
                if e[0] == "hit":
                    att, combo = e[1], e[5]
                    if combo == 1 and cur[att] > 0:
                        combo_lens.append(cur[att])
                    cur[att] = combo
                    st["hits"] += 1
                    st["dmg"] += e[4]
                elif e[0] == "hit_bonus":
                    st["bonus_hits"] += 1
                elif e[0] == "break":
                    st["break"] += 1
                    combo_lens.append(cur[e[2]])
                    cur[e[2]] = 0
                elif e[0] == "break_fail":
                    st["fail"] += 1
            d = w.f[0].hp - w.f[1].hp
            worst[0] = max(worst[0], -d)
            worst[1] = max(worst[1], d)
        for k in (0, 1):
            if cur[k]:
                combo_lens.append(cur[k])
        win = info["winner"]
        st["matches"] += 1
        st["ko"] += int(any(f.hp <= 0 for f in env.world.f))
        st["frames"] += env.world.frame
        for k in (0, 1):
            if worst[k] >= 30:
                st["behind30"] += 1
                st["comeback30"] += int(win == k)
    st["long_combos"] = sum(1 for c in combo_lens if c >= 3)
    st["combos"] = len(combo_lens)
    st["combo_hits"] = sum(combo_lens)
    return st


def report(name, st):
    m = st["matches"]
    cb = st["comeback30"] / max(1, st["behind30"])
    tries = st["break"] + st["fail"]
    print(f"{name:28s} | 브레이크 {st['break']/m:4.2f}/판 성공률 {st['break']/max(1,tries):4.0%} "
          f"| 평균콤보 {st['combo_hits']/max(1,st['combos']):4.2f}타 3타이상 {st['long_combos']/max(1,st['combos']):4.0%} "
          f"| 보너스타 {st['bonus_hits']/max(1,st['hits']):4.0%} "
          f"| 30뒤진쪽 역전 {cb:4.0%} ({st['behind30']}) "
          f"| KO {st['ko']/m:4.0%} 평균 {st['frames']/m/60:4.1f}s", flush=True)


CONFIGS = {
    "끔 (브레이커 없음)": dict(gauge_max=1e9),
    "기본 (2.5, +15%)": dict(),
    "충전 느림 (1.5)": dict(gain_per_damage=1.5),
    "충전 빠름 (4.0)": dict(gain_per_damage=4.0),
    "보너스 없음 (+0%)": dict(full_bonus=0.0),
    "보너스 큼 (+30%)": dict(full_bonus=0.30),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    base = C.BREAKER
    for name, kw in CONFIGS.items():
        if a.only and a.only not in name:
            continue
        C.BREAKER = C.BreakerData(**kw)
        t = time.time()
        report(name, play(a.n, a.seed))
    C.BREAKER = base
