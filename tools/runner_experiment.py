"""도망 전략 실험: 한 대 때리고 도망가는 봇(RunnerBot) vs 쫓는 공격형 봇(AggroBot).

    python -m tools.runner_experiment --n 200

설정별로 쫓는 쪽 승률, 시간 끝 판정승 비율, 대쉬 공격 사용·명중을 비교한다.
설정은 실행 중에만 바꿔 끼운다 (config.py 파일은 안 바뀜).
"""
import argparse
import dataclasses
import time
from collections import Counter

from game import config as C
from ai.bots import AggroBot, RunnerBot
from ai.env import StickFightEnv
from ai.perception import SeenWorld


def play(n, seed, dash_attack):
    st = Counter()
    env = StickFightEnv(relative_actions=False, opponent=RunnerBot(seed=seed), seed=seed)
    me = AggroBot(seed=seed + 1, dash_attack=dash_attack)
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, done, _, info = env.step(me.act(obs, SeenWorld(env.world, env.eyes[0]), 0))
            for e in info["events"]:
                if e[0] == "dash_attack" and e[1] == 0:
                    st["da_used"] += 1
                elif e[0] == "hit" and e[3] == C.DASH_ATK and e[1] == 0:
                    st["da_hit"] += 1
                elif e[0] == "parry" and e[2] == 0 and env.world.f[0].kind == C.DASH_ATK:
                    st["da_parried"] += 1
                elif e[0] == "wall_splat":
                    st["wall%d" % e[1]] += 1
        w = env.world
        st["matches"] += 1
        st["chaser_win"] += int(info["winner"] == 0)
        st["runner_win"] += int(info["winner"] == 1)
        timeout = all(f.hp > 0 for f in w.f)
        st["runner_timeout_win"] += int(timeout and info["winner"] == 1)
        st["hp_diff"] += w.f[0].hp - w.f[1].hp
    return st


def report(name, st):
    m = st["matches"]
    print(f"{name:28s} 쫓는쪽 승 {st['chaser_win'] / m:5.1%}  도망쪽 승 {st['runner_win'] / m:5.1%}"
          f" (시간끝 {st['runner_timeout_win'] / m:5.1%})  체력차 {st['hp_diff'] / m:+6.1f}"
          f"  대쉬공격 {st['da_used'] / m:4.1f}회/판 명중 {st['da_hit'] / max(1, st['da_used']):5.1%}"
          f"  벽꽝(도망쪽) {st['wall1'] / m:4.1f}회/판")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t = time.time()
    dash0, wall0 = C.DASH, C.WALL
    v5_dash = dataclasses.replace(dash0, back_recovery=dash0.recovery)
    no_wall = dataclasses.replace(wall0, min_speed=1e9)
    configs = (("v5 (아무것도 없음)", False, v5_dash, no_wall),
               ("+ 대쉬 공격", True, v5_dash, no_wall),
               ("+ 대쉬 공격 + 백대쉬 후딜", True, dash0, no_wall),
               ("+ 대쉬 공격 + 벽꽝", True, v5_dash, wall0),
               ("v6 전부", True, dash0, wall0))
    try:
        for name, da, dash, wall in configs:
            C.DASH, C.WALL = dash, wall
            report(name, play(a.n, a.seed, da))
    finally:
        C.DASH, C.WALL = dash0, wall0
    print(f"({time.time() - t:.0f}초)")


if __name__ == "__main__":
    main()
