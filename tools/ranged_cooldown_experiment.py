"""원거리 쿨타임 실험: 원거리로 견제하는 봇(ZonerBot) vs 공격형 봇(AggroBot), 쿨타임별 승률.

    python -m tools.ranged_cooldown_experiment --n 200

설정은 실행 중에만 바꿔 끼운다 (config.py 파일은 안 바뀜).
"""
import argparse
import dataclasses
from collections import Counter

from game import config as C
from ai.bots import AggroBot, ZonerBot
from ai.env import StickFightEnv
from ai.perception import SeenWorld


def play(n, seed=0):
    st = Counter()
    env = StickFightEnv(relative_actions=False, opponent=AggroBot(seed=seed), seed=seed)
    me = ZonerBot(seed=seed + 1)
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, done, _, info = env.step(me.act(obs, SeenWorld(env.world, env.eyes[0]), 0))
            for e in info["events"]:
                if e[0] == "shoot":
                    st["shoot%d" % e[1]] += 1
                elif e[0] == "hit" and e[3] == C.Act.RANGED:
                    st["phit%d" % e[1]] += 1
                    st["pdmg%d" % e[1]] += e[4]
        st["win"] += int(info["winner"] == 0)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cds", default="75,100,120,150,180")
    a = ap.parse_args()
    r0 = C.RANGED
    try:
        for cd in map(int, a.cds.split(",")):
            C.RANGED = dataclasses.replace(r0, cooldown=cd)
            st = play(a.n)
            n = a.n
            print(f"쿨타임 {cd:3d} ({cd / 60:.2f}초): 견제 봇 승률 {st['win'] / n:5.1%}  "
                  f"판당 발사 {st['shoot0'] / n:4.1f}회, 명중 {st['phit0'] / max(1, st['shoot0']):5.1%}, "
                  f"원거리 데미지 {st['pdmg0'] / n:4.1f}/판")
    finally:
        C.RANGED = r0


if __name__ == "__main__":
    main()
