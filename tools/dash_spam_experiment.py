"""대쉬 공격 연타 실험: '앞 대쉬 → 대쉬 공격'만 반복하는 단순 전략이 얼마나 이기는지.

    python -m tools.dash_spam_experiment --n 200

연타하는 쪽은 사람처럼 상대를 바로 본다 (반응 지연 없음, 대쉬 k프레임째에 J).
상대는 공격형 봇 / 도망 봇 (반응 지연 있음). 설정은 실행 중에만 바꿔 끼운다.
"""
import argparse
import dataclasses
from collections import Counter

from game import config as C
from game.config import Act, St
from ai.bots import AggroBot, RunnerBot
from ai.env import StickFightEnv


class DashSpam:
    def __init__(self, k=4, start_dist=200):
        self.k, self.start_dist = k, start_dist

    def act(self, world, me):
        f, o = world.f[me], world.f[1 - me]
        dist = abs(o.x - f.x)
        if f.state == St.DASH and f.dash_dir == f.facing and f.sf >= self.k - 1:
            return Act.MID
        if f.state != St.IDLE:
            return Act.LOW if f.can_cancel() and f.kind != Act.LOW else Act.IDLE
        if f.dash_cd == 0 and dist < self.start_dist:
            return Act.DASH_F
        return Act.RIGHT if o.x > f.x else Act.LEFT


def play(opp_cls, n, spam):
    st = Counter()
    env = StickFightEnv(relative_actions=False, opponent=opp_cls(seed=0), seed=0)
    for _ in range(n):
        env.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(spam.act(env.world, 0))
        st[info["winner"]] += 1
    return st[0] / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    d0 = C.DASH
    try:
        for name, end in (("앞 대쉬 무적 1~7", 8), ("앞 대쉬 무적 1~3", 4), ("앞 대쉬 무적 없음", 1)):
            C.DASH = dataclasses.replace(d0, invuln_end=end)
            row = []
            for k, sd in ((3, 150), (6, 200), (10, 260)):
                sp = DashSpam(k, sd)
                row.append(f"J {k}프레임째: 공격형 {play(AggroBot, a.n, sp):4.0%} 도망 {play(RunnerBot, a.n, sp):4.0%}")
            print(f"{name:14s} | " + " | ".join(row))
    finally:
        C.DASH = d0


if __name__ == "__main__":
    main()
