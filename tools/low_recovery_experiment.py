"""하단 후딜 실험: 헛친 하단을 얼마나 벌할 수 있는지, 하단 연타가 얼마나 약해지는지.

    python -m tools.low_recovery_experiment --n 100

1) 벌하기 창: 하단을 사거리(95) 밖 거리 d에서 헛침. 상대는 w프레임째부터 걸어 들어가 정면(닿으면 바로).
   하단 판정에 안 맞고 정면을 맞히는 w의 개수를 센다 (많을수록 벌하기 쉬움).
2) 하단만 연타(사거리 끝 95)하는 전략의 봇 상대 승률.
설정은 실행 중에만 바꿔 끼운다.
"""
import argparse
import dataclasses

from game import config as C
from game.config import Act, St
from game.core import World
from ai.bots import EasyBot, HardBot, NormalBot, RunnerBot, ZonerBot
from tools.low_spam_experiment import LowSpam, play

I = Act.IDLE


def punish(dist, w, use=Act.MID):
    """하단이 아직 끝나지 않았을 때(후딜 중) 정면을 맞히고, 나는 안 맞았으면 True."""
    world = World(300, 300 + dist)
    rng = C.ATTACKS[use].box[1] + C.BODY_W / 2 - 1
    for t in range(70):
        f0, f1 = world.f
        a0 = Act.LOW if t == 0 else I
        a1 = I
        if t >= w and f1.state == St.IDLE:
            a1 = use if abs(f1.x - f0.x) < rng else Act.LEFT
        low_on = f0.state == St.ATTACK and f0.kind == Act.LOW
        hp0 = f0.hp
        world.step(a0, a1)
        if world.f[0].hp < hp0:
            return low_on and world.f[1].hp == 100
        if world.f[1].hp < 100:
            return False
    return False


def windows(dist):
    """정면으로 벌하는 데 성공하는 출발 프레임들 (하단 시작 = 0프레임)."""
    return [w for w in range(0, 40) if punish(dist, w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--recs", default="16,20,24,28")
    a = ap.parse_args()
    lo0 = C.ATTACKS[Act.LOW]
    opps = (("하위", EasyBot), ("중간", NormalBot), ("상위", HardBot), ("도망", RunnerBot), ("견제", ZonerBot))
    try:
        for rec in map(int, a.recs.split(",")):
            C.ATTACKS[Act.LOW] = dataclasses.replace(lo0, recovery=rec)
            ws = {d: windows(d) for d in (100, 110, 120)}
            desc = "  ".join(f"{d}px: {len(w)}개" + (f"({w[0]}~{w[-1]}프레임)" if w else "") for d, w in ws.items())
            print(f"후딜 {rec:2d} (전체 {lo0.startup + lo0.active + rec}) | 정면으로 벌하는 출발 타이밍 {desc}")
            row = []
            for name, cls in opps:
                st = play(cls(seed=1), a.n, LowSpam(95))
                row.append(f"{name} {st['win'] / a.n:4.0%}")
            print("          하단만 연타 승률: " + "  ".join(row), flush=True)
    finally:
        C.ATTACKS[Act.LOW] = lo0


if __name__ == "__main__":
    main()
