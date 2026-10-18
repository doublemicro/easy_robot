"""하단 연타 실험: 다가가서 하단만 계속 차는 전략이 얼마나 이기는지 (규칙은 안 바꿈, 측정만).

    python -m tools.low_spam_experiment --n 200

연타하는 쪽은 사람처럼 상대를 바로 본다 (반응 지연 없음). 상대 봇들은 반응 지연 있음.
"""
import argparse
from collections import Counter

from game import config as C
from game.config import Act, St
from ai.bots import EasyBot, HardBot, NormalBot, RunnerBot, ZonerBot
from ai.env import StickFightEnv


class LowSpam:
    """사거리(start) 안이면 하단, 아니면 걸어서 다가간다. 하단이 맞으면 cancel 확률로 정면→상단 콤보."""

    def __init__(self, start=95, combo=False):
        self.start, self.combo = start, combo

    def act(self, world, me):
        f, o = world.f[me], world.f[1 - me]
        dist = abs(o.x - f.x)
        if f.state != St.IDLE:
            if self.combo and f.can_cancel():
                return Act.MID if f.kind == Act.LOW else Act.HIGH
            return Act.IDLE
        if dist < self.start:
            return Act.LOW
        return Act.RIGHT if o.x > f.x else Act.LEFT


def play(opp, n, spam, seed=0):
    st = Counter()
    env = StickFightEnv(relative_actions=False, opponent=opp, seed=seed)
    for _ in range(n):
        env.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(spam.act(env.world, 0))
            for e in info["events"]:
                if e[0] == "hit":
                    st["dmg%d" % e[1]] += e[4]
                    if e[1] == 0 and e[3] == Act.LOW:
                        st["low_hit"] += 1
                elif e[0] == "parry" and e[1] == 1:
                    st["parried"] += 1
        st["win"] += info["winner"] == 0
        st["frames"] += env.world.frame
    return st


def frame_table():
    lo, mid, hi, hv = (C.ATTACKS[k] for k in (Act.LOW, Act.MID, Act.HIGH, Act.HEAVY))
    reach = lambda a: a.box[1] + C.BODY_W / 2
    print("사거리(중심 간 거리): 하단 %.0f / 정면 %.0f / 상단 %.0f / 강공격 %.0f"
          % (reach(lo), reach(mid), reach(hi), reach(hv)))
    print("하단: 선딜 %d 판정 %d 후딜 %d (전체 %d), 데미지 %d, 숙인 자세 높이 %.0f → 상단(%.0f~)·투사체(%.0f) 빗나감"
          % (lo.startup, lo.active, lo.recovery, lo.total, lo.damage, C.BODY_H_LOW, hi.box[2], C.RANGED.height))
    # 헛친 하단을 벌하기: 판정이 끝난 뒤 남은 후딜 동안, 하단 사거리 끝(95)에서 정면이 닿으려면
    gap = reach(lo) - reach(mid)
    walk = gap / C.WALK_SPEED
    print("하단 사거리 끝에서 헛쳤을 때 정면으로 벌하기: 걸어서 %.0fpx(%.1f프레임) + 정면 선딜 %d = %.1f프레임 필요, 후딜은 %d프레임"
          % (gap, walk, mid.startup, walk + mid.startup, lo.recovery))
    print("  반응(8~14프레임)을 더하면: 하단이 시작된 걸 보고 움직이면 %d~%d프레임 뒤 도착, 하단은 %d프레임에 끝남"
          % (8 + walk + mid.startup, 14 + walk + mid.startup, lo.total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    frame_table()
    opps = (("하위", EasyBot), ("중간", NormalBot), ("상위", HardBot), ("도망", RunnerBot), ("견제", ZonerBot))
    for label, spam in (("하단만 (사거리 95)", LowSpam(95)), ("하단만 (붙어서 70)", LowSpam(70)),
                        ("하단→정면→상단 콤보", LowSpam(95, combo=True))):
        row = []
        for name, cls in opps:
            st = play(cls(seed=1), a.n, spam)
            n = a.n
            row.append(f"{name} {st['win'] / n:4.0%} (딜 {st['dmg0'] / n:3.0f}:{st['dmg1'] / n:3.0f}, {st['frames'] / n / 60:4.1f}초)")
        print(f"{label:16s} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
