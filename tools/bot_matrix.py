"""봇 리그전: 모든 봇을 서로(같은 봇끼리 포함) 붙여 승률 표를 만든다.

    python -m tools.bot_matrix --n 200
    python -m tools.bot_matrix --n 200 --bots 하위,중간,상위,도망,견제,대쉬연타   # 난이도 봇 확인

- 각 조합은 n판, 절반은 왼쪽(P1)·절반은 오른쪽(P2)에서 시작해 자리 차이를 없앤다.
- 표의 값 = 행 봇이 열 봇을 이긴 비율. 무승부(시간 끝 동체력)는 따로 센다.
- 지금 config.py 설정(v6: 대쉬 무적 없음, 원거리 쿨타임 120) 그대로 돌린다.
- 하위/중간/상위 = EasyBot/NormalBot/HardBot (봇마다 자기 반응 시간 reaction_ms를 쓴다).
- 대쉬연타(DashSpam)는 tools.dash_spam_experiment 것과 같고 반응 지연 없이 상대를 바로 본다.
  나머지 봇은 모두 반응 지연된 화면(SeenWorld)을 본다.
"""
import argparse
import time
from collections import Counter

from ai.bots import AggroBot, EasyBot, HardBot, NormalBot, RandomBot, RunnerBot, ZonerBot
from ai.env import StickFightEnv
from ai.perception import SeenWorld
from tools.dash_spam_experiment import DashSpam


class _SpamAdapter:
    """DashSpam(world, me) → act(obs, world, me) 형태. 지연 없는 실제 world를 본다."""

    def __init__(self, env_ref, k=6, start_dist=200):
        self.env_ref, self.spam = env_ref, DashSpam(k, start_dist)

    def act(self, obs, world, me):
        return self.spam.act(self.env_ref[0].world, me)


ENTRANTS = ("공격형", "도망", "견제", "대쉬연타", "무작위")


def make(name, seed, env_ref):
    if name == "공격형":
        return AggroBot(seed=seed)
    if name == "도망":
        return RunnerBot(seed=seed)
    if name == "견제":
        return ZonerBot(seed=seed)
    if name == "하위":
        return EasyBot(seed=seed)
    if name == "중간":
        return NormalBot(seed=seed)
    if name == "상위":
        return HardBot(seed=seed)
    if name == "대쉬연타":
        return _SpamAdapter(env_ref)
    return RandomBot(seed=seed)


def duel(a, b, n, seed=0):
    """a가 이긴 판, b가 이긴 판, 무승부, 시간끝 판 수."""
    st = Counter()
    for side in (0, 1):
        env_ref = [None]
        bot_a = make(a, seed + 11, env_ref)
        bot_b = make(b, seed + 23, env_ref)
        env = StickFightEnv(relative_actions=False, opponent=bot_b, me=side, seed=seed + side,
                            reaction_ms=getattr(bot_a, "reaction_ms", (130, 180, 240)))
        env_ref[0] = env
        for _ in range(n // 2):
            obs, _ = env.reset()
            done = False
            while not done:
                act = bot_a.act(obs, SeenWorld(env.world, env.eyes[side]), side)
                obs, _, done, _, info = env.step(act)
            w = info["winner"]
            st["a" if w == side else "b" if w == 1 - side else "draw"] += 1
            st["timeout"] += int(all(f.hp > 0 for f in env.world.f))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bots", default=",".join(ENTRANTS))
    a = ap.parse_args()
    names = a.bots.split(",")
    t = time.time()
    res = {}
    for i, x in enumerate(names):
        for y in names[i:]:
            st = duel(x, y, a.n)
            res[x, y] = st
            res[y, x] = Counter(a=st["b"], b=st["a"], draw=st["draw"], timeout=st["timeout"])
            print(f"  {x} vs {y}: {st['a']}-{st['b']} (무 {st['draw']}, 시간끝 {st['timeout']})", flush=True)
    print("\n행 봇의 승률 (무승부 제외 아님, n판 기준)")
    print(f"{'':8s}" + "".join(f"{y:>9s}" for y in names) + f"{'평균':>9s}")
    for x in names:
        row = [res[x, y]["a"] / a.n for y in names]
        others = [r for y, r in zip(names, row) if y != x]
        print(f"{x:8s}" + "".join(f"{r:9.0%}" for r in row) + f"{sum(others) / len(others):9.0%}")
    print(f"({time.time() - t:.0f}초)")


if __name__ == "__main__":
    main()
