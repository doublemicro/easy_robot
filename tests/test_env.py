"""v9: 학습 환경(ai/env.py) 테스트 — 상대 기준 행동, 상대 바꾸기, 무승부 보상."""
from game.config import Act
from ai.bots import IdleBot, AggroBot, HardBot
from ai.env import StickFightEnv


def _walk(me, action, relative=True, n=10):
    env = StickFightEnv(opponent=IdleBot(), me=me, seed=0, randomize_start=False,
                        relative_actions=relative)
    env.reset(seed=0)
    x0 = env.world.f[me].x
    for _ in range(n):
        env.step(action)
    return env.world.f[me].x - x0


def test_relative_right_moves_toward_opponent_on_both_sides():
    # P1(왼쪽)은 RIGHT = 오른쪽으로, P2(오른쪽)는 RIGHT = 상대 쪽 = 왼쪽으로
    assert _walk(0, Act.RIGHT) > 0
    assert _walk(1, Act.RIGHT) < 0
    assert _walk(0, Act.LEFT) < 0
    assert _walk(1, Act.LEFT) > 0


def test_absolute_mode_keeps_screen_direction():
    assert _walk(1, Act.RIGHT, relative=False) > 0


class _RelIdleWalker:
    """학습된 정책처럼 상대 기준 행동을 내는 상대: 항상 '앞으로 걷기'."""
    relative_actions = True

    def act(self, obs, world, me):
        return Act.RIGHT


def test_relative_opponent_walks_toward_me():
    env = StickFightEnv(opponent=_RelIdleWalker(), me=0, seed=0, randomize_start=False)
    env.reset(seed=0)
    x0 = env.world.f[1].x
    for _ in range(10):
        env.step(Act.IDLE)
    assert env.world.f[1].x < x0


def test_set_opponent_uses_its_reaction():
    env = StickFightEnv(opponent=AggroBot(seed=0), seed=0)
    env.set_opponent(HardBot(seed=0) if "seed" in HardBot.__init__.__code__.co_varnames else HardBot())
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        obs, r, done, _, info = env.step(Act.IDLE)
    assert info["winner"] in (0, 1, -1)


def test_draw_reward():
    env = StickFightEnv(opponent=IdleBot(), seed=0, draw_reward=-0.3, shaping=0.0)
    env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        _, r, done, _, info = env.step(Act.IDLE)
        total += r
    assert info["winner"] == -1          # 둘 다 가만히 → 체력 같음 → 무승부
    assert abs(total - (-0.3)) < 1e-9
