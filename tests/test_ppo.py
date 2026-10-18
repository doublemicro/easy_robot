"""v10: 1단계 학습 코드 테스트 — 신경망 크기, GAE, PolicyBot 대전, 저장/불러오기.

    python -m tests.test_ppo
"""
import os
import tempfile

import numpy as np
import torch

from game import config as C
from ai.bots import EasyBot
from ai.env import StickFightEnv
from ai.obs import OBS_DIM
from ai.policy import ActorCritic, PolicyBot, load_net
from ai.repeat import MoveTracker, check, summarize
from game.config import Act
from train import gae, save


def test_net_shapes():
    net = ActorCritic()
    logits, v = net(torch.zeros(5, OBS_DIM))
    assert logits.shape == (5, C.N_ACTIONS) and v.shape == (5,)


def test_gae_matches_monte_carlo_when_lambda_1():
    # λ=1, 가치 0이면 이득 = 판이 끝날 때까지 할인된 보상 합
    R = np.array([[1.0], [0.0], [2.0], [5.0]], np.float32)
    DN = np.array([[0], [0], [1], [0]], np.float32)     # 3번째 행동에서 판이 끝남
    V = np.zeros_like(R)
    adv, ret = gae(R, V, DN, np.array([10.0], np.float32), gamma=0.5, lam=1.0)
    assert np.allclose(adv[:, 0], [1 + 0.25 * 2, 0 + 0.5 * 2, 2, 5 + 0.5 * 10])
    assert np.allclose(ret, adv)


def test_policy_bot_plays_both_sides():
    net = ActorCritic()
    for side in (0, 1):
        env = StickFightEnv(opponent=EasyBot(seed=0), me=side, seed=side)
        bot = PolicyBot(net, seed=0)
        obs, _ = env.reset()
        done = False
        while not done:
            a = bot.act(obs)
            assert 0 <= a < C.N_ACTIONS
            obs, _, done, _, info = env.step(a)
        assert info["winner"] in (0, 1, -1)


def test_save_and_load_same_output():
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters())
    path = os.path.join(tempfile.mkdtemp(), "t.pt")

    class _A:
        pass
    save(path, net, opt, _A(), stage=1, steps=123)
    net2, ck = load_net(path)
    x = torch.randn(3, OBS_DIM)
    assert torch.allclose(net(x)[0], net2(x)[0]) and ck["steps"] == 123 and ck["stage"] == 1
    assert PolicyBot(net2, greedy=True).act(x[0].numpy()) == int(net(x[:1])[0].argmax())


def test_move_tracker_counts_new_moves_only():
    """MID를 계속 눌러도 한 번 나간 정면 공격은 1번으로 센다. 다시 나가면 또 센다."""
    env = StickFightEnv(opponent=EasyBot(seed=0), me=0, seed=0, randomize_start=False)
    env.reset(seed=0)
    tr = MoveTracker()
    tr.update(env.world.f[0])
    for _ in range(3):
        env.step(Act.MID)
        tr.update(env.world.f[0])
    assert tr.moves == ["MID"], tr.moves
    for _ in range(6):
        env.step(Act.MID)
        tr.update(env.world.f[0])
    assert tr.moves.count("MID") >= 2, tr.moves


def test_summarize_shares():
    s = summarize([["MID", "LOW", "MID", "LOW", "MID"], ["DASH"]])
    assert s["top_move"] == "MID" and abs(s["move_share"] - 0.5) < 1e-9
    assert s["top_gram"] == "MID > LOW > MID" and abs(s["gram_share"] - 2 / 3) < 1e-9



def test_check_picks_longest_pattern():
    """(v11) 원거리 > 대쉬 > 대쉬 공격 > 정면 > 정면 > 하단 을 계속 반복하면 6연속 패턴이 대표로 걸린다."""
    loop = ["RANGED", "DASH", "DASH_ATK", "MID", "MID", "LOW"]
    rep = summarize([loop * 8])
    assert rep["long"]["6"][0] == " > ".join(loop) or rep["long"]["6"][1] > 0.06
    flag = check(rep)
    assert flag is not None and flag["len"] == 6, flag
    # 섞여 있으면 안 걸린다
    import random
    rng = random.Random(0)
    names = ["HIGH", "MID", "LOW", "RANGED", "PARRY", "DASH", "JUMP", "CROUCH"]
    assert check(summarize([[rng.choice(names) for _ in range(300)]])) is None

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, e)
    raise SystemExit(1 if failed else 0)
