"""(v16) 학습된 AI를 numpy만으로 돌린다 — 게임(PC·웹)용. 학습(train.py)은 그대로 PyTorch(ai/policy.py).

    bot = NumpyPolicyBot.load("models/ai.npz")

- models/ai.npz 는 tools/export_ai.py 가 ai.pt 와 같이 만든다 (정책 쪽 가중치만: 관측 → 256 → 256 → 행동).
- PyTorch가 없어도 되고, 브라우저(pygbag)에서도 돌아간다.
- 계산은 ai/policy.py의 ActorCritic.pi 와 같다 (Linear → tanh → Linear → tanh → Linear). 가치 쪽은 게임에 필요 없어 뺐다.
"""
import numpy as np

from game import config as C
from .obs import OBS_DIM


class NumpyPolicyBot:
    relative_actions = True          # 행동이 "상대가 오른쪽" 기준 (env·BotController가 바꿔 넣음)
    reaction_ms = (130, 180, 240)

    def __init__(self, layers, greedy=False, seed=None):
        self.layers = layers         # [(W, b), ...]  W: (입력, 출력)
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, path, greedy=False, seed=None):
        d = np.load(path)
        n_in, n_out = int(d["obs_dim"]), int(d["n_actions"])
        if (n_in, n_out) != (OBS_DIM, C.N_ACTIONS):
            raise ValueError(f"{path}: 이전 버전 게임용 AI (관측 {n_in} / 행동 {n_out}, "
                             f"지금 게임은 {OBS_DIM} / {C.N_ACTIONS}). 규칙이 바뀌어서 다시 학습해야 합니다.")
        layers = [(d[f"w{k}"].astype(np.float32), d[f"b{k}"].astype(np.float32)) for k in range(3)]
        return cls(layers, greedy, seed)

    def logits(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for k, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if k < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def act(self, obs, world=None, me=None):
        z = self.logits(obs)
        if self.greedy:
            return int(np.argmax(z))
        p = np.exp(z - z.max())
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))


def export_npz(net, path):
    """ActorCritic(torch) → npz. net.pi = [Linear, Tanh, Linear, Tanh, Linear]."""
    lin = [m for m in net.pi if hasattr(m, "weight")]
    arrays = {"obs_dim": net.obs_dim, "n_actions": net.n_actions}
    for k, m in enumerate(lin):
        arrays[f"w{k}"] = m.weight.detach().cpu().numpy().T.astype(np.float32)
        arrays[f"b{k}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)
