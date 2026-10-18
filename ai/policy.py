"""학습하는 AI의 신경망과, 학습된 체크포인트를 봇처럼 쓰는 래퍼 (v10, 1단계).

    net = ActorCritic()                     # 관측 80 → 256 → 256 → 행동 12개 확률 + 가치 1개
    bot = PolicyBot.load("checkpoints/stage1/best.pt")
    env = StickFightEnv(opponent=bot)       # 학습된 AI를 상대로 (env가 행동 방향을 바꿔 준다)

- 행동은 "상대가 오른쪽" 기준이다 (env의 relative_actions=True와 같은 기준). 그래서 PolicyBot은 relative_actions = True.
  play.py의 BotController도 이 속성을 보고 실제 화면 방향으로 바꿔 넣는다.
- PolicyBot은 obs만 쓴다 (world는 안 읽음). 반응 시간은 사람 기준(130~240ms)과 같다.
"""
import numpy as np
import torch
from torch import nn

from game import config as C
from .obs import OBS_DIM


def _layer(n_in, n_out, gain=np.sqrt(2)):
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    """정책(행동 확률)과 가치(앞으로 받을 보상 예상)가 몸통을 따로 쓰는 작은 MLP."""

    def __init__(self, obs_dim=OBS_DIM, n_actions=C.N_ACTIONS, hidden=256):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, hidden
        self.pi = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                _layer(hidden, n_actions, gain=0.01))
        self.v = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                               _layer(hidden, 1, gain=1.0))

    def forward(self, obs):
        return self.pi(obs), self.v(obs).squeeze(-1)

    def config(self):
        return {"obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": self.hidden}


def load_net(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["net_config"]
    if (cfg["obs_dim"], cfg["n_actions"]) != (OBS_DIM, C.N_ACTIONS):
        raise ValueError(f"{path}: 이전 버전 게임용 AI (관측 {cfg['obs_dim']} / 행동 {cfg['n_actions']}, "
                         f"지금 게임은 {OBS_DIM} / {C.N_ACTIONS}). 규칙이 바뀌어서 다시 학습해야 합니다.")
    net = ActorCritic(**cfg).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


class PolicyBot:
    """학습된 정책을 act(obs, world, me) 형태로 감싼다.
    greedy=False면 확률대로 고르고(학습 때와 같음), True면 가장 확률 높은 행동만."""

    relative_actions = True
    reaction_ms = (130, 180, 240)

    def __init__(self, net, greedy=False, seed=None):
        self.net = net
        self.greedy = greedy
        self.gen = torch.Generator().manual_seed(seed if seed is not None else int(np.random.randint(1 << 31)))

    @classmethod
    def load(cls, path, greedy=False, seed=None):
        net, _ = load_net(path)
        return cls(net, greedy, seed)

    @torch.no_grad()
    def act(self, obs, world=None, me=None):
        logits, _ = self.net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        if self.greedy:
            return int(logits.argmax(-1))
        return int(torch.multinomial(torch.softmax(logits, -1), 1, generator=self.gen))
