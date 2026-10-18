"""강화학습용 환경 (Gymnasium 스타일 reset/step).

    env = StickFightEnv(opponent=AggroBot())
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

- action_repeat 프레임마다 한 번 결정한다 (기본 4 = 초당 15회).
  이동과 강공격 누르기는 그 동안 유지하고, 점프/공격 같은 단발 행동은 첫 프레임에만 입력한다.
- 두 플레이어 모두 각자의 Perception(반응 지연된 눈)으로 상대를 본다.
  반응 시간은 reaction_ms = (최소, 최빈, 최대) 삼각분포. 판단 주기 대기 시간까지 포함한 값이다.
- 상대(opponent)는 act(obs, world, me) 를 가진 객체 (reaction_ms 속성이 있으면 그 반응 시간을 쓴다). 학습된 정책도 같은 형태로 감싸면 된다.
  world 자리에는 SeenWorld(상대가 지연된 세계)를 넘긴다 → 룰 기반 봇도 같은 반응 지연을 받는다.
- (v9) 관측은 항상 "상대가 오른쪽"으로 뒤집혀 있으므로, 학습하는 쪽 행동의 LEFT/RIGHT도 같은 기준이다.
  relative_actions=True(기본)면 RIGHT = 상대 쪽으로, LEFT = 상대 반대쪽으로 걷기. env가 실제 방향으로 바꿔 넣는다.
  상대(opponent)가 relative_actions = True 속성을 가지면(학습된 정책을 상대로 쓸 때) 그쪽 행동도 똑같이 바꾼다.
  룰 기반 봇은 원래대로 실제 방향(LEFT = 화면 왼쪽)을 쓴다.
- (v9) set_opponent()로 판 사이에 상대를 바꿀 수 있다 (셀프플레이 리그용).
- (v9) draw_reward: 시간 끝 무승부일 때 받는 보상 (기본 0, 소극적 플레이를 막으려면 음수).
"""
import random

from game import config as C
from game.config import Act
from game.core import World
from .bots import IdleBot
from .obs import OBS_DIM, build_obs
from .perception import Perception, Reaction, SeenWorld


class StickFightEnv:
    observation_size = OBS_DIM
    n_actions = C.N_ACTIONS

    def __init__(self, opponent=None, me=0, action_repeat=4, reaction_ms=(130, 180, 240),
                 shaping=0.5, randomize_start=True, seed=None,
                 relative_actions=True, draw_reward=0.0):
        self.me = me
        self.relative_actions = relative_actions
        self.draw_reward = draw_reward
        self.default_ms = reaction_ms
        self.repeat = action_repeat
        self.shaping = shaping
        self.randomize_start = randomize_start
        self.rng = random.Random(seed)
        self.world = World()
        self._side = [1.0, 1.0]     # 마지막 관측에서 상대가 실제로 어느 쪽에 보였나 (+1 오른쪽 / -1 왼쪽)
        self.set_opponent(opponent or IdleBot())

    def set_opponent(self, opponent):
        """상대를 바꾼다. 다음 reset()부터 적용하는 게 안전하다."""
        self.opponent = opponent
        # 상대 봇이 자기 반응 시간(reaction_ms)을 갖고 있으면 그 눈에는 그 값을 쓴다 (하위 봇은 더 느림)
        op_ms = getattr(opponent, "reaction_ms", self.default_ms)
        self.eyes = [Perception(i, self.repeat,
                                Reaction(*(self.default_ms if i == self.me else op_ms),
                                         rng=random.Random(self.rng.random())))
                     for i in (0, 1)]

    def _obs(self, who):
        """관측을 만들고, 그 관측이 어느 쪽 기준으로 뒤집혔는지 기억한다.
        obs[4] = 내 facing × side 이므로 side = obs[4] × facing."""
        o = build_obs(self.world, self.eyes[who])
        self._side[who] = float(o[4]) * self.world.f[who].facing
        return o

    def _to_world(self, a, who):
        """'상대가 오른쪽' 기준 행동 → 실제 화면 방향 행동."""
        if self._side[who] < 0 and a in (Act.LEFT, Act.RIGHT):
            return Act.RIGHT if a == Act.LEFT else Act.LEFT
        return a

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
            for e in self.eyes:
                e.reaction.rng.seed(self.rng.random())
        x0, x1 = C.START_X
        if self.randomize_start:
            x0 += self.rng.uniform(-80, 80)
            x1 += self.rng.uniform(-80, 80)
        self.world.reset(x0, x1)
        for e in self.eyes:
            e.reset(self.world)
        return self._obs(self.me), {}

    def step(self, action):
        w, me = self.world, self.me
        op = 1 - me
        hp_me, hp_op = w.f[me].hp, w.f[op].hp
        a_me = Act(int(action))
        if self.relative_actions:
            a_me = self._to_world(a_me, me)
        obs_op = self._obs(op)
        a_op = Act(int(self.opponent.act(obs_op, SeenWorld(w, self.eyes[op]), op)))
        if getattr(self.opponent, "relative_actions", False):
            a_op = self._to_world(a_op, op)

        events = []
        for k in range(self.repeat):
            acts = [Act.IDLE, Act.IDLE]
            acts[me] = a_me if (k == 0 or a_me not in C.BUFFERABLE) else Act.IDLE
            acts[op] = a_op if (k == 0 or a_op not in C.BUFFERABLE) else Act.IDLE
            events += w.step(*acts)
            for e in self.eyes:
                e.push(w)
            if w.done:
                break

        dealt = hp_op - w.f[op].hp
        taken = hp_me - w.f[me].hp
        reward = self.shaping * (dealt - taken) / C.MAX_HP
        if w.done:
            reward += 1.0 if w.winner == me else (-1.0 if w.winner == op else self.draw_reward)
        obs = self._obs(me)
        return obs, reward, w.done, False, {"events": events, "winner": w.winner}
