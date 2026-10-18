"""AI의 반응 지연 (사람처럼 보고 반응하게 만드는 부분).

규칙 (프로젝트 설계 결정):
- 반응 시간은 130~240ms 삼각분포, 가장 많이 나오는 값 180ms.
- 상대가 "새 행동"을 시작할 때마다 반응 시간을 새로 뽑는다. 그 행동이 이어지는 동안은 유지.
- AI는 repeat(기본 4) 프레임마다 한 번 판단한다. 판단 시점까지 기다리는 시간도 반응에 포함되도록,
  목표 반응 시점에 가장 가까운 판단 시점에 처음 보이게 지연을 정한다 (pick_delay).
  130ms보다 빨라지면 다음 판단 시점으로 늦춘다.
- AI가 보는 시점은 절대 과거로 돌아가지 않는다.
- 지연은 상대 정보에만 적용한다 (내 상태는 바로 안다).

여기서 "프레임"은 실제로 흘러간 프레임이다 (히트스톱 중인 프레임도 센다).
판단 시점은 reset 이후 0, repeat, 2*repeat, ... 번째 프레임이다.
"""
import math
import random
from collections import deque

from game import config as C

# 상대가 이 상태로 바뀐 건 대부분 AI 자신이 원인이라 새 자극으로 보지 않는다
_NOT_STIMULUS = {C.LABELS.index("HITSTUN"), C.LABELS.index("PARRIED"), C.LABELS.index("KO")}
_IDLE = C.LABELS.index("IDLE")
PARRY_LABEL = C.LABELS.index("PARRY")

# Fighter.snapshot() 순서
X, Y, VX, VY, HP, FACING, LABEL, SF, STUN, RCD, DCD, PCD, STAGE, GAUGE, LOCK, WALL = range(16)


def ms_to_frames(ms: float) -> float:
    return ms * C.FPS / 1000.0


def pick_delay(s: int, T: float, repeat: int = 4, min_frames: int = 8) -> int:
    """프레임 s에 시작한 상대 행동을 목표 반응 시간 T(프레임) 뒤에 보게 하는 지연.

    목표 시점 s+T에 가장 가까운 판단 시점에 처음 보이도록 정한다.
    그 결과가 min_frames보다 빠르면 다음 판단 시점으로 늦춘다.
    """
    target = s + T
    lower = math.floor(target / repeat) * repeat
    upper = lower + repeat
    t = lower if (target - lower) <= (upper - target) else upper
    while t - s < min_frames:
        t += repeat
    return t - s


class Reaction:
    """반응 시간 분포."""

    def __init__(self, low_ms=130, mode_ms=180, high_ms=240, rng=None):
        self.low, self.mode, self.high = low_ms, mode_ms, high_ms
        self.rng = rng or random.Random()
        self.min_frames = math.ceil(ms_to_frames(low_ms))

    def sample_frames(self) -> float:
        return ms_to_frames(self.rng.triangular(self.low, self.high, self.mode))


def _walk_dir(o):
    if o[LABEL] != _IDLE or o[Y] > 0:
        return 0
    if o[VX] > 0.5:
        return 1
    if o[VX] < -0.5:
        return -1
    return 0


def is_new_action(prev, cur, prev_nproj, cur_nproj) -> bool:
    """상대가 화면에서 보이는 새 행동을 시작했는가.

    포함: 공격/원거리/패링/대쉬/차지 시작, 같은 행동 다시 시작, 차지 단계 변화, 손을 떼고 휘두르기,
          점프(땅 → 공중), 걷기 방향 전환(멈춤 포함), 원거리 투사체 발사
    제외: 같은 행동의 중간 프레임, 피격 경직/패링당함/KO로 바뀜
    """
    pl, cl = prev[LABEL], cur[LABEL]
    if cl != pl:
        return cl not in _NOT_STIMULUS
    if cl in _NOT_STIMULUS:
        return False
    if cl != _IDLE and cur[SF] < prev[SF]:          # 같은 행동을 다시 시작
        return True
    if cur[STAGE] != prev[STAGE]:                   # 차지 단계가 바뀜 (빛이 바뀜)
        return True
    if prev[Y] <= 0 < cur[Y]:                        # 점프
        return True
    if _walk_dir(cur) != _walk_dir(prev):            # 걷기 시작/멈춤/방향 전환
        return True
    return cur_nproj > prev_nproj                    # 투사체 발사


class Perception:
    """한 플레이어(me) 입장에서 상대를 보는 눈.

        p = Perception(me=0)
        p.reset(world)
        매 프레임 world.step(...) 뒤에 p.push(world)
        판단할 때 p.view() → 지연된 (상대 스냅샷, 상대 투사체 목록)
    """

    HIST = 64

    def __init__(self, me: int, repeat: int = 4, reaction: Reaction = None, seed=None):
        self.me = me
        self.op = 1 - me
        self.repeat = repeat
        self.reaction = reaction or Reaction(rng=random.Random(seed))
        self.reset_state()

    def reset_state(self):
        self.t = 0
        self.hist = deque(maxlen=self.HIST)   # (fighter snapshot, enemy projectiles)
        self.delay = 0
        self.last_seen = 0
        self.parry_seen = None    # AI가 마지막으로 본 상대 패링 시작 프레임
        self.seen = None          # 마지막 view() 결과 (SeenWorld용)
        self.stimuli = []         # (프레임, 정해진 지연) 기록 — 검증용

    def reset(self, world):
        self.reset_state()
        self._record(world)
        self.delay = pick_delay(0, self.reaction.sample_frames(), self.repeat,
                                self.reaction.min_frames)

    def _record(self, world):
        fighters, projs = world.snapshot()
        enemy = tuple(p for p in projs if p[0] == self.op)
        self.hist.append((fighters[self.op], enemy))

    def push(self, world):
        prev_o, prev_p = self.hist[-1]
        self.t += 1
        self._record(world)
        cur_o, cur_p = self.hist[-1]
        if is_new_action(prev_o, cur_o, len(prev_p), len(cur_p)):
            self.delay = pick_delay(self.t, self.reaction.sample_frames(), self.repeat,
                                    self.reaction.min_frames)
            self.stimuli.append((self.t, self.delay))

    def _at(self, frame):
        oldest = self.t - len(self.hist) + 1
        return self.hist[max(0, frame - oldest)]

    def view(self):
        """지금 판단에 쓰는 상대 정보 (지연 적용). 판단할 때마다 한 번 부른다."""
        v = max(self.t - self.delay, self.last_seen, self.t - len(self.hist) + 1, 0)
        # 지난 판단 이후 새로 보게 된 프레임들에서 상대 패링 시작을 찾는다 (쿨타임 추정용)
        for fr in range(self.last_seen + 1, v + 1):
            cur = self._at(fr)[0]
            prev = self._at(fr - 1)[0]
            if cur[LABEL] == PARRY_LABEL and (prev[LABEL] != PARRY_LABEL or cur[SF] < prev[SF]):
                self.parry_seen = fr
        self.last_seen = v
        o, projs = self._at(v)
        self.seen = (o, projs)
        return o, projs, v

    def since_parry(self, v):
        """AI가 본 화면 기준으로 상대가 마지막 패링을 시작한 뒤 지난 프레임 (못 봤으면 None)."""
        return None if self.parry_seen is None else v - self.parry_seen


# ---------------------------------------------------------------- 룰 기반 봇용 지연된 세계
_STATE_OF_LABEL = {0: "IDLE", 1: "ATTACK", 2: "ATTACK", 3: "ATTACK", 4: "RANGED", 5: "PARRY",
                   6: "DASH", 7: "HITSTUN", 8: "PARRIED", 9: "KO", 10: "ATTACK", 11: "CHARGE",
                   12: "RECOVER", 13: "BREAK", 14: "ATTACK", 15: "CROUCH"}
_KIND_OF_LABEL = {1: C.Act.HIGH, 2: C.Act.MID, 3: C.Act.LOW, 10: C.Act.HEAVY, 14: C.DASH_ATK}


class SeenFighter:
    """지연된 스냅샷으로 만든 상대 (봇이 world.f[상대]로 읽는 값)."""

    def __init__(self, idx, o, hp):
        self.idx = idx
        self.x, self.y, self.vx, self.vy = o[X], o[Y], o[VX], o[VY]
        self.facing, self.label, self.sf, self.stun = o[FACING], o[LABEL], o[SF], o[STUN]
        self.ranged_cd, self.dash_cd = o[RCD], o[DCD]
        self.charge_stage = o[STAGE]
        self.break_gauge, self.break_locked = o[GAUGE], o[LOCK]   # 화면 게이지·X 표시로 보임
        self.state = getattr(C.St, _STATE_OF_LABEL[o[LABEL]])
        self.kind = _KIND_OF_LABEL.get(o[LABEL])
        self.hp = hp

    @property
    def grounded(self):
        return self.y <= 0.0

    @property
    def gauge_full(self):
        return self.break_gauge >= C.BREAKER.gauge_max

    @property
    def can_break(self):
        return self.break_gauge >= C.BREAKER.use_min


class SeenProjectile:
    def __init__(self, p):
        self.owner, self.x, self.y, self.vx = p
        self.alive = True


class SeenWorld:
    """룰 기반 봇에게 주는 세계: 나는 지금 상태, 상대와 상대 투사체는 지연된 화면.
    build_obs(또는 perception.view())를 먼저 부른 뒤에 만든다."""

    def __init__(self, world, perception):
        me, op = perception.me, perception.op
        o, projs = perception.seen
        f = [None, None]
        f[me] = world.f[me]
        f[op] = SeenFighter(op, o, world.f[op].hp)
        self.f = f
        self.proj = [p for p in world.proj if p.owner == me] + [SeenProjectile(p) for p in projs]
        self.frame = world.frame
        self.done = world.done
