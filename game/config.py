"""게임 규칙과 수치. 밸런스 조정은 이 파일에서만 하면 된다.

단위: 거리 = 픽셀(월드 좌표, y는 바닥에서 위로), 시간 = 프레임(60fps).
"""
from dataclasses import dataclass
from enum import IntEnum

FPS = 60

# ---------------------------------------------------------------- 무대/물리
ARENA_W = 960.0
WALL_MARGIN = 20.0
GRAVITY = 0.8
WALK_SPEED = 4.0
BACK_WALK_SPEED = 4.0    # 뒤로 걷기 (상대 반대쪽). 지금은 앞과 같음 — 도망 견제용으로 낮출지 검토 중
AIR_ACCEL = 0.5          # 공중에서 좌우 입력 시 가속
JUMP_VY = 15.0           # 체공 약 37프레임, 최고 높이 약 140
GROUND_FRICTION = 0.8    # 경직/KO 중 지상 미끄러짐 감쇠

# ---------------------------------------------------------------- 캐릭터/라운드
MAX_HP = 100
ROUND_SECONDS = 60
ROUND_FRAMES = ROUND_SECONDS * FPS
START_X = (300.0, 660.0)
BODY_W = 30.0
BODY_H = 110.0
BODY_H_LOW = 72.0        # 하단 공격 중에는 자세가 낮아져 상단 공격/투사체를 피한다
BODY_H_CROUCH = 64.0     # (v13) 숙이기는 더 낮다 → 상단·투사체에 더해 정면(판정 68~100)도 피한다. 하단·대쉬 공격·강공격은 맞음

INPUT_BUFFER = 6         # 행동 입력을 몇 프레임 기억할지 (선입력)
HITSTOP_HIT = 4          # 히트 시 화면 정지 프레임
HITSTOP_PARRY = 8
HITSTOP_PARRY_PROJ = 3


class Act(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    DASH_F = 4    # 바라보는 방향으로 대쉬
    DASH_B = 5    # 뒤로 대쉬
    HIGH = 6      # 상단 공격
    MID = 7       # 정면(바라보는 방향) 공격
    LOW = 8       # 하단 공격
    RANGED = 9    # 원거리
    PARRY = 10
    HEAVY = 11    # 강공격 버튼을 누르고 있음 (길게 누르면 강공격, 패링 불가)
    CROUCH = 12   # (v11) 숙이기: 누르고 있는 동안 낮은 자세 (v13: 상단·투사체·정면을 피함, 하단·대쉬 공격·강공격은 맞음)


N_ACTIONS = len(Act)
MELEE = (Act.HIGH, Act.MID, Act.LOW, Act.HEAVY)
CANCELABLE = (Act.HIGH, Act.MID, Act.LOW)   # 콤보 캔슬은 이 셋끼리만 (강공격은 콤보에 못 넣음)
# 단발 입력 (선입력됨). HEAVY·CROUCH는 이동처럼 '누르고 있는' 입력이라 여기 없다.
BUFFERABLE = frozenset({Act.JUMP, Act.DASH_F, Act.DASH_B, Act.HIGH, Act.MID,
                        Act.LOW, Act.RANGED, Act.PARRY})


class Special(IntEnum):
    """입력 버튼이 아니라 상황으로 나가는 기술 (AI 행동 공간 N_ACTIONS에는 안 들어감)."""
    DASH_ATK = 20   # 대쉬 공격: 앞 대쉬 중에 정면 공격 버튼


DASH_ATK = Special.DASH_ATK


class St(IntEnum):
    IDLE = 0      # 자유 상태 (서기/걷기/공중)
    ATTACK = 1
    RANGED = 2
    PARRY = 3
    DASH = 4
    HITSTUN = 5   # 피격 경직
    PARRIED = 6   # 패링 당해서 크게 휘청임
    KO = 7
    CHARGE = 8    # 강공격 기 모으는 중
    RECOVER = 9   # 페인트 후 짧은 경직
    BREAK = 10    # 콤보 브레이커 발동 (전부 무적)
    CROUCH = 11   # (v11) 숙이는 중 (자유 상태처럼 바로 공격·점프·대쉬로 넘어갈 수 있음, 걷기는 안 됨)


@dataclass(frozen=True)
class AttackData:
    startup: int      # 예비동작 (판정 없음)
    active: int       # 판정 프레임
    recovery: int     # 후딜레이
    damage: int
    hitstun: int      # 맞은 쪽 경직
    knockback: float  # 맞은 쪽 수평 밀림 속도
    box: tuple        # 히트박스 (x0, x1, y0, y1), 발 기준, 앞쪽이 +x
    launch: float = 0.0  # 맞은 쪽을 띄우는 수직 속도
    parryable: bool = True

    @property
    def total(self):
        return self.startup + self.active + self.recovery


# 상단: 느리지만 강하고 공중을 잡는다 (점프 대응). 하단 자세에 빗나간다.
# 정면: 빠른 견제. 콤보 시동기.
# 하단: 사거리가 길다. 점프로 피할 수 있다.
ATTACKS = {
    Act.MID:  AttackData(7, 3, 12, 7, 20, 1.5, (12.0, 66.0, 68.0, 100.0)),
    # (v8) 후딜 16 → 24. 사거리 끝에서 하단만 연타하는 전략이 너무 셌음 (중간 봇 상대 99%).
    # 사거리 밖(100~120px)에서 헛치면 정면으로 벌하는 출발 타이밍이 5개 → 13개로 늘어 반응으로 벌할 수 있다.
    # (v11) 경직 22 → 30, 넉백 2.0 → 0.5 (후딜 24는 그대로). 맞혀도 -5라 상대가 바로 정면으로 반격했고,
    # 넉백 때문에 80px 이상에서 맞히면 캔슬이 하나도 안 이어졌음 → 맞히면 +3, 85px까지 정면 캔슬이 이어진다.
    # 사거리 끝(86~94px)에서 맞히면 여전히 안 이어짐 (안전하게 찌르기 vs 붙어서 콤보).
    Act.LOW:  AttackData(11, 3, 24, 9, 30, 0.5, (12.0, 80.0, 0.0, 34.0)),
    Act.HIGH: AttackData(14, 4, 18, 12, 24, 2.5, (8.0, 58.0, 78.0, 150.0), launch=7.0),
    # 강공격: 버튼을 누르고 있다가 떼면 이 데이터로 내려친다. 패링으로 받을 수 없다.
    # damage는 쓰지 않는다 (차지 단계별 데미지는 HEAVY_DAMAGE).
    # startup 10 = 손을 뗀 뒤 팔을 크게 들어올리는 시간. 가장 빠른 강공격도
    # 차지 시작부터 24프레임이 걸려서, 반응이 가장 느린 상대(240ms)도 보고 끊을 수 있다.
    # 헛치면 후딜이 길어서 크게 반격당한다.
    # (v8) 휘두르면서 반걸음 앞으로 내딛는다 (HeavyStepData) + 히트박스 앞 끝 78 → 100. 사거리가 정면·하단보다 길다.
    Act.HEAVY: AttackData(10, 4, 26, 16, 28, 6.0, (14.0, 100.0, 20.0, 112.0), parryable=False),
    # 대쉬 공격: 앞 대쉬 중에 정면 버튼 → 대쉬 속도를 그대로 싣고 낮게 찌르며 미끄러진다.
    # 도망가는 상대를 잡는 기술. 사거리(미끄러짐 포함)가 길지만 패링되고, 헛치면 후딜 20으로 크게 반격당한다.
    # 판정 높이 44~96 → 하단 자세(72)도 맞지만 점프로 피할 수 있다. 맞히면 상단/정면/하단으로 캔슬 가능.
    DASH_ATK: AttackData(7, 4, 20, 8, 22, 3.0, (10.0, 78.0, 44.0, 96.0)),
}


# 강공격 차지 단계. 누른 시간(프레임) → 단계
#   0단계  0~13  떼면 페인트 (HEAVY_FEINT_LAG 경직), 빛 없음
#   1단계 14~27  데미지 12, 약한 주황빛
#   2단계 28~43  데미지 16, 밝은 주황빛
#   3단계 44~59  데미지 22, 가장 밝은 빛 + 깜빡임
#   60프레임(1초)이 되면 무조건 3단계로 나간다
HEAVY_STAGE_START = (14, 28, 44)
HEAVY_DAMAGE = (12, 16, 22)
HEAVY_MAX_CHARGE = 60
HEAVY_FEINT_LAG = 6


def heavy_stage(held: int) -> int:
    """누른 프레임 수 → 차지 단계 (0~3)."""
    s = 0
    for k, start in enumerate(HEAVY_STAGE_START):
        if held >= start:
            s = k + 1
    return s


@dataclass(frozen=True)
class RangedData:
    startup: int = 14
    recovery: int = 16
    cooldown: int = 120    # (v6) 75 → 120. 대쉬 무적이 없어져 투사체를 대쉬로 못 뚫어서, 쏘는 사이에 걸어서 약 480px(화면 절반) 다가갈 시간을 줌
    speed: float = 8.0
    damage: int = 6
    hitstun: int = 16
    knockback: float = 3.0
    height: float = 86.0   # 발사 높이 (머리 높이: 하단 자세로 숙이면 피함)
    radius: float = 8.0
    spawn_x: float = 30.0

    @property
    def total(self):
        return self.startup + self.recovery


@dataclass(frozen=True)
class ParryData:
    startup: int = 2
    active: int = 8       # 이 구간에 맞으면 패링 성공
    recovery: int = 20    # 헛치면 이만큼 무방비
    stun: int = 30        # 패링당한 쪽 경직
    cooldown: int = 40    # 패링 시작부터 다시 쓸 수 있을 때까지 (끝나고 10프레임 대기)

    @property
    def total(self):
        return self.startup + self.active + self.recovery


@dataclass(frozen=True)
class DashData:
    duration: int = 12
    speed: float = 12.0
    invuln_start: int = 1   # 앞 대쉬 무적 구간 [start, end). start == end 이면 무적 없음
    invuln_end: int = 1     # (v6) 무적 없음. 1~7이면 무적으로 상대 공격을 지나친 뒤 대쉬 공격으로 때리는 연타가 너무 셌음
    back_invuln: bool = False  # 백대쉬 무적 없음 (v6). 넓은 곳에선 거리로 피하고, 구석에선 못 피함
    recovery: int = 6        # 앞 대쉬 후딜
    back_recovery: int = 12  # 백대쉬 후딜 (도망 견제: 백대쉬 끝을 대쉬 공격·원거리로 잡을 수 있게)
    cooldown: int = 45

    def rec(self, forward: bool) -> int:
        return self.recovery if forward else self.back_recovery


@dataclass(frozen=True)
class DashAttackData:
    """대쉬 공격 입력과 움직임.

    앞 대쉬(땅) 중 window_start 프레임째부터 대쉬 이동이 끝날 때까지(DASH.duration) 정면 버튼을 누르면 나간다.
    대쉬 무적은 공격으로 바뀌는 순간 사라진다. 대쉬 쿨타임은 그대로 남는다.
    시작 속도 lunge_speed에서 매 프레임 lunge_decay배로 줄며 미끄러진다 (판정 끝까지 약 60px 전진).
    """
    window_start: int = 3
    lunge_speed: float = 11.0
    lunge_decay: float = 0.85
    stop_decay: float = 0.6      # 후딜 중 감속


@dataclass(frozen=True)
class HeavyStepData:
    """(v8) 강공격 내딛기: 팔을 끝까지 들어 올린 뒤 내려치기 직전(손을 떼고 step_start 프레임째)에
    step_speed로 앞발을 내딛고, 판정이 끝날 때까지 매 프레임 step_decay배로 줄어든다 (약 25px).
    후딜은 stop_decay로 멈춘다. 돌진이 아니라 체중을 실어 한 발 내딛는 정도. 기 모으는 동안과
    팔을 들어 올리는 동안은 제자리."""
    step_start: int = 8
    step_speed: float = 6.0
    step_decay: float = 0.85
    stop_decay: float = 0.5


@dataclass(frozen=True)
class WallData:
    """벽꽝: 경직 중에 min_speed 이상의 속도로 벽에 부딪히면 extra_stun만큼 경직이 늘고 살짝 튕겨 나온다.
    한 콤보(경직이 끝날 때까지)에 한 번만. 콤보 브레이커에 밀려난 공격자는 벽꽝이 없다.
    구석에 몰린 채 맞으면 손해가 커져서, 계속 뒤로 도망가는 것에 비용이 생긴다."""
    min_speed: float = 1.5
    extra_stun: int = 16
    bounce: float = 1.5
    hitstop: int = 6


@dataclass(frozen=True)
class ComboData:
    max_chain: int = 4      # 한 콤보에 이을 수 있는 최대 공격 수
    cancel_window: int = 10  # 히트 후 이 프레임 안에 다른 공격을 누르면 캔슬
    scaling: float = 0.12   # 콤보 한 타마다 데미지 감소율
    min_scale: float = 0.4


@dataclass(frozen=True)
class BreakerData:
    """콤보 브레이커.

    - 게이지는 맞을 때 받은 데미지만큼 찬다 (gain_per_damage). use_min 이상이면 쓸 수 있고, 쓰면 전부 소모.
    - 콤보 2타째부터 끊을 수 있다: 상대가 캔슬로 이어 낸 공격(chain >= 2)을 하고 있을 때,
      경직 중에 그 공격과 같은 버튼(상단/정면/하단)을 누르면 끊긴다.
    - 다른 버튼을 누르면 그 콤보가 끝날 때까지 브레이커가 잠긴다 (게이지는 남음).
    - 성공: 게이지를 전부 쓰고, 나는 burst 프레임 동안 무적 동작, 상대는 밀려나며 attacker_stun 경직.
      burst < attacker_stun 이라 내가 살짝 먼저 움직이지만, 거리가 벌어져 확정 반격은 없다.
    - 게이지가 가득 찬 동안에는 내 모든 공격 데미지가 full_bonus 만큼 늘어난다
      (게이지를 아껴 딜을 넣을지, 써서 콤보를 끊을지 고르는 선택).
    - 마지막으로 맞은 뒤 decay_delay 프레임이 지나면 게이지가 decay_per_frame 씩 줄어든다.
      MAX 아래로 내려가는 순간 데미지 증가와 브레이커가 사라진다 (맞은 효과가 계속 남지 않게).
      다시 맞으면 타이머가 처음부터.
    """
    gauge_max: float = 100.0
    gain_per_damage: float = 2.5   # 받은 데미지 40이면 가득
    use_min: float = 60.0          # 이 이상이면 브레이커 사용 가능 (받은 데미지 24)
    burst: int = 16
    attacker_stun: int = 20
    push: float = 9.0
    hitstop: int = 10
    full_bonus: float = 0.15       # 데미지 +15% (정면 7→8, 상단 12→14, 강공격 3단계 22→25)
    decay_delay: int = 180         # 마지막으로 맞은 뒤 3초 동안은 안 줄어듦
    decay_per_frame: float = 0.2   # 그 뒤 초당 12씩 감소 (가득 → 0 약 8.3초)


RANGED = RangedData()
PARRY = ParryData()
DASH = DashData()
DASH_ATTACK = DashAttackData()
HEAVY_STEP = HeavyStepData()
WALL = WallData()
COMBO = ComboData()
BREAKER = BreakerData()

# 관측용 상태 라벨 (공격은 종류별로 나눔)
LABELS = ("IDLE", "HIGH", "MID", "LOW", "RANGED", "PARRY", "DASH",
          "HITSTUN", "PARRIED", "KO", "HEAVY", "CHARGE", "RECOVER", "BREAK",
          "DASH_ATK", "CROUCH")
N_LABELS = len(LABELS)
