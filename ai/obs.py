"""AI가 보는 관측 벡터.

사람이 화면에서 알 수 있는 것만 준다 (프로젝트 설계 결정).
- 내 상태: 지연 없이 (내 쿨타임, 차지 단계, 콤보 가능 여부 등)
- 상대 상태와 상대 투사체: Perception을 거쳐 반응 지연된 화면 기준
- 상대 체력: 지연 없이 (AI가 때릴 때만 바뀌므로)
- 상대 패링 쿨타임: 화면에 안 보이므로 주지 않는다.
  대신 AI가 본 화면에서 상대가 마지막으로 패링을 시작한 뒤 지난 시간을 준다 (사람이 속으로 세는 것).
- 상대 원거리/대쉬 쿨타임: 화면 위 게이지로 보이므로 지연된 값을 준다.
- 상대 차지 단계: 빛의 밝기로 보이는 단계(0~3)만 준다. 정확한 누른 프레임 수는 주지 않는다.
- 콤보 브레이커: 내 게이지·잠김·게이지가 줄어들기 시작할 때까지 남은 시간은 바로, 상대 게이지·잠김(화면 게이지와 머리 위 X로 보임)은 지연된 값.
  상대가 지금 몇 번째 콤보 공격(chain)인지는 주지 않는다 — 화면에 나오는 공격 종류와 프레임으로 판단.
- 벽꽝: 이번 콤보에서 이미 벽꽝을 당했는지 (나: 바로, 상대: 지연). 한 콤보에 한 번이라 다음 벽꽝이 되는지 판단용.
좌우는 항상 "상대가 오른쪽"이 되도록 뒤집어서 준다.
"""
import numpy as np

from game import config as C
from .perception import (DCD, FACING, GAUGE, LABEL, LOCK, RCD, SF, STAGE, STUN, VX, VY, WALL,
                         X, Y, Perception)


def _onehot(i, n=C.N_LABELS):
    v = [0.0] * n
    v[i] = 1.0
    return v


SELF_DIM = 6 + C.N_LABELS + 9 + 1 + 2 + 3 + 1
OPP_DIM = 7 + C.N_LABELS + 4 + 1 + 1 + 2 + 1
PROJ_DIM = 2 * 4 + 3
OBS_DIM = SELF_DIM + OPP_DIM + PROJ_DIM + 1


def build_obs(world, perception: Perception) -> np.ndarray:
    """판단 시점에 한 번 부른다 (perception.view()가 보는 시점을 앞으로 옮긴다)."""
    me = perception.me
    W = C.ARENA_W
    f = world.f[me]
    o, enemy_projs, v = perception.view()
    ox, oy = o[X], o[Y]
    s = 1.0 if ox >= f.x else -1.0

    # ---- 나 (지연 없음)
    vec = [f.y / 150.0, f.vx * s / 15.0, f.vy / 15.0, f.hp / C.MAX_HP,
           f.facing * s, 1.0 if f.grounded else 0.0]
    vec += _onehot(f.label)
    vec += [min(f.sf, 60) / 60.0, f.stun / 30.0,
            f.ranged_cd / C.RANGED.cooldown, f.dash_cd / C.DASH.cooldown,
            f.parry_cd / C.PARRY.cooldown, float(f.air_dash_used), f.chain / C.COMBO.max_chain,
            float(f.can_cancel()), min(f.combo_taken, 4) / 4.0]
    vec.append(f.charge_stage / 3.0)
    back = f.x / W if s > 0 else (W - f.x) / W
    vec += [back, 1.0 - back]
    bk = C.BREAKER
    vec += [f.break_gauge / bk.gauge_max, float(f.break_locked),
            max(0, bk.decay_delay - f.break_idle) / bk.decay_delay]
    vec.append(float(f.wall_hit))          # 이번 콤보에서 벽꽝을 이미 당함 (화면에서 봤음)

    # ---- 상대 (지연된 화면)
    vec += [(ox - f.x) * s / W, (oy - f.y) / 150.0, o[VX] * s / 15.0, o[VY] / 15.0,
            world.f[1 - me].hp / C.MAX_HP, o[FACING] * s, 1.0 if oy <= 0 else 0.0]
    vec += _onehot(o[LABEL])
    vec += [min(o[SF], 60) / 60.0, o[STUN] / 30.0, o[RCD] / C.RANGED.cooldown,
            o[DCD] / C.DASH.cooldown]
    vec.append(o[STAGE] / 3.0)
    since = perception.since_parry(v)
    vec.append(1.0 if since is None else min(since, C.PARRY.cooldown) / C.PARRY.cooldown)
    vec += [o[GAUGE] / C.BREAKER.gauge_max, float(o[LOCK])]
    vec.append(float(o[WALL]))

    # ---- 투사체: 상대 것은 지연된 화면, 내 것은 지금
    enemy = sorted(enemy_projs, key=lambda p: abs(p[1] - f.x))[:2]
    for i in range(2):
        if i < len(enemy):
            _, px, py, pvx = enemy[i]
            vec += [1.0, (px - f.x) * s / W, (py - f.y) / 150.0, pvx * s / 15.0]
        else:
            vec += [0.0, 0.0, 0.0, 0.0]
    mine = [p for p in world.proj if p.owner == me and p.alive][:1]
    if mine:
        p = mine[0]
        vec += [1.0, (p.x - f.x) * s / W, (p.y - f.y) / 150.0]
    else:
        vec += [0.0, 0.0, 0.0]

    vec.append((C.ROUND_FRAMES - world.frame) / C.ROUND_FRAMES)
    out = np.asarray(vec, dtype=np.float32)
    assert out.shape == (OBS_DIM,)
    return out
