"""졸라맨 포즈와 관절 계산 (렌더링 전용, 게임 로직에는 영향 없음).

포즈 = (px, ph, torso, head, af_u, af_l, ab_u, ab_l, ffx, ffy, bfx, bfy)
  px, ph   : 골반 위치 (발 기준, 앞쪽 +x)
  torso    : 상체 기울기 (0 = 수직, + = 앞으로)
  head     : 상체 대비 고개 기울기
  af_*/ab_*: 앞팔/뒷팔의 위팔·아래팔 각도 (0 = 아래로 늘어뜨림, 90 = 앞으로 수평, 180 = 위)
  ff*/bf*  : 앞발/뒷발 목표 위치 → 무릎은 IK로 계산 (발이 바닥에 붙어 미끄러지지 않음)
행동마다 키포즈를 두고, 행동의 진행 프레임에 맞춰 보간한다.
"""
import math

from . import config as C
from .config import Act, St

TORSO = 36.0
NECK_HEAD = 14.0
HEAD_R = 11.0
UARM = 22.0
LARM = 22.0
THIGH = 28.0
SHIN = 28.0

FIELDS = ("px", "ph", "torso", "head", "af_u", "af_l", "ab_u", "ab_l",
          "ffx", "ffy", "bfx", "bfy")
_BASE = dict(px=0, ph=50, torso=4, head=0, af_u=35, af_l=110, ab_u=15, ab_l=95,
             ffx=16, ffy=0, bfx=-18, bfy=0)


def pose(**kw):
    d = dict(_BASE)
    d.update(kw)
    return tuple(float(d[k]) for k in FIELDS)


P = {
    "idle": pose(),
    # 점프 (상승/하강)
    "rise": pose(ph=56, torso=6, af_u=70, af_l=150, ab_u=-25, ab_l=30,
                 ffx=14, ffy=26, bfx=-12, bfy=14),
    "fall": pose(ph=56, torso=2, af_u=50, af_l=120, ab_u=40, ab_l=110,
                 ffx=12, ffy=8, bfx=-16, bfy=2),
    # ---- 공격 키포즈 (v5): 예비1 → 예비2 → 휘두름 → 끝까지 뻗음 → 판정 자세 → 팔로스루
    # 세 공격은 첫 2~3프레임부터 손/발이 가는 방향이 다르다:
    #   상단 = 주먹이 아래-뒤로 떨어지며 웅크림 / 정면 = 주먹이 어깨 높이로 뒤로 당겨짐
    #   하단 = 몸 전체가 앞으로 숙이며 내려가고 앞무릎이 들림
    # 상단(어퍼): 웅크리며 주먹을 아래 뒤로 → 발끝으로 일어서며 위로 크게 올려친다
    "high_a1": pose(ph=44, torso=-8, head=4, af_u=-35, af_l=-15, ab_u=50, ab_l=115,
                    ffx=18, bfx=-24),
    "high_a": pose(ph=39, torso=-16, head=8, af_u=-62, af_l=-40, ab_u=45, ab_l=95,
                   ffx=19, bfx=-27),
    "high_w": pose(ph=47, torso=2, head=2, af_u=55, af_l=70, ab_u=10, ab_l=60,
                   ffx=15, bfx=-24),
    "high_o": pose(ph=58, torso=17, head=-10, af_u=158, af_l=170, ab_u=-42, ab_l=-8,
                   ffx=12, bfx=-20),
    "high_s": pose(ph=55, torso=14, head=-8, af_u=140, af_l=150, ab_u=-35, ab_l=0,
                   ffx=12, bfx=-22),
    "high_f": pose(ph=52, torso=8, head=-4, af_u=120, af_l=105, ab_u=-10, ab_l=40,
                   ffx=13, bfx=-22),
    # 정면(스트레이트): 주먹을 어깨 높이로 뒤로 당기고 뒷손은 앞으로 가드 → 곧게 지른다
    "mid_a1": pose(ph=48, torso=-5, af_u=-78, af_l=78, ab_u=70, ab_l=140,
                   ffx=18, bfx=-22),
    "mid_a": pose(ph=46, torso=-11, af_u=-98, af_l=72, ab_u=82, ab_l=150,
                  ffx=20, bfx=-24),
    "mid_w": pose(px=4, ph=47, torso=6, af_u=15, af_l=92, ab_u=35, ab_l=105,
                  ffx=26, bfx=-21),
    "mid_o": pose(px=10, ph=46, torso=22, af_u=92, af_l=92, ab_u=-35, ab_l=20,
                  ffx=34, bfx=-20),
    "mid_s": pose(px=8, ph=47, torso=18, af_u=90, af_l=90, ab_u=-25, ab_l=25,
                  ffx=32, bfx=-20),
    "mid_f": pose(px=5, ph=47, torso=11, af_u=75, af_l=100, ab_u=5, ab_l=70,
                  ffx=28, bfx=-21),
    # 하단(쓸어차기): 앞으로 숙이며 낮아지고 → 앞무릎을 들어 모았다가 → 바닥을 쓸듯 찬다
    "low_a1": pose(ph=39, torso=20, head=-8, af_u=45, af_l=95, ab_u=20, ab_l=75,
                   ffx=21, bfx=-24),
    "low_a": pose(ph=33, torso=28, head=-10, af_u=55, af_l=105, ab_u=30, ab_l=90,
                  ffx=12, ffy=11, bfx=-26),
    "low_w": pose(px=4, ph=31, torso=8, head=0, af_u=-5, af_l=20, ab_u=-25, ab_l=-10,
                  ffx=46, ffy=15, bfx=-27),
    "low_o": pose(px=7, ph=29, torso=-15, head=12, af_u=-48, af_l=-26, ab_u=-68,
                  ab_l=-56, ffx=78, ffy=10, bfx=-28),
    "low_s": pose(px=6, ph=30, torso=-12, head=10, af_u=-40, af_l=-20, ab_u=-60,
                  ab_l=-50, ffx=74, ffy=12, bfx=-28),
    "low_f": pose(px=4, ph=33, torso=-2, head=4, af_u=-10, af_l=20, ab_u=-30, ab_l=-10,
                  ffx=56, ffy=5, bfx=-27),
    # (v11) 숙이기: 무릎을 깊게 굽혀 골반을 내리고 상체를 앞으로 숙여 머리를 판정 높이(v13: 64) 아래로.
    # 두 주먹은 얼굴 앞 가드, 고개는 들어 상대를 본다. 하단 공격 자세보다 웅크린 실루엣 (공격이 아님을 구분)
    "crouch": pose(px=-2, ph=20, torso=55, head=-30, af_u=66, af_l=158, ab_u=44, ab_l=152,
                   ffx=22, bfx=-6),
    # (v15) 승리 화면: 웅크려 힘을 모았다가 → 앞주먹을 하늘로 뻗고 뒷주먹은 허리에, 고개를 든다
    "win_a": pose(ph=43, torso=12, head=8, af_u=15, af_l=150, ab_u=5, ab_l=140, ffx=18, bfx=-22),
    "win": pose(ph=53, torso=-6, head=-16, af_u=172, af_l=178, ab_u=-22, ab_l=92, ffx=20, bfx=-22),
    # 강공격: 두 팔을 머리 뒤로 크게 들어올려 몸을 젖히고 → 앞으로 내려찍는다
    "heavy_a": pose(px=-6, ph=44, torso=-26, head=10, af_u=200, af_l=215, ab_u=190,
                    ab_l=205, ffx=24, bfx=-28),
    # 손을 뗀 뒤 휘두르기 직전: 발끝으로 서며 팔을 머리 위 뒤쪽으로 끝까지 들어올린다
    "heavy_up": pose(px=-8, ph=52, torso=-32, head=12, af_u=215, af_l=230, ab_u=205,
                     ab_l=222, ffx=22, bfx=-26),
    # (v8) 휘두르며 앞발을 크게 내딛는다: 앞발이 앞으로 뻗고 뒷발은 끌려오며, 골반·상체가 앞으로 실린다
    "heavy_w": pose(px=8, ph=48, torso=10, head=0, af_u=150, af_l=140, ab_u=145,
                    ab_l=138, ffx=44, ffy=6, bfx=-30),
    "heavy_s": pose(px=18, ph=38, torso=42, head=-8, af_u=88, af_l=78, ab_u=84,
                    ab_l=80, ffx=56, bfx=-30),
    "heavy_f": pose(px=18, ph=35, torso=48, head=-10, af_u=66, af_l=52, ab_u=62,
                    ab_l=54, ffx=56, bfx=-28),
    # 원거리: 뒷손으로 상대를 겨누고, 앞손은 귀 뒤로 젖혔다가 → 던지듯 뻗는다
    # (각도를 360 쪽으로 써서 보간이 머리 위로 지나가게 = 오버핸드)
    "rng_a1": pose(ph=48, torso=-6, af_u=275, af_l=220, ab_u=80, ab_l=88,
                   ffx=18, bfx=-23),
    "rng_a": pose(ph=48, torso=-14, af_u=250, af_l=200, ab_u=88, ab_l=92,
                  ffx=19, bfx=-25),
    "rng_w": pose(px=2, ph=48, torso=4, af_u=150, af_l=130, ab_u=30, ab_l=50,
                  ffx=24, bfx=-23),
    "rng_s": pose(px=4, ph=48, torso=16, af_u=88, af_l=95, ab_u=-30, ab_l=10,
                  ffx=28, bfx=-22),
    # 패링: 두 팔을 앞에서 교차
    "parry": pose(ph=47, torso=-6, head=-4, af_u=55, af_l=165, ab_u=48, ab_l=150,
                  ffx=14, bfx=-22),
    # 대쉬
    "dash_f": pose(px=6, ph=46, torso=32, head=-12, af_u=-45, af_l=-25, ab_u=-55,
                   ab_l=-35, ffx=30, ffy=8, bfx=-34, bfy=12),
    "dash_b": pose(px=-4, ph=48, torso=-20, head=8, af_u=45, af_l=125, ab_u=30,
                   ab_l=105, ffx=26, ffy=6, bfx=-24, bfy=10),
    # 대쉬 공격: 대쉬 자세(앞으로 숙임, 팔 뒤로) 그대로 미끄러지며 → 앞주먹을 옆구리로 당기고
    # → 몸을 낮게 던지며 길게 찌른다. 정면 공격보다 상체가 훨씬 숙여지고 다리가 크게 벌어져 구분된다.
    "dsh_a1": pose(px=6, ph=44, torso=28, head=-12, af_u=-60, af_l=55, ab_u=60, ab_l=130,
                   ffx=30, ffy=4, bfx=-34, bfy=8),
    "dsh_a": pose(px=6, ph=42, torso=22, head=-10, af_u=-88, af_l=68, ab_u=72, ab_l=145,
                  ffx=32, ffy=2, bfx=-36, bfy=4),
    "dsh_w": pose(px=8, ph=41, torso=28, head=-12, af_u=25, af_l=85, ab_u=20, ab_l=90,
                  ffx=38, bfx=-34),
    "dsh_o": pose(px=14, ph=38, torso=38, head=-14, af_u=92, af_l=90, ab_u=-55, ab_l=-25,
                  ffx=46, bfx=-32),
    "dsh_s": pose(px=12, ph=39, torso=34, head=-12, af_u=90, af_l=90, ab_u=-48, ab_l=-15,
                  ffx=44, bfx=-32),
    "dsh_f": pose(px=8, ph=43, torso=20, head=-6, af_u=70, af_l=100, ab_u=0, ab_l=60,
                  ffx=36, bfx=-27),
    # 피격 / 패링당함 / KO
    "hit": pose(px=-6, ph=48, torso=-28, head=-22, af_u=-30, af_l=20, ab_u=60,
                ab_l=40, ffx=20, bfx=-14),
    "parried": pose(px=-4, ph=46, torso=-24, head=-16, af_u=150, af_l=175, ab_u=125,
                    ab_l=160, ffx=22, bfx=-26),
    # 콤보 브레이커: 가슴을 펴고 두 팔을 활짝 벌려 상대를 튕겨낸다
    "burst": pose(ph=54, torso=-10, head=-8, af_u=125, af_l=140, ab_u=-125, ab_l=-140,
                  ffx=22, bfx=-26),
    "ko": pose(px=-10, ph=10, torso=-84, head=-10, af_u=170, af_l=175, ab_u=110,
               ab_l=130, ffx=44, bfx=36),
}

ATTACK_KEYS = {Act.HIGH: "high", Act.MID: "mid", Act.LOW: "low", Act.HEAVY: "heavy",
               C.DASH_ATK: "dsh"}


def lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 2


def smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def _set(p, **kw):
    d = dict(zip(FIELDS, p))
    d.update(kw)
    return tuple(d[k] for k in FIELDS)


def _with_legs(p, q):
    """p의 상체 + q의 다리 (공중 공격용)."""
    return p[:1] + q[1:2] + p[2:8] + q[8:]


def air_pose(f):
    t = min(1.0, max(0.0, (f.vy + 12) / 24))
    return lerp(P["fall"], P["rise"], t)


def walk_pose(f, frame):
    phi = f.x * 0.085
    s, c = math.sin(phi), math.cos(phi)
    moving_fwd = (f.vx * f.facing) > 0
    return _set(P["idle"],
                ph=49 - 1.5 * abs(s), torso=8 if moving_fwd else 0,
                af_u=35 + 8 * s, ab_u=15 - 8 * s,
                ffx=16 + 13 * s, ffy=6 * max(0.0, c),
                bfx=-18 - 13 * s, bfy=6 * max(0.0, -c))


def idle_pose(frame):
    b = math.sin(frame * 0.07)
    return _set(P["idle"], ph=50 + 0.8 * b, af_u=35 + 2 * b, ab_u=15 + 1.5 * b)


def ease_in(t):
    t = min(1.0, max(0.0, t))
    return t * t


def _timeline(x, keys):
    """keys = [(프레임, 포즈, 이징), ...] (프레임 오름차순). x 프레임에서의 포즈."""
    if x <= keys[0][0]:
        return keys[0][1]
    for (f0, p0, _), (f1, p1, ez) in zip(keys, keys[1:]):
        if x <= f1:
            return lerp(p0, p1, ez((x - f0) / max(1e-6, f1 - f0)))
    return keys[-1][1]


def _strike_keys(name, base, S, A, R):
    """예비1 → 예비2 → 휘두름 → 끝까지 뻗음 → 판정 자세 → 팔로스루 → 기본.
    판정 프레임(S ~ S+A-1)은 config 값 그대로. 모션만 촘촘해진다."""
    k1 = max(2.0, round(S * 0.3))
    q = P
    return [(0, base, smooth),
            (k1, q[name + "_a1"], ease_out),          # 첫 2~3프레임: 공격마다 다른 실루엣
            (S - 1, q[name + "_a"], smooth),          # 가장 깊게 모음
            (S, q[name + "_w"], ease_in),             # 휘두르기 시작 (가속)
            (S + 1, q[name + "_o"], lerp_lin),        # 판정 첫 프레임: 끝까지 뻗음
            (S + A, q[name + "_s"], smooth),
            (S + A + R * 0.4, q[name + "_f"], smooth),  # 팔로스루
            (S + A + R, base, smooth)]


def lerp_lin(t):
    return min(1.0, max(0.0, t))


def target_pose(f, frame):
    st = f.state
    g = f.grounded
    base = idle_pose(frame)
    if st == St.IDLE:
        if not g:
            return air_pose(f)
        if abs(f.vx) > 0.5:
            return walk_pose(f, frame)
        return base
    if st == St.ATTACK:
        ad = C.ATTACKS[f.kind]
        name = ATTACK_KEYS[f.kind]
        x = f.sf + 1                      # 1프레임 앞을 보여줘서 입력 즉시 반응
        if f.kind == Act.HEAVY:   # 손을 떼면 팔을 끝까지 들어올렸다가 → 한 번에 내려찍는다
            S, A, R = ad.startup, ad.active, ad.recovery
            # (v8) S-2부터 몸이 앞으로 나가므로 그때부터 발을 내딛기 시작한다
            p = _timeline(x, [(0, P["heavy_a"], smooth), (S - 3, P["heavy_up"], ease_out),
                              (S, P["heavy_w"], ease_in), (S + 1, P["heavy_s"], lerp_lin),
                              (S + A + 3, P["heavy_f"], smooth),        # 내려찍은 채 더 눌러줌
                              (S + A + R * 0.35, P["heavy_s"], smooth),
                              (S + A + R, base, smooth)])
        elif f.kind == C.DASH_ATK:   # 대쉬 자세에서 바로 이어진다 (기본 자세를 거치지 않음)
            keys = _strike_keys(name, base, ad.startup, ad.active, ad.recovery)
            keys[0] = (0, P["dash_f"], smooth)
            p = _timeline(x, keys)
        else:
            p = _timeline(x, _strike_keys(name, base, ad.startup, ad.active, ad.recovery))
        if not g and f.kind != Act.LOW:
            p = _with_legs(p, air_pose(f))
        return p
    if st == St.CROUCH:
        # 3프레임에 걸쳐 내려간다 (판정은 누른 순간부터 낮아짐), 숨쉬듯 아주 약하게 움직임
        p = lerp(base, P["crouch"], ease_out((f.sf + 1) / 3))
        b = math.sin(frame * 0.07)
        return _set(p, ph=p[1] + 0.5 * b, af_u=p[4] + 1.5 * b)
    if st == St.CHARGE:
        # 처음 8프레임에 크게 젖히고, 이후엔 모일수록 떨림이 커진다
        p = lerp(base, P["heavy_a"], ease_out(f.sf / 8))
        k = min(1.0, f.sf / C.HEAVY_MAX_CHARGE)
        jit = math.sin(f.sf * 2.1) * 2.5 * k
        return _set(p, torso=p[2] - 4 * k + jit, ph=p[1] - 3 * k)
    if st == St.RECOVER:
        t = 1 - f.stun / max(1, f.stun_max)
        return lerp(P["heavy_a"], base, smooth(t))
    if st == St.RANGED:
        r = C.RANGED
        S, R = r.startup, r.recovery
        p = _timeline(f.sf + 1, [(0, base, smooth), (max(2.0, round(S * 0.3)), P["rng_a1"], ease_out),
                                 (S - 1, P["rng_a"], smooth), (S, P["rng_w"], ease_in),
                                 (S + 1, P["rng_s"], lerp_lin), (S + 4, P["rng_s"], smooth),
                                 (S + R, base, smooth)])
        return p if g else _with_legs(p, air_pose(f))
    if st == St.PARRY:
        pr = C.PARRY
        if f.sf < pr.startup:
            p = lerp(base, P["parry"], (f.sf + 1) / pr.startup)
        elif f.sf < pr.startup + pr.active:
            p = P["parry"]
        else:
            # 헛친 패링: 팔이 풀리며 무방비
            t = (f.sf - pr.startup - pr.active) / pr.recovery
            slump = _set(P["parry"], torso=12, af_u=20, af_l=40, ab_u=10, ab_l=30)
            p = lerp(P["parry"], slump, ease_out(t * 2)) if t < 0.5 else \
                lerp(slump, base, smooth((t - 0.5) * 2))
        return p if g else _with_legs(p, air_pose(f))
    if st == St.DASH:
        key = "dash_f" if f.dash_dir == f.facing else "dash_b"
        if f.sf < C.DASH.duration:
            return lerp(base, P[key], min(1.0, (f.sf + 1) / 3))
        rec = C.DASH.rec(f.dash_dir == f.facing)
        return lerp(P[key], base, smooth((f.sf - C.DASH.duration + 1) / rec))
    if st == St.HITSTUN:
        t = f.stun / max(1, f.stun_max)
        p = lerp(base, P["hit"], smooth(t * 1.6))
        return p if g else _with_legs(p, air_pose(f))
    if st == St.PARRIED:
        t = f.stun / max(1, f.stun_max)
        wob = math.sin(f.sf * 0.45) * 5 * t
        p = lerp(base, P["parried"], smooth(t * 3))
        return _set(p, torso=p[2] + wob)
    if st == St.BREAK:
        if f.sf < 4:                                  # 맞던 자세에서 한 번에 펼친다
            p = lerp(P["hit"], P["burst"], ease_out((f.sf + 1) / 4))
        else:
            t = (f.sf - 4) / max(1, f.stun_max - 4)
            p = lerp(P["burst"], base, smooth(max(0.0, t - 0.3) / 0.7))
        return p if g else _with_legs(p, air_pose(f))
    if st == St.KO:
        return lerp(P["hit"], P["ko"], ease_out(f.sf / 22))
    return base


def victory_pose(t):
    """(v15) 승리 화면 모션. t = 화면이 뜬 뒤 지난 프레임 (렌더링 전용)."""
    if t < 14:
        return lerp(idle_pose(0), P["win_a"], ease_out(t / 14))
    if t < 24:
        return lerp(P["win_a"], P["win"], ease_out((t - 14) / 10))
    b = math.sin((t - 24) * 0.12)
    return _set(P["win"], af_u=172 + 3 * b, ph=53 + 0.8 * b, ab_u=-22 - 3 * b)


# ------------------------------------------------------------------ 관절 계산
def _limb(o, a1, a2, l1, l2):
    r1, r2 = math.radians(a1), math.radians(a2)
    e = (o[0] + l1 * math.sin(r1), o[1] - l1 * math.cos(r1))
    h = (e[0] + l2 * math.sin(r2), e[1] - l2 * math.cos(r2))
    return e, h


def _ik(hip, target, l1, l2):
    dx, dy = target[0] - hip[0], target[1] - hip[1]
    d = math.hypot(dx, dy)
    maxd = l1 + l2 - 0.01
    if d > maxd:
        target = (hip[0] + dx / d * maxd, hip[1] + dy / d * maxd)
        d = maxd
    d = max(d, 1e-3)
    a = math.acos(max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
    base = math.atan2(dy, dx)
    k1 = (hip[0] + l1 * math.cos(base + a), hip[1] + l1 * math.sin(base + a))
    k2 = (hip[0] + l1 * math.cos(base - a), hip[1] + l1 * math.sin(base - a))
    return (k1 if k1[0] >= k2[0] else k2), target   # 무릎은 앞으로 굽힌다


def solve(f, p):
    """포즈 → 월드 좌표 관절 dict."""
    px, ph, torso, head, afu, afl, abu, abl, ffx, ffy, bfx, bfy = p
    hip = (px, ph)
    tr = math.radians(torso)
    neck = (px + TORSO * math.sin(tr), ph + TORSO * math.cos(tr))
    hr = math.radians(torso + head)
    headc = (neck[0] + NECK_HEAD * math.sin(hr), neck[1] + NECK_HEAD * math.cos(hr))
    sh = (neck[0] - 3 * math.sin(tr), neck[1] - 3 * math.cos(tr))
    ef, hf = _limb(sh, afu, afl, UARM, LARM)
    eb, hb = _limb(sh, abu, abl, UARM, LARM)
    kf, ff = _ik(hip, (ffx, ffy), THIGH, SHIN)
    kb, fb = _ik(hip, (bfx, bfy), THIGH, SHIN)
    loc = dict(hip=hip, neck=neck, head=headc, shoulder=sh, elbow_f=ef, hand_f=hf,
               elbow_b=eb, hand_b=hb, knee_f=kf, foot_f=ff, knee_b=kb, foot_b=fb)
    s = f.facing
    return {k: (f.x + v[0] * s, f.y + v[1]) for k, v in loc.items()}
