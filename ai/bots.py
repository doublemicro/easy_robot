"""테스트/초기 학습 상대용 간단한 봇들.

모든 봇은 act(obs, world, me) -> Act 를 구현한다.
(룰 기반 봇은 편의상 world 객체를 읽지만, env/play에서는 SeenWorld를 넘긴다:
 상대와 상대 투사체는 반응 지연된 화면이고, 상대 패링 쿨타임은 들어 있지 않다. 학습된 AI는 obs만 쓴다.)
"""
import random

from game import config as C
from game.config import Act, St


class IdleBot:
    def act(self, obs, world, me):
        return Act.IDLE


class RandomBot:
    """무작위 행동을 몇 스텝씩 유지한다."""

    def __init__(self, seed=None, hold=(1, 5)):
        self.rng = random.Random(seed)
        self.hold = hold
        self.left = 0
        self.a = Act.IDLE

    def act(self, obs, world, me):
        if self.left <= 0:
            self.a = Act(self.rng.randrange(C.N_ACTIONS))
            self.left = self.rng.randint(*self.hold)
        self.left -= 1
        return self.a


class AggroBot:
    """다가가서 때리는 단순한 공격형 봇. 가끔 패링, 원거리, 점프."""

    # 기를 모으기 시작할 때 언제 뗄지 정한다: 페인트 / 1단계 / 2단계 / 3단계 / 끝까지
    RELEASE_AT = (8, 16, 30, 46, C.HEAVY_MAX_CHARGE)
    RELEASE_W = (1, 3, 3, 2, 1)

    def __init__(self, seed=None, parry_rate=0.25, break_rate=0.6, dash_attack=True):
        self.rng = random.Random(seed)
        self.parry_rate = parry_rate
        self.break_rate = break_rate   # 콤보를 맞는 중 게이지가 차 있으면 브레이커를 시도할 확률
        self.dash_attack = dash_attack  # 도망가는 상대에게 앞 대쉬 → 대쉬 공격을 쓰는지
        self.release_at = None

    def act(self, obs, world, me):
        r = self.rng
        f, o = world.f[me], world.f[1 - me]
        if f.state == St.CHARGE:
            if self.release_at is None:
                self.release_at = r.choices(self.RELEASE_AT, self.RELEASE_W)[0]
            return Act.HEAVY if f.sf < self.release_at else Act.IDLE
        self.release_at = None
        if (f.state == St.HITSTUN and f.can_break and not f.break_locked
                and o.state == St.ATTACK and o.kind in C.CANCELABLE
                and r.random() < self.break_rate):
            # 콤보 브레이커: 지연된 화면에서 보이는 상대 공격과 같은 버튼을 누른다.
            # 상대가 그 사이 다른 공격으로 캔슬했으면 틀려서 잠긴다.
            return o.kind
        dx = o.x - f.x
        dist = abs(dx)
        if (self.dash_attack and f.state == St.DASH and f.dash_dir == f.facing
                and f.grounded and dist < 150):
            return Act.MID                       # 대쉬 중 정면 → 대쉬 공격
        if f.state != St.IDLE:
            # 공격이 맞았으면 콤보를 잇는다
            if f.can_cancel():
                return r.choice([k for k in C.CANCELABLE if k != f.kind])
            return Act.IDLE
        toward = Act.RIGHT if dx > 0 else Act.LEFT

        for p in world.proj:
            if p.owner != me and (f.x - p.x) * p.vx > 0 and abs(f.x - p.x) < 110:
                return r.choice([Act.JUMP, Act.PARRY, Act.LOW])   # v6: 대쉬 무적이 없어 대쉬로는 못 뚫음
        if (o.state == St.ATTACK and o.sf < C.ATTACKS[o.kind].startup
                and dist < 110 and f.parry_cd == 0 and r.random() < self.parry_rate):
            return Act.PARRY
        if o.state == St.CHARGE and dist < 100:     # 기 모으는 상대는 끊거나 피한다
            return r.choice([Act.MID, Act.MID, Act.DASH_B, Act.JUMP])
        if dist < 75:
            return r.choice([Act.MID, Act.MID, Act.LOW, Act.HIGH, Act.HEAVY, Act.DASH_B])
        away = o.vx * (1 if dx > 0 else -1) > 0.5          # 상대가 멀어지고 있음
        if (self.dash_attack and o.state == St.DASH and away and dist < 260
                and f.dash_cd == 0 and f.grounded):
            return Act.DASH_F                    # 백대쉬를 보면 바로 쫓아가 후딜을 노린다
        if (90 < dist < 220 and away and f.dash_cd == 0 and f.grounded
                and r.random() < 0.5):
            return Act.DASH_F                    # 도망가는 상대를 대쉬로 쫓는다
        if dist > 300 and f.ranged_cd == 0 and r.random() < 0.3:
            return Act.RANGED
        if r.random() < 0.04:
            return Act.JUMP
        return toward


class RunnerBot:
    """한 대 때리고 도망가는 봇 (대쉬 공격 검증용).

    체력이 앞서면 계속 뒤로 걷고, 상대가 대쉬로 쫓아오면 백대쉬로 피한다. 벽에 몰리거나
    상대가 붙으면 정면으로 견제한다. 체력이 같거나 뒤지면 다가가서 한 대 친다.
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, obs, world, me):
        r = self.rng
        f, o = world.f[me], world.f[1 - me]
        if f.state != St.IDLE:
            return Act.IDLE
        dx = o.x - f.x
        dist = abs(dx)
        toward = Act.RIGHT if dx > 0 else Act.LEFT
        away = Act.LEFT if dx > 0 else Act.RIGHT
        lo = C.WALL_MARGIN + C.BODY_W / 2
        cornered = (f.x - lo < 40 and dx > 0) or (C.ARENA_W - lo - f.x < 40 and dx < 0)
        for p in world.proj:
            if p.owner != me and (f.x - p.x) * p.vx > 0 and abs(f.x - p.x) < 110:
                return r.choice([Act.JUMP, Act.LOW])
        leading = f.hp > o.hp
        if not leading:
            return Act.MID if dist < 70 else toward
        # 앞서고 있다 → 도망
        if o.state == St.DASH and dist < 200 and f.dash_cd == 0 and not cornered:
            return Act.DASH_B
        if dist < 70:
            if (o.state == St.ATTACK and o.sf < C.ATTACKS[o.kind].startup
                    and f.parry_cd == 0 and r.random() < 0.3):
                return Act.PARRY
            if not cornered and f.dash_cd == 0 and r.random() < 0.5:
                return Act.DASH_B
            return Act.MID
        if cornered:
            return Act.IDLE if dist > 120 else Act.MID
        if dist > 300 and f.ranged_cd == 0 and r.random() < 0.2:
            return Act.RANGED
        return away


class ZonerBot:
    """원거리로 견제하며 거리를 유지하는 봇 (원거리 쿨타임 검증용).

    쿨타임이 돌면 쏘고, 상대가 다가오면 뒤로 걷거나 백대쉬, 붙으면 정면/하단으로 견제.
    """

    def __init__(self, seed=None, keep=(220, 380)):
        self.rng = random.Random(seed)
        self.keep = keep

    def act(self, obs, world, me):
        r = self.rng
        f, o = world.f[me], world.f[1 - me]
        if f.state != St.IDLE:
            return Act.IDLE
        dx = o.x - f.x
        dist = abs(dx)
        toward = Act.RIGHT if dx > 0 else Act.LEFT
        away = Act.LEFT if dx > 0 else Act.RIGHT
        lo = C.WALL_MARGIN + C.BODY_W / 2
        cornered = (f.x - lo < 40 and dx > 0) or (C.ARENA_W - lo - f.x < 40 and dx < 0)
        for p in world.proj:
            if p.owner != me and (f.x - p.x) * p.vx > 0 and abs(f.x - p.x) < 110:
                return r.choice([Act.JUMP, Act.LOW])
        if dist < 75:
            if not cornered and f.dash_cd == 0 and r.random() < 0.4:
                return Act.DASH_B
            return r.choice([Act.MID, Act.LOW])
        if f.ranged_cd == 0 and dist > 120:
            return Act.RANGED
        if dist < self.keep[0] and not cornered:
            return away
        if dist > self.keep[1]:
            return toward
        return Act.IDLE


# ====================================================================== 난이도별 봇
# 학습시킬 AI의 상대(커리큘럼)와 사람용 봇 대전에 쓴다.
# 세 봇 모두 반응 지연된 화면(SeenWorld)만 보고, 상대 버튼·패링 쿨타임·미래 시뮬레이션은 쓰지 않는다.
# 반응 시간(reaction_ms)은 사람 기준(130~240ms)보다 빠를 수 없다. 하위 봇만 더 느리게 둔다.

HEAVY_STEP_PX = 25   # (v8) 강공격은 내려치며 약 25px 내딛는다


def _reach(kind):
    """공격 판정이 닿는 중심 간 거리 (히트박스 앞 끝 + 상대 몸 절반, 강공격은 내딛는 거리 포함)."""
    r = C.ATTACKS[kind].box[1] + C.BODY_W / 2
    return r + HEAVY_STEP_PX if kind == Act.HEAVY else r


def _cornered(f, dx, margin=40):
    lo = C.WALL_MARGIN + C.BODY_W / 2
    return (f.x - lo < margin and dx > 0) or (C.ARENA_W - lo - f.x < margin and dx < 0)


class EasyBot:
    """하위 봇. 반응이 느리고(180~360ms), 자주 머뭇거리고, 공격을 고르는 게 거의 무작위.
    콤보는 가끔 2타까지만, 브레이커·대쉬 공격은 안 쓰고, 패링도 드물다."""

    reaction_ms = (180, 260, 360)

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.release_at = None

    def act(self, obs, world, me):
        r = self.rng
        f, o = world.f[me], world.f[1 - me]
        if f.state == St.CHARGE:
            if self.release_at is None:
                self.release_at = r.choice((8, 16, 16, 30))
            return Act.HEAVY if f.sf < self.release_at else Act.IDLE
        self.release_at = None
        if f.state != St.IDLE:
            if f.can_cancel() and f.chain < 2 and r.random() < 0.3:
                return r.choice([k for k in C.CANCELABLE if k != f.kind])
            return Act.IDLE
        dx = o.x - f.x
        dist = abs(dx)
        toward = Act.RIGHT if dx > 0 else Act.LEFT
        for p in world.proj:
            if (p.owner != me and (f.x - p.x) * p.vx > 0 and abs(f.x - p.x) < 100
                    and r.random() < 0.35):
                return r.choice([Act.JUMP, Act.LOW])
        if r.random() < 0.25:                       # 머뭇거림
            return Act.IDLE
        if dist < 85:
            if (o.state == St.ATTACK and f.parry_cd == 0 and r.random() < 0.08):
                return Act.PARRY
            if r.random() < 0.45:
                return r.choice([Act.MID, Act.MID, Act.LOW, Act.HIGH, Act.HIGH, Act.HEAVY])
            return r.choice([Act.IDLE, toward, Act.LEFT if dx > 0 else Act.RIGHT])
        if dist > 300 and f.ranged_cd == 0 and r.random() < 0.1:
            return Act.RANGED
        if r.random() < 0.03:
            return Act.JUMP
        return toward


class NormalBot(AggroBot):
    """중간 봇 = 지금까지 쓰던 공격형 봇 (다가가서 때리고, 가끔 패링·원거리·브레이커·대쉬 공격)."""

    reaction_ms = (130, 180, 240)


class HardBot:
    """상위 봇. 반응 속도는 중간 봇과 같고(130~240ms), 판단으로 이긴다.

    - 프레임 데이터로 상대 공격의 남은 선딜·후딜을 어림한다. 화면이 평균 LAG 프레임 늦다는 것을 알고 보정한다.
    - 헛친 공격·헛 패링·백대쉬 후딜을 정면/하단/대쉬 공격으로 벌한다.
    - 상대 공격 종류에 맞춰 피한다: 상단은 하단으로 숙여 받아치고, 하단은 점프, 정면은 패링·백대쉬.
    - 거리 싸움: 상대 정면 사거리 바로 밖에 서 있다가, 들어오면 하단(사거리 김)으로 견제.
    - 콤보는 끊기지 않는 루트로, 상대 브레이커가 차 있으면 일찍 끊거나 바로바로 캔슬해 브레이커를 속인다.
    - 상대 습관(패링·점프·백대쉬·원거리 빈도)을 세어 공격을 고른다. 패링을 자주 하면 강공격·기다렸다 벌하기.
    - 체력이 앞서고 시간이 얼마 안 남으면 거리를 벌리고 원거리로 버틴다.
    """

    reaction_ms = (130, 180, 240)
    LAG = 11                      # 화면이 늦은 정도의 평균 (프레임)
    ROUTES = {Act.MID: (Act.LOW, Act.MID, Act.HIGH),
              Act.LOW: (Act.MID, Act.LOW, Act.HIGH),
              C.DASH_ATK: (Act.LOW, Act.MID, Act.HIGH),
              Act.HIGH: (Act.MID, Act.LOW, Act.MID)}

    def __init__(self, seed=None, parry_skill=0.7):
        self.rng = random.Random(seed)
        self.parry_skill = parry_skill
        self.release_at = None
        self.prev = None          # 지난 판단 때 본 상대 (라벨, sf, y)
        self.seen_n = 1.0         # 습관 기록 (지수 이동 평균의 분모 대신 쓰는 단순 카운트)
        self.habit = dict(parry=0.0, jump=0.0, backdash=0.0, ranged=0.0, attack=0.0)
        self.stop_combo = False
        self._route_key = None

    # ---------------------------------------------------------------- 습관 기록
    def _watch(self, f, o):
        cur = (o.label, o.sf, o.y)
        if self.prev is not None:
            pl, psf, py = self.prev
            new = o.label != pl or o.sf < psf
            if new:
                d = 0.9                           # 오래된 기록은 천천히 잊는다
                for k in self.habit:
                    self.habit[k] *= d
                self.seen_n = self.seen_n * d + 1
                if o.state == St.PARRY:
                    self.habit["parry"] += 1
                elif o.state == St.DASH and o.vx * (o.x - f.x) > 0:
                    self.habit["backdash"] += 1
                elif o.state == St.RANGED:
                    self.habit["ranged"] += 1
                elif o.state == St.ATTACK:
                    self.habit["attack"] += 1
            if py <= 0 < o.y:
                self.habit["jump"] += 1
        self.prev = cur

    def rate(self, k):
        return self.habit[k] / self.seen_n

    # ---------------------------------------------------------------- 판단
    def act(self, obs, world, me):
        r = self.rng
        f, o = world.f[me], world.f[1 - me]
        self._watch(f, o)
        dx = o.x - f.x
        dist = abs(dx)
        toward = Act.RIGHT if dx > 0 else Act.LEFT
        away = Act.LEFT if dx > 0 else Act.RIGHT
        corner = _cornered(f, dx)

        # 강공격 차지: 정해 둔 단계에서 뗀다. 상대가 먼저 공격을 내밀면 바로 뗀다(1단계 이상일 때).
        if f.state == St.CHARGE:
            if self.release_at is None:
                self.release_at = 16 if dist < _reach(Act.HEAVY) else r.choice((16, 30))
            if f.sf >= 14 and o.state == St.ATTACK:
                return Act.IDLE
            return Act.HEAVY if f.sf < self.release_at else Act.IDLE
        self.release_at = None

        # 맞는 중: 브레이커. 게이지 MAX로 앞서 있으면 가끔 아껴서 데미지 보너스를 유지
        if f.state == St.HITSTUN:
            if (f.can_break and not f.break_locked and o.state == St.ATTACK
                    and o.kind in C.CANCELABLE):
                keep = f.gauge_full and f.hp > o.hp + 15 and r.random() < 0.5
                if not keep:
                    return o.kind
            return Act.IDLE

        # 대쉬 중: 사거리 안이면 대쉬 공격
        if f.state == St.DASH:
            if f.dash_dir == f.facing and f.grounded and dist < 150 and o.state != St.PARRY:
                return Act.MID
            return Act.IDLE

        # 공격 중: 콤보
        if f.state == St.ATTACK:
            if f.can_cancel():
                if f.chain == 1:
                    # 상대 브레이커가 살아 있으면 1타에서 끝내거나, 끝까지 바로바로 캔슬해서 속인다
                    self._route_key = f.kind
                    self.stop_combo = (o.can_break and not o.break_locked and r.random() < 0.4)
                if self.stop_combo:
                    return Act.IDLE
                route = self.ROUTES.get(self._route_key, ())
                k = f.chain - 1
                if k < len(route) and route[k] != f.kind:
                    return route[k]
            return Act.IDLE
        if f.state != St.IDLE:
            return Act.IDLE

        # ---- 자유 상태 ----
        # 1) 날아오는 투사체: 가까우면 하단(숙이면서 공격), 멀면 점프
        for p in world.proj:
            if p.owner != me and (f.x - p.x) * p.vx > 0:
                pd = abs(f.x - p.x)
                if pd < 190:
                    if dist < _reach(Act.LOW):
                        return Act.LOW
                    return Act.JUMP if pd > 120 and f.grounded else Act.LOW

        est = o.sf + self.LAG                    # 상대 행동이 지금 몇 프레임째일지 어림

        # 2) 상대 공격 대응
        if o.state == St.ATTACK and o.kind in C.ATTACKS:
            ad = C.ATTACKS[o.kind]
            left_start = ad.startup - est              # 판정까지 남은 프레임
            left_total = ad.total - est                # 끝날 때까지 남은 프레임
            threat = dist < _reach(o.kind) + 12 or o.kind == C.DASH_ATK
            if left_start > 0 and threat:
                if o.kind == Act.HIGH:
                    return Act.LOW if dist < _reach(Act.LOW) else away   # 숙여서 피하며 받아침
                if o.kind == Act.LOW:
                    return Act.JUMP
                if (ad.parryable and f.parry_cd == 0 and 1 <= left_start <= 9
                        and r.random() < self.parry_skill):
                    return Act.PARRY
                if o.kind == Act.HEAVY:
                    if dist < _reach(Act.MID) and left_start >= C.ATTACKS[Act.MID].startup:
                        return Act.MID
                    return Act.DASH_B if not corner and f.dash_cd == 0 else Act.JUMP
                if not corner and f.dash_cd == 0:
                    return Act.DASH_B
                return Act.PARRY if f.parry_cd == 0 else Act.LOW
            # (v10) 대쉬 공격은 늘 위협으로 봐서 헛쳐도 벌하지 않았음 → 판정이 끝났으면 벌한다
            whiffed = not threat or (o.kind == C.DASH_ATK and est >= ad.startup + ad.active)
            if left_start <= 0 and left_total > 0 and whiffed:
                # 헛친 공격 → 후딜을 벌한다
                p = self._punish(f, dist, left_total)
                if p is not None:
                    return p

        # 3) 헛 패링, 원거리 후딜, 페인트, 패링당함 → 벌하기
        if o.state == St.PARRY and est > C.PARRY.startup + C.PARRY.active:
            p = self._punish(f, dist, C.PARRY.total - est)
            if p is not None:
                return p
        if o.state in (St.RANGED, St.PARRIED, St.RECOVER):
            left = (C.RANGED.total if o.state == St.RANGED else
                    C.PARRY.stun if o.state == St.PARRIED else C.HEAVY_FEINT_LAG) - est
            p = self._punish(f, dist, left)
            if p is not None:
                return p
        # 상대가 백대쉬 → 대쉬로 쫓아 후딜을 노린다
        if (o.state == St.DASH and o.vx * dx > 0 and dist < 260 and f.dash_cd == 0
                and f.grounded):
            return Act.DASH_F

        # 4) 기 모으는 상대: 붙어 있으면 끊고, 조금 떨어져 있으면 대쉬 공격
        if o.state == St.CHARGE:
            if dist < _reach(Act.MID):
                return Act.MID
            if dist < 170 and f.dash_cd == 0 and f.grounded:
                return Act.DASH_F
            if dist < _reach(Act.HEAVY) + 10:      # (v8) 강공격 사거리 안: 빠지거나 숙여 받는다
                return Act.DASH_B if not corner and f.dash_cd == 0 else Act.LOW
            return toward

        # 5) 벽꽝 등으로 상대가 아직 경직 중이면 이어서 때린다
        if o.state == St.HITSTUN and o.stun > self.LAG + 7 and dist < _reach(Act.MID):
            return Act.MID

        time_left = C.ROUND_FRAMES - world.frame
        protect = f.hp > o.hp + 10 and time_left < 20 * C.FPS

        # 6) 거리 싸움
        if dist < _reach(Act.MID) - 4:
            if o.state == St.HITSTUN or o.state in (St.DASH,):
                return Act.MID
            pr, jr = self.rate("parry"), self.rate("jump")
            roll = r.random()
            if pr > 0.25 and roll < 0.35:
                # 패링을 자주 한다 → 기다렸다 헛 패링을 벌하거나 강공격(패링 불가)
                return Act.HEAVY if roll < 0.2 else Act.IDLE
            if jr > 0.2 and roll < 0.5:
                return Act.HIGH
            if roll < 0.55:
                return Act.MID
            if roll < 0.70:
                return Act.LOW
            if roll < 0.80 and f.parry_cd == 0:
                return Act.PARRY
            if not corner and f.dash_cd == 0:
                return Act.DASH_B
            return Act.MID
        approaching = o.vx * dx < -0.5
        if dist < _reach(Act.LOW) + 6:
            if approaching or r.random() < 0.35:
                return Act.LOW
            return Act.IDLE if r.random() < 0.5 else toward
        if protect:
            if dist > 300 and f.ranged_cd == 0:
                return Act.RANGED
            return away if not corner and dist < 260 else Act.IDLE
        running = o.vx * dx > 0.5
        if running and 90 < dist < 220 and f.dash_cd == 0 and f.grounded:
            return Act.DASH_F
        if dist > 300 and f.ranged_cd == 0 and (self.rate("ranged") > 0.15 or r.random() < 0.4):
            return Act.RANGED
        if dist < 130 and not approaching and r.random() < 0.3:
            return Act.IDLE                       # 사거리 밖에서 기다리며 상대가 먼저 들어오게
        return toward

    def _punish(self, f, dist, left):
        """상대가 left 프레임 동안 못 움직일 때 닿는 가장 좋은 공격."""
        mid, low = C.ATTACKS[Act.MID], C.ATTACKS[Act.LOW]
        if dist < _reach(Act.MID) and left >= mid.startup:
            return Act.MID
        if dist < _reach(Act.LOW) and left >= low.startup:
            return Act.LOW
        if dist < 200 and left >= 14 and f.dash_cd == 0 and f.grounded:
            return Act.DASH_F
        if dist < 160 and left >= 10:
            return Act.RIGHT if f.facing > 0 else Act.LEFT
        return None


class SpacerBot:
    """(v10) 거리 조절 봇. 상대 사거리 바로 밖(keep)에서 기다리다가, 상대가 헛치면 대쉬 공격·정면·하단으로 벌한다.
    상대가 대쉬로 들어오거나 공격을 내밀면 뒤로 빠져서 헛치게 만든다. 구석에 몰리면 붙어서 싸운다.
    사람이 1단계 AI의 대쉬 공격 연타를 이긴 방법을 그대로 옮긴 것 (학습 상대 + 시험용)."""

    reaction_ms = (130, 180, 240)
    LAG = 11
    ROUTE = {Act.MID: Act.LOW, C.DASH_ATK: Act.LOW, Act.LOW: Act.MID}

    def __init__(self, seed=None, keep=(150, 230), parry_skill=0.5):
        self.rng = random.Random(seed)
        self.keep = keep
        self.parry_skill = parry_skill

    def _punish(self, f, dist, left, toward):
        if dist < _reach(Act.MID) and left >= C.ATTACKS[Act.MID].startup:
            return Act.MID
        if dist < _reach(Act.LOW) and left >= C.ATTACKS[Act.LOW].startup:
            return Act.LOW
        if dist < 230 and left >= 12 and f.dash_cd == 0 and f.grounded:
            return Act.DASH_F                     # 대쉬 → 아래 DASH 상태에서 대쉬 공격
        if dist < 300 and left >= 8:
            return toward
        return None

    def act(self, obs, world, me):
        r = self.rng
        f, o = world.f[me], world.f[1 - me]
        dx = o.x - f.x
        dist = abs(dx)
        toward = Act.RIGHT if dx > 0 else Act.LEFT
        away = Act.LEFT if dx > 0 else Act.RIGHT
        corner = _cornered(f, dx)

        if f.state == St.CHARGE:
            return Act.IDLE
        if f.state == St.HITSTUN:
            if f.can_break and not f.break_locked and o.state == St.ATTACK and o.kind in C.CANCELABLE:
                return o.kind
            return Act.IDLE
        if f.state == St.DASH:
            if f.dash_dir == f.facing and f.grounded and dist < 190 and o.state != St.PARRY:
                return Act.MID
            return Act.IDLE
        if f.state == St.ATTACK:
            if f.can_cancel() and f.chain < 3:
                return self.ROUTE.get(f.kind, Act.IDLE)
            return Act.IDLE
        if f.state != St.IDLE:
            return Act.IDLE

        est = o.sf + self.LAG
        # 상대 공격: 아직 나오는 중이면 빠지고, 헛쳤으면 벌한다
        if o.state == St.ATTACK and o.kind in C.ATTACKS:
            ad = C.ATTACKS[o.kind]
            left_start, left_total = ad.startup - est, ad.total - est
            if left_start > 0:
                if dist < _reach(o.kind) + 40 or o.kind == C.DASH_ATK:
                    if (ad.parryable and f.parry_cd == 0 and 1 <= left_start <= 9
                            and dist < _reach(o.kind) + 10 and r.random() < self.parry_skill):
                        return Act.PARRY
                    if o.kind == Act.LOW and dist < _reach(Act.LOW) + 10:
                        return Act.JUMP
                    if not corner:
                        return Act.DASH_B if f.dash_cd == 0 and dist < 140 else away
                    return Act.PARRY if f.parry_cd == 0 else Act.LOW
            elif est >= ad.startup + ad.active:
                p = self._punish(f, dist, left_total, toward)
                if p is not None:
                    return p
        if o.state in (St.RANGED, St.PARRIED, St.RECOVER) or (
                o.state == St.PARRY and est > C.PARRY.startup + C.PARRY.active):
            left = {St.RANGED: C.RANGED.total, St.PARRIED: C.PARRY.stun,
                    St.RECOVER: C.HEAVY_FEINT_LAG, St.PARRY: C.PARRY.total}[o.state] - est
            p = self._punish(f, dist, left, toward)
            if p is not None:
                return p
        if o.state == St.CHARGE:
            if dist < _reach(Act.MID):
                return Act.MID
            return away if not corner else Act.LOW
        # 상대가 대쉬로 들어온다 → 뒤로 빠져 대쉬 공격을 헛치게
        if o.state == St.DASH and o.vx * dx < 0:
            if not corner:
                return away
            return Act.PARRY if f.parry_cd == 0 and dist < 160 else Act.LOW
        if o.state == St.HITSTUN and o.stun > self.LAG + 7 and dist < _reach(Act.MID):
            return Act.MID
        for p in world.proj:
            if p.owner != me and (f.x - p.x) * p.vx > 0 and abs(f.x - p.x) < 170:
                return Act.JUMP if f.grounded and abs(f.x - p.x) > 110 else Act.LOW

        # 거리 조절
        lo, hi = self.keep
        if dist < lo:
            if corner or dist < _reach(Act.MID) - 10:
                roll = r.random()
                if roll < 0.35:
                    return Act.MID
                if roll < 0.55:
                    return Act.LOW
                if roll < 0.65 and f.parry_cd == 0:
                    return Act.PARRY
                if corner and f.dash_cd == 0 and roll < 0.75:
                    return Act.JUMP
            return away if not corner else Act.IDLE
        if dist > hi:
            if dist > 380 and f.ranged_cd == 0 and r.random() < 0.15:
                return Act.RANGED
            return toward
        return r.choice((Act.IDLE, Act.IDLE, Act.IDLE, toward, away))


class CrouchLowBot:
    """(v13) 숙이기 + 하단만 하는 봇. 멀면 걸어오고, 가까우면 숙여서 상단·정면·투사체를 피하며 기다리다
    하단 사거리에 들어오면 하단. 사람이 "숙이고 하단만 치면 할 게 없지 않나?" 하고 걱정한 전략 (학습 상대 + 시험용).
    대응: 거리를 두고 헛친 하단(후딜 24)을 벌하기, 점프로 넘기, 대쉬 공격·강공격·하단으로 숙인 몸 때리기."""

    reaction_ms = (130, 180, 240)

    def __init__(self, seed=None, wait=130):
        self.rng = random.Random(seed)
        self.wait = wait            # 이 거리 안에서는 숙여서 기다린다

    def act(self, obs, world, me):
        f, o = world.f[me], world.f[1 - me]
        if f.state not in (St.IDLE, St.CROUCH):
            return Act.IDLE
        dx = o.x - f.x
        if abs(dx) < _reach(Act.LOW) - 2:
            return Act.LOW
        if abs(dx) < self.wait:
            return Act.CROUCH
        return Act.RIGHT if dx > 0 else Act.LEFT


BOTS = {"idle": IdleBot, "random": RandomBot, "aggro": AggroBot, "runner": RunnerBot,
        "zoner": ZonerBot, "easy": EasyBot, "normal": NormalBot, "hard": HardBot,
        "spacer": SpacerBot, "crouchlow": CrouchLowBot}
TIERS = ("easy", "normal", "hard")   # 봇 대전 난이도 (하위 / 중간 / 상위)
