"""헤드리스 게임 시뮬레이션. 화면과 무관하게 돌아가며 결정론적이다
(같은 입력이면 항상 같은 결과).

사용법:
    w = World()
    events = w.step(action_p1, action_p2)   # 1프레임 진행
"""
from __future__ import annotations

from . import config as C
from .config import Act, St

_LABEL_OF_KIND = {Act.HIGH: 1, Act.MID: 2, Act.LOW: 3, Act.HEAVY: 10, C.DASH_ATK: 14}
_LABEL_OF_STATE = {St.IDLE: 0, St.RANGED: 4, St.PARRY: 5, St.DASH: 6,
                   St.HITSTUN: 7, St.PARRIED: 8, St.KO: 9, St.CHARGE: 11, St.RECOVER: 12,
                   St.BREAK: 13, St.CROUCH: 15}


class Fighter:
    __slots__ = ("idx", "x", "y", "vx", "vy", "facing", "hp", "state", "sf",
                 "kind", "stun", "stun_max", "hit_done", "hit_frame", "chain",
                 "ranged_cd", "dash_cd", "parry_cd", "air_dash_used", "dash_dir", "buf",
                 "buf_t", "combo_taken", "heavy_lv", "break_gauge", "break_locked", "break_idle",
                 "wall_hit")

    def __init__(self, idx: int, x: float, facing: int):
        self.idx = idx
        self.x = float(x)
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.facing = facing
        self.hp = C.MAX_HP
        self.state = St.IDLE
        self.sf = 0              # 현재 상태에서 지난 프레임 수
        self.kind = None         # ATTACK일 때 공격 종류
        self.stun = 0
        self.stun_max = 1
        self.hit_done = False    # 현재 공격이 이미 맞았는지
        self.hit_frame = 0
        self.chain = 0           # 현재 콤보에서 몇 번째 공격인지
        self.ranged_cd = 0
        self.dash_cd = 0
        self.parry_cd = 0
        self.air_dash_used = False
        self.dash_dir = 1
        self.buf = None          # 선입력된 행동
        self.buf_t = 0
        self.combo_taken = 0     # 지금 연속으로 맞은 횟수 (데미지 보정용)
        self.heavy_lv = 0        # 지금 휘두르는 강공격의 차지 단계 (1~3)
        self.break_gauge = 0.0   # 콤보 브레이커 게이지 (맞으면 참)
        self.break_locked = False  # 브레이커를 잘못 눌러서 이번 콤보 동안 잠김
        self.break_idle = 0      # 마지막으로 맞은 뒤 지난 프레임 (게이지 감소 타이머)
        self.wall_hit = False    # 이번 콤보에서 이미 벽꽝을 당했는지 (한 콤보에 한 번)

    # ------------------------------------------------------------ 조회
    @property
    def grounded(self) -> bool:
        return self.y <= 0.0

    @property
    def label(self) -> int:
        if self.state == St.ATTACK:
            return _LABEL_OF_KIND[self.kind]
        return _LABEL_OF_STATE[self.state]

    @property
    def gauge_full(self) -> bool:
        return self.break_gauge >= C.BREAKER.gauge_max

    @property
    def can_break(self) -> bool:
        return self.break_gauge >= C.BREAKER.use_min

    @property
    def charge_stage(self) -> int:
        """화면에 보이는 차지 단계 (기 모으는 중: 0~3, 강공격 휘두르는 중: 1~3, 그 외 0)."""
        if self.state == St.CHARGE:
            return C.heavy_stage(self.sf)
        if self.state == St.ATTACK and self.kind == Act.HEAVY:
            return self.heavy_lv
        return 0

    def attack_active(self) -> bool:
        if self.state != St.ATTACK:
            return False
        ad = C.ATTACKS[self.kind]
        return ad.startup <= self.sf < ad.startup + ad.active

    def parry_active(self) -> bool:
        return (self.state == St.PARRY
                and C.PARRY.startup <= self.sf < C.PARRY.startup + C.PARRY.active)

    def invulnerable(self) -> bool:
        if self.state == St.BREAK:
            return True
        return (self.state == St.DASH
                and (self.dash_dir == self.facing or C.DASH.back_invuln)
                and C.DASH.invuln_start <= self.sf < C.DASH.invuln_end)

    def can_cancel(self) -> bool:
        return (self.state == St.ATTACK and (self.kind in C.CANCELABLE or self.kind == C.DASH_ATK)
                and self.hit_done
                and self.chain < C.COMBO.max_chain
                and self.sf - self.hit_frame <= C.COMBO.cancel_window)

    def hurtbox(self):
        if self.state == St.KO:
            h = 30.0
        elif self.state == St.CROUCH:
            h = C.BODY_H_CROUCH
        elif self.state == St.ATTACK and self.kind == Act.LOW:
            h = C.BODY_H_LOW
        else:
            h = C.BODY_H
        hw = C.BODY_W / 2
        return (self.x - hw, self.x + hw, self.y, self.y + h)

    def hitbox(self):
        x0, x1, y0, y1 = C.ATTACKS[self.kind].box
        if self.facing > 0:
            return (self.x + x0, self.x + x1, self.y + y0, self.y + y1)
        return (self.x - x1, self.x - x0, self.y + y0, self.y + y1)

    # ------------------------------------------------------------ 입력 버퍼
    def push_buffer(self, act):
        if act in C.BUFFERABLE:
            self.buf = act
            self.buf_t = C.INPUT_BUFFER

    def clear_buffer(self):
        self.buf = None
        self.buf_t = 0

    def snapshot(self):
        return (self.x, self.y, self.vx, self.vy, self.hp, self.facing,
                self.label, self.sf, self.stun, self.ranged_cd, self.dash_cd,
                self.parry_cd, self.charge_stage, self.break_gauge, self.break_locked,
                self.wall_hit)


class Projectile:
    __slots__ = ("owner", "x", "y", "vx", "alive")

    def __init__(self, owner, x, y, vx):
        self.owner = owner
        self.x = x
        self.y = y
        self.vx = vx
        self.alive = True


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _center(a, b):
    return ((max(a[0], b[0]) + min(a[1], b[1])) / 2,
            (max(a[2], b[2]) + min(a[3], b[3])) / 2)


class World:
    def __init__(self, x0: float = C.START_X[0], x1: float = C.START_X[1],
                 training: bool = False):
        # training=True: 훈련장 모드. 시간 제한과 KO가 없고, 체력이 0이 되면 바로 가득 찬다.
        # 넉백·경직·콤보 보정 등 나머지 규칙은 실전과 똑같다.
        self.training = training
        self.reset(x0, x1)

    def reset(self, x0: float = C.START_X[0], x1: float = C.START_X[1]):
        self.f = [Fighter(0, x0, 1 if x1 >= x0 else -1),
                  Fighter(1, x1, -1 if x1 >= x0 else 1)]
        self.proj: list[Projectile] = []
        self.frame = 0
        self.hitstop = 0
        self.done = False
        self.winner = None      # 0, 1, 또는 -1(무승부)
        self.events = []

    def snapshot(self):
        return (tuple(f.snapshot() for f in self.f),
                tuple((p.owner, p.x, p.y, p.vx) for p in self.proj if p.alive))

    # ------------------------------------------------------------ 메인 루프
    def step(self, a0, a1):
        """1프레임 진행. 이번 프레임에 일어난 이벤트 목록을 반환한다."""
        self.events = []
        acts = (Act(a0), Act(a1))
        if self.done:
            acts = (Act.IDLE, Act.IDLE)   # 끝난 뒤에도 쓰러지는 연출용으로 물리만 진행
        if self.hitstop > 0:
            self.hitstop -= 1
            for f, a in zip(self.f, acts):
                if f.state != St.KO:
                    f.push_buffer(a)
            return self.events

        self.frame += 1
        for f, a in zip(self.f, acts):
            self._update_fighter(f, a)
        for f in self.f:
            self._physics(f)
        self._update_facing()
        self._push_apart()
        self._move_projectiles()
        if not self.done:
            self._resolve_hits()
            if not self.training:
                self._check_end()
        return self.events

    # ------------------------------------------------------------ 상태 기계
    def _update_fighter(self, f: Fighter, act: Act):
        if f.state != St.KO:
            f.push_buffer(act)
        if f.ranged_cd > 0:
            f.ranged_cd -= 1
        if f.dash_cd > 0:
            f.dash_cd -= 1
        if f.parry_cd > 0:
            f.parry_cd -= 1
        bk = C.BREAKER
        f.break_idle += 1
        if f.break_idle > bk.decay_delay and f.break_gauge > 0:
            f.break_gauge = max(0.0, f.break_gauge - bk.decay_per_frame)

        st = f.state
        if st == St.KO:
            f.sf += 1
        elif st == St.CHARGE:
            # f.sf = 지금까지 누르고 있던 프레임 수
            if act != Act.HEAVY:                 # 손을 뗐다
                lv = C.heavy_stage(f.sf)
                if lv == 0:                      # 최소 차지 전에 뗐다 → 페인트
                    f.state = St.RECOVER
                    f.sf = 0
                    f.stun = f.stun_max = C.HEAVY_FEINT_LAG
                else:                            # 그 단계의 강공격으로 내려친다
                    self._start_heavy(f, lv)
            else:
                f.sf += 1
                prev = C.heavy_stage(f.sf - 1)
                if C.heavy_stage(f.sf) != prev:
                    self.events.append(("charge_stage", f.idx, C.heavy_stage(f.sf)))
                if f.sf >= C.HEAVY_MAX_CHARGE:   # 최대 시간 → 무조건 3단계로 나간다
                    self._start_heavy(f, 3)
        elif st == St.HITSTUN and self._try_break(f):
            pass
        elif st in (St.HITSTUN, St.PARRIED, St.RECOVER, St.BREAK):
            f.sf += 1
            f.stun -= 1
            if f.stun <= 0:
                f.state = St.IDLE
                f.sf = 0
                if st == St.HITSTUN:
                    f.combo_taken = 0
                    f.break_locked = False
                    f.wall_hit = False
        elif st == St.ATTACK:
            f.sf += 1
            ad = C.ATTACKS[f.kind]
            b = f.buf
            if b in C.CANCELABLE and b != f.kind and f.can_cancel():
                self._start_attack(f, b, cancel=True)
                f.clear_buffer()
            elif f.sf >= ad.total:
                f.state = St.IDLE
                f.sf = 0
        elif st == St.RANGED:
            f.sf += 1
            if f.sf == C.RANGED.startup:
                self._spawn_projectile(f)
            if f.sf >= C.RANGED.total:
                f.state = St.IDLE
                f.sf = 0
        elif st == St.PARRY:
            f.sf += 1
            if f.sf >= C.PARRY.total:
                f.state = St.IDLE
                f.sf = 0
        elif st == St.CROUCH:
            # 손을 떼면 바로 일어선다. 숙인 채 다른 행동(공격·점프·대쉬 등)을 누르면 그 행동으로 바로 넘어간다
            f.sf += 1
            f.vx = 0.0
            if act != Act.CROUCH or not f.grounded:
                f.state = St.IDLE
                f.sf = 0
            elif f.buf is not None:
                f.state = St.IDLE
                if self._try_action(f, f.buf):
                    f.clear_buffer()
                    if f.state == St.IDLE:
                        f.sf = 0
                    return                       # 이번 프레임은 그 행동으로 끝 (아래에서 다시 숙이지 않게)
                f.state = St.CROUCH              # 쿨타임 등으로 못 나감 → 계속 숙인 채
        elif st == St.DASH:
            f.sf += 1
            if (f.buf == Act.MID and f.grounded and f.dash_dir == f.facing
                    and C.DASH_ATTACK.window_start <= f.sf <= C.DASH.duration):
                self._start_dash_attack(f)       # 앞 대쉬 중 정면 버튼 → 대쉬 공격
                f.clear_buffer()
            elif f.sf == C.DASH.duration:
                f.vx = 0.0 if f.grounded else f.dash_dir * C.WALK_SPEED
            if f.state == St.DASH and f.sf >= C.DASH.duration + C.DASH.rec(f.dash_dir == f.facing):
                f.state = St.IDLE
                f.sf = 0

        if f.state == St.IDLE:
            self._free_action(f, act)

        if f.buf_t > 0:
            f.buf_t -= 1
            if f.buf_t == 0:
                f.buf = None

    def _try_break(self, f: Fighter) -> bool:
        """경직 중 콤보 브레이커 판정. 선입력 버퍼에 들어온 상단/정면/하단 버튼을 본다.
        발동하면 True. 조건이 안 되면 버퍼를 그대로 둔다 (경직이 끝난 뒤 선입력으로 나감)."""
        b = f.buf
        if b not in C.CANCELABLE or not f.can_break or f.break_locked or f.combo_taken < 1:
            return False
        a = self.f[1 - f.idx]
        if not (a.state == St.ATTACK and a.chain >= 2 and a.kind in C.CANCELABLE):
            return False                          # 아직 콤보로 이어진 공격이 아님 (1타째)
        f.clear_buffer()
        if b != a.kind:                           # 틀렸다 → 이번 콤보 동안 잠김
            f.break_locked = True
            self.events.append(("break_fail", f.idx, b, a.kind))
            return False
        bk = C.BREAKER
        d = 1 if a.x > f.x else (-1 if a.x < f.x else f.facing)   # 상대를 내 반대쪽으로 민다
        f.break_gauge = 0.0
        f.break_locked = False
        f.state = St.BREAK
        f.sf = 0
        f.stun = f.stun_max = bk.burst
        f.combo_taken = 0
        f.kind = None
        f.chain = 0
        if f.grounded:
            f.vx = 0.0
        a.state = St.HITSTUN
        a.sf = 0
        a.kind = None
        a.chain = 0
        a.stun = a.stun_max = bk.attacker_stun
        a.vx = d * bk.push
        a.wall_hit = True                         # 브레이커로 밀려난 건 벽꽝 없음
        a.clear_buffer()
        self.hitstop = max(self.hitstop, bk.hitstop)
        pt = ((f.x + a.x) / 2, (f.y + a.y) / 2 + 70.0)
        self.events.append(("break", f.idx, a.idx, b, pt))
        return True

    def _free_action(self, f: Fighter, act: Act):
        if f.buf is not None and self._try_action(f, f.buf):
            f.clear_buffer()
            return
        if act == Act.CROUCH and f.grounded and f.vy <= 0.0:
            f.state = St.CROUCH
            f.sf = 0
            f.vx = 0.0
            return
        if act == Act.HEAVY and f.grounded:
            f.state = St.CHARGE
            f.sf = 1               # 차지 중 sf = 누르고 있던 프레임 수 (누른 첫 프레임 = 1)
            f.vx = 0.0
            return
        if act == Act.LEFT or act == Act.RIGHT:
            d = -1.0 if act == Act.LEFT else 1.0
            if f.grounded:
                f.vx = d * (C.WALK_SPEED if d * f.facing > 0 else C.BACK_WALK_SPEED)
            else:
                f.vx = max(-C.WALK_SPEED, min(C.WALK_SPEED, f.vx + d * C.AIR_ACCEL))
        elif f.grounded:
            f.vx = 0.0

    def _try_action(self, f: Fighter, a: Act) -> bool:
        g = f.grounded
        if a == Act.JUMP:
            if not g:
                return False
            f.vy = C.JUMP_VY
            return True
        if a in (Act.DASH_F, Act.DASH_B):
            if f.dash_cd > 0 or (not g and f.air_dash_used):
                return False
            f.state = St.DASH
            f.sf = 0
            f.dash_dir = f.facing if a == Act.DASH_F else -f.facing
            f.vx = f.dash_dir * C.DASH.speed
            f.dash_cd = C.DASH.cooldown
            if not g:
                f.air_dash_used = True
            return True
        if a in C.MELEE:
            self._start_attack(f, a, cancel=False)
            return True
        if a == Act.RANGED:
            if f.ranged_cd > 0:
                return False
            f.state = St.RANGED
            f.sf = 0
            f.ranged_cd = C.RANGED.cooldown
            if g:
                f.vx = 0.0
            return True
        if a == Act.PARRY:
            if f.parry_cd > 0:
                return False
            f.state = St.PARRY
            f.sf = 0
            f.parry_cd = C.PARRY.cooldown
            if g:
                f.vx = 0.0
            return True
        return False

    def _start_attack(self, f: Fighter, kind: Act, cancel: bool):
        f.state = St.ATTACK
        f.kind = kind
        f.sf = 0
        f.hit_done = False
        f.chain = f.chain + 1 if cancel else 1
        if f.grounded:
            f.vx = f.facing * 1.5 if cancel else 0.0  # 캔슬 시 살짝 전진해 콤보가 이어지게

    def _start_dash_attack(self, f: Fighter):
        self._start_attack(f, C.DASH_ATK, cancel=False)
        f.vx = f.facing * C.DASH_ATTACK.lunge_speed
        self.events.append(("dash_attack", f.idx))

    def _start_heavy(self, f: Fighter, lv: int):
        self._start_attack(f, Act.HEAVY, cancel=False)
        f.heavy_lv = lv
        self.events.append(("heavy_release", f.idx, lv))

    def _spawn_projectile(self, f: Fighter):
        r = C.RANGED
        self.proj.append(Projectile(f.idx, f.x + f.facing * r.spawn_x,
                                    f.y + r.height, f.facing * r.speed))
        self.events.append(("shoot", f.idx))

    # ------------------------------------------------------------ 물리
    def _physics(self, f: Fighter):
        if f.state == St.DASH and f.sf < C.DASH.duration:
            f.vy = 0.0
            f.vx = f.dash_dir * C.DASH.speed
        elif f.y > 0.0 or f.vy > 0.0:
            f.vy -= C.GRAVITY
        if f.state == St.ATTACK and f.grounded and f.kind == C.DASH_ATK:
            da, ad = C.DASH_ATTACK, C.ATTACKS[C.DASH_ATK]
            f.vx *= da.lunge_decay if f.sf < ad.startup + ad.active else da.stop_decay
        elif f.state == St.ATTACK and f.grounded and f.kind == Act.HEAVY:
            hs, ad = C.HEAVY_STEP, C.ATTACKS[Act.HEAVY]
            if f.sf < hs.step_start:
                f.vx = 0.0
            elif f.sf == hs.step_start:
                f.vx = f.facing * hs.step_speed      # (v8) 내려치며 한 발 내딛는다
            else:
                f.vx *= hs.step_decay if f.sf < ad.startup + ad.active else hs.stop_decay
        elif f.state == St.ATTACK and f.grounded and f.chain > 1:
            f.vx *= 0.85  # 캔슬 전진 감쇠
        v_in = f.vx                       # 벽꽝 판정용 (마찰 전 속도)
        f.x += f.vx
        f.y += f.vy
        if f.y <= 0.0:
            f.y = 0.0
            if f.vy < 0.0:
                f.vy = 0.0
            f.air_dash_used = False
            if f.state in (St.HITSTUN, St.KO, St.PARRIED):
                f.vx *= C.GROUND_FRICTION
        lo = C.WALL_MARGIN + C.BODY_W / 2
        hi = C.ARENA_W - lo
        into = -1 if (f.x <= lo and v_in < 0) else (1 if (f.x >= hi and v_in > 0) else 0)
        if into and f.state == St.HITSTUN and not f.wall_hit and abs(v_in) >= C.WALL.min_speed:
            self._wall_splat(f, into)
        f.x = min(hi, max(lo, f.x))

    def _wall_splat(self, f: Fighter, side: int):
        """경직 중 벽에 세게 부딪힘 → 경직 연장 + 살짝 튕겨 나옴 (한 콤보에 한 번)."""
        w = C.WALL
        f.wall_hit = True
        f.stun += w.extra_stun
        f.stun_max = max(f.stun_max, f.stun)
        f.vx = -side * w.bounce
        self.hitstop = max(self.hitstop, w.hitstop)
        wx = C.WALL_MARGIN if side < 0 else C.ARENA_W - C.WALL_MARGIN
        self.events.append(("wall_splat", f.idx, (wx, f.y + 70.0)))

    def _update_facing(self):
        a, b = self.f
        for me, op in ((a, b), (b, a)):
            if me.state in (St.IDLE, St.CROUCH) and abs(op.x - me.x) > 1.0:
                me.facing = 1 if op.x > me.x else -1

    def _push_apart(self):
        a, b = self.f
        if a.state == St.KO or b.state == St.KO:
            return
        if abs(a.y - b.y) >= C.BODY_H * 0.6:
            return  # 높이 차가 크면 넘어갈 수 있다 (점프로 넘기)
        lo = C.WALL_MARGIN + C.BODY_W / 2
        hi = C.ARENA_W - lo
        for _ in range(2):
            dx = b.x - a.x
            overlap = C.BODY_W - abs(dx)
            if overlap <= 0:
                return
            s = 1.0 if dx > 0 else (-1.0 if dx < 0 else float(a.facing))
            a.x -= s * overlap / 2
            b.x += s * overlap / 2
            a.x = min(hi, max(lo, a.x))
            b.x = min(hi, max(lo, b.x))

    def _move_projectiles(self):
        for p in self.proj:
            p.x += p.vx
            if p.x < 0 or p.x > C.ARENA_W:
                p.alive = False
        alive = [p for p in self.proj if p.alive]
        r2 = 2 * C.RANGED.radius
        for i, p in enumerate(alive):
            for q in alive[i + 1:]:
                if (p.alive and q.alive and p.owner != q.owner
                        and abs(p.x - q.x) < r2 and abs(p.y - q.y) < r2):
                    p.alive = q.alive = False
                    self.events.append(("clash", ((p.x + q.x) / 2, (p.y + q.y) / 2)))
        self.proj = [p for p in self.proj if p.alive]

    # ------------------------------------------------------------ 판정
    def _resolve_hits(self):
        melee = []
        for a in self.f:
            d = self.f[1 - a.idx]
            if (a.attack_active() and not a.hit_done and d.state != St.KO
                    and not d.invulnerable()):
                hb, hu = a.hitbox(), d.hurtbox()
                if _overlap(hb, hu):
                    melee.append((a, d, _center(hb, hu), a.kind, a.facing))
        shots = []
        r = C.RANGED.radius
        for p in self.proj:
            d = self.f[1 - p.owner]
            if d.state == St.KO or d.invulnerable():
                continue
            if _overlap((p.x - r, p.x + r, p.y - r, p.y + r), d.hurtbox()):
                shots.append((p, d))

        # 동시에 맞으면 둘 다 들어간다 (상쇄)
        for a, d, pt, kind, facing in melee:
            self._apply_melee(a, d, pt, kind, facing)
        for p, d in shots:
            if not p.alive:
                continue
            p.alive = False
            if d.parry_active():
                d.state = St.IDLE
                d.sf = 0
                self.hitstop = max(self.hitstop, C.HITSTOP_PARRY_PROJ)
                self.events.append(("parry_proj", d.idx, p.owner, (p.x, p.y)))
                continue
            rd = C.RANGED
            self._damage(d, p.owner, 1 if p.vx > 0 else -1, rd.damage, rd.hitstun,
                         rd.knockback, 0.0, Act.RANGED, (p.x, p.y))
            self.hitstop = max(self.hitstop, C.HITSTOP_HIT)
        self.proj = [p for p in self.proj if p.alive]

    def _apply_melee(self, a: Fighter, d: Fighter, pt, kind, facing):
        ad = C.ATTACKS[kind]
        if a.state == St.ATTACK:   # 상쇄로 이미 맞은 경우엔 상태를 건드리지 않음
            a.hit_done = True
            a.hit_frame = a.sf
        if d.parry_active() and not ad.parryable:
            self.events.append(("parry_break", d.idx, a.idx, pt))
        elif d.parry_active() and a.state == St.ATTACK:
            a.state = St.PARRIED
            a.stun = a.stun_max = C.PARRY.stun
            a.sf = 0
            a.chain = 0
            if a.grounded:
                a.vx = 0.0
            d.state = St.IDLE
            d.sf = 0
            self.hitstop = max(self.hitstop, C.HITSTOP_PARRY)
            self.events.append(("parry", d.idx, a.idx, pt))
            return
        dmg = C.HEAVY_DAMAGE[a.heavy_lv - 1] if kind == Act.HEAVY else ad.damage
        self._damage(d, a.idx, facing, dmg, ad.hitstun, ad.knockback,
                     ad.launch, kind, pt)
        self.hitstop = max(self.hitstop, C.HITSTOP_HIT)

    def _damage(self, d: Fighter, attacker: int, direction: int, dmg: int,
                hitstun: int, knockback: float, launch: float, kind, pt):
        cb = C.COMBO
        scale = max(cb.min_scale, 1.0 - cb.scaling * d.combo_taken)
        bonus = self.f[attacker].gauge_full      # 게이지가 가득 찬 쪽은 데미지 증가
        if bonus:
            scale *= 1.0 + C.BREAKER.full_bonus
        dealt = max(1, round(dmg * scale))
        d.hp = max(0, d.hp - dealt)
        d.break_gauge = min(C.BREAKER.gauge_max,
                            d.break_gauge + dealt * C.BREAKER.gain_per_damage)
        d.break_idle = 0
        d.combo_taken += 1
        d.state = St.HITSTUN
        d.sf = 0
        d.kind = None
        d.chain = 0
        d.stun = d.stun_max = hitstun
        d.vx = knockback * direction
        if launch > 0:
            d.vy = max(d.vy, launch)
        d.clear_buffer()
        self.events.append(("hit", attacker, d.idx, kind, dealt, d.combo_taken, pt))
        if bonus:
            self.events.append(("hit_bonus", attacker, d.idx, pt))
        if d.hp <= 0 and self.training:
            d.hp = C.MAX_HP
        elif d.hp <= 0:
            d.state = St.KO
            d.sf = 0
            d.vx = 4.0 * direction
            d.vy = max(d.vy, 6.0)

    def _check_end(self):
        ko = [f.hp <= 0 for f in self.f]
        if ko[0] or ko[1]:
            self.done = True
            self.winner = -1 if (ko[0] and ko[1]) else (1 if ko[0] else 0)
            self.events.append(("ko", self.winner))
        elif self.frame >= C.ROUND_FRAMES:
            self.done = True
            h0, h1 = self.f[0].hp, self.f[1].hp
            self.winner = 0 if h0 > h1 else (1 if h1 > h0 else -1)
            self.events.append(("timeup", self.winner))
