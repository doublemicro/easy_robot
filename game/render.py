"""pygame 렌더러. 게임 상태(World)를 읽어서 그리기만 한다.

부드러움을 위해:
- 2배 해상도로 그린 뒤 축소 (안티앨리어싱)
- 행동 진행 프레임에 맞춘 포즈 보간 + 상태 전환 시 짧은 블렌딩
- 공격 궤적 잔상, 대쉬 잔상, 히트 스파크, 패링 링, 화면 흔들림
(v14) 배경은 야간 경기장(arena.py), HUD는 격투 게임 스타일 (기울어진 체력바, 가운데 타이머, 이름, FIGHT!/K.O. 배너)
"""
import math
import random
import sys

import pygame
from pygame import gfxdraw

from . import arena
from . import fonts
from . import config as C
from . import poses
from .config import Act, St

SCREEN_W, SCREEN_H = 960, 540
FLOOR = 468            # 화면에서 바닥 y
SS = 1 if sys.platform == "emscripten" else 2   # 슈퍼샘플링 배율 (v16: 브라우저에선 빠르게 1)

BG_TOP = (22, 24, 33)
BG_BOT = arena.BG_MID   # (v14) 잔상·효과가 사라질 때 섞는 배경색
FLOOR_COL = (88, 94, 120)
PLAYER_COLS = [(118, 196, 255), (255, 132, 112)]
WHITE = (240, 242, 248)
HEAVY_COL = (255, 168, 64)   # 강공격(패링 불가) 경고색
HEAVY_HOT = (255, 236, 190)  # 3단계에서 섞이는 밝은 색
BREAK_COL = (130, 236, 220)  # 콤보 브레이커 / 게이지 색
LOCK_COL = (232, 70, 70)     # 브레이커 잠김 표시
# 차지 단계별 빛의 세기 (0단계는 빛 없음 → 빛이 켜지면 "떼면 진짜 나간다"는 신호)
STAGE_GLOW = (0.0, 0.45, 0.70, 0.95)
STAGE_WIDTH = (0.0, 20.0, 25.0, 30.0)
STAGE_FLASH = 3              # 단계가 바뀌는 순간 번쩍이는 프레임 수


def _dark(c, k=0.55):
    return tuple(int(v * k) for v in c)


def _mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


DUMMY_COL = (222, 190, 132)  # 훈련장 허수아비 (짚 색)
GLOVE_COLS = [(236, 72, 72), (70, 120, 236)]   # P1 빨간 글러브, P2 파란 글러브
DUMMY_GLOVE = (150, 112, 70)
BAND_COL = (248, 240, 228)   # 머리띠·허리띠
OUTLINE = (12, 13, 20)
LIGHT = (-0.6, 0.8)          # 빛 방향 (월드 좌표, 왼쪽 위)


class Renderer:
    """mode: "match" (기본), "training" (타이머·KO 문구 없음, P2를 허수아비로 그림),
    "menu" (HUD 없이 캐릭터만)."""

    def __init__(self, screen, ss=SS, mode="match"):
        self.screen = screen
        self.mode = mode
        self.cols = list(PLAYER_COLS)
        self.gloves = list(GLOVE_COLS)
        if mode == "training":
            self.cols[1] = DUMMY_COL
            self.gloves[1] = DUMMY_GLOVE
        self.ss = ss
        self.canvas = pygame.Surface((SCREEN_W * ss, SCREEN_H * ss))
        self.bg = self._make_bg()
        self.glow = arena.make_glow(ss)
        self.vignette = arena.make_vignette()
        pygame.font.init()
        self.font = pygame.font.Font(None, 30)
        self.small = pygame.font.Font(None, 22)
        self._fonts = {}
        # HUD에 보이는 이름 (장면에서 바꾼다: "AI", "HARD BOT" ...)
        self.names = ["P1", "DUMMY" if mode == "training" else "P2"]
        self.rng = random.Random(3)
        self.reset()

    def reset(self):
        self.disp = [None, None]
        self.joints = [None, None]
        self.trails = [[], []]
        self.trail_tint = list(self.cols)
        self.ghosts = [[], []]
        self.fx = []
        self.shake = 0.0
        self.hp_disp = [float(C.MAX_HP)] * 2
        self.combo = [(0, 0), (0, 0)]
        self.stage_flash = [0, 0]
        self.break_text = None       # [브레이커 쓴 사람, 남은 프레임]
        self.lock_pop = [0, 0]       # 브레이커 틀린 순간 X 표시를 크게
        self.gauge_disp = [0.0, 0.0]
        self.t = 0
        self.ox = self.oy = 0
        self.flashes = []            # 관중 카메라 플래시 [x, y, 남은 프레임]

    # ------------------------------------------------------------ 좌표/도형
    def S(self, p):
        return ((p[0] + self.ox) * self.ss, (FLOOR - p[1] + self.oy) * self.ss)

    def _disc(self, color, c, r):
        x, y, r = int(round(c[0])), int(round(c[1])), max(1, int(round(r)))
        gfxdraw.aacircle(self.canvas, x, y, r, color)
        gfxdraw.filled_circle(self.canvas, x, y, r, color)

    def _line(self, color, a, b, w):
        """월드 좌표 두 점을 두께 w(월드 픽셀)인 둥근 선으로."""
        p1, p2 = self.S(a), self.S(b)
        self._sline(color, p1, p2, w * self.ss)

    def _sline(self, color, p1, p2, w):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / L * w / 2, dx / L * w / 2
        pts = [(round(p1[0] + nx), round(p1[1] + ny)), (round(p2[0] + nx), round(p2[1] + ny)),
               (round(p2[0] - nx), round(p2[1] - ny)), (round(p1[0] - nx), round(p1[1] - ny))]
        gfxdraw.aapolygon(self.canvas, pts, color)
        gfxdraw.filled_polygon(self.canvas, pts, color)
        self._disc(color, p1, w / 2)
        self._disc(color, p2, w / 2)

    def _make_bg(self):
        return arena.make_background(self.ss)     # (v14) 야간 경기장

    def _flash(self, n=1):
        for _ in range(n):
            x, y = arena.flash_spot(self.rng)
            self.flashes.append([x, y, self.rng.randint(4, 7)])

    # ------------------------------------------------------------ 상태 갱신
    def update(self, world, events):
        """시뮬레이션 1프레임마다 한 번 호출."""
        self.t += 1
        for e in events:
            kind = e[0]
            if kind == "hit":
                _, att, dfn, k, dmg, combo, pt = e
                self.fx.append(dict(type="spark", pos=pt, t=0, life=14,
                                    col=self.cols[att], size=1.0 + 0.25 * min(combo, 4),
                                    seed=self.t))
                self.shake = max(self.shake, 3 + min(combo, 4))
                self.combo[att] = (combo, 70)
                if dmg >= 8 or combo >= 3:            # 큰 타격엔 관중 플래시
                    self._flash(2 + min(combo, 4))
            elif kind == "hit_bonus":
                self.fx.append(dict(type="ring", pos=e[3], t=0, life=10, col=BREAK_COL, size=0.6))
            elif kind == "break":
                self.fx.append(dict(type="ring", pos=e[4], t=0, life=24, col=BREAK_COL, size=2.2))
                self.fx.append(dict(type="ring", pos=e[4], t=0, life=16, col=WHITE, size=1.2))
                self.fx.append(dict(type="flash", t=0, life=8))
                self.shake = max(self.shake, 11)
                self.combo[e[2]] = (0, 0)
                self.break_text = [e[1], 60]
            elif kind == "break_fail":
                self.lock_pop[e[1]] = 18
            elif kind == "parry":
                self.fx.append(dict(type="ring", pos=e[3], t=0, life=18, col=WHITE, size=1.4))
                self.fx.append(dict(type="flash", t=0, life=6))
                self.shake = max(self.shake, 8)
            elif kind == "parry_break":
                self.fx.append(dict(type="break", pos=e[3], t=0, life=20, col=HEAVY_COL, size=1.3))
                self.shake = max(self.shake, 9)
            elif kind == "parry_proj":
                self.fx.append(dict(type="ring", pos=e[3], t=0, life=12, col=WHITE, size=0.8))
            elif kind == "clash":
                self.fx.append(dict(type="spark", pos=e[1], t=0, life=10, col=WHITE,
                                    size=0.7, seed=self.t))
            elif kind == "charge_stage":
                self.stage_flash[e[1]] = STAGE_FLASH
            elif kind == "heavy_release":
                self.stage_flash[e[1]] = STAGE_FLASH
            elif kind == "wall_splat":   # 벽꽝: 벽에 부딪힌 자리에 충격 고리 + 흔들림
                self.fx.append(dict(type="ring", pos=e[2], t=0, life=16, col=WHITE, size=1.3))
                self.fx.append(dict(type="spark", pos=e[2], t=0, life=12, col=WHITE,
                                    size=1.2, seed=self.t))
                self.shake = max(self.shake, 9)
            elif kind in ("ko", "timeup"):
                self.shake = max(self.shake, 10)
                self._flash(26)

        for i, f in enumerate(world.f):
            target = poses.target_pose(f, self.t)
            striking = (f.state == St.ATTACK and f.attack_active()) or \
                       (f.state == St.RANGED and C.RANGED.startup <= f.sf < C.RANGED.startup + 3)
            if self.disp[i] is None or striking or world.hitstop > 0:
                self.disp[i] = target
            else:
                k = 0.75 if f.state in (St.ATTACK, St.RANGED) else 0.55   # 공격은 키포즈가 촘촘해서 덜 뭉갬
                self.disp[i] = poses.lerp(self.disp[i], target, k)
            j = poses.solve(f, self.disp[i])
            self.joints[i] = j

            # 공격 궤적
            tr = self.trails[i]
            if f.state == St.ATTACK and f.sf >= C.ATTACKS[f.kind].startup - 1:
                end, root = ("foot_f", "knee_f") if f.kind == Act.LOW else ("hand_f", "elbow_f")
                ad = C.ATTACKS[f.kind]
                if f.sf < ad.startup + ad.active + 4:
                    tr.append((j[end], j[root]))
                self.trail_tint[i] = HEAVY_COL if f.kind == Act.HEAVY else self.cols[i]
            if f.state != St.ATTACK or len(tr) > 7:
                tr[:] = tr[-7:] if f.state == St.ATTACK else tr[1:]

            # 대쉬 잔상
            gh = self.ghosts[i]
            lunging = (f.state == St.ATTACK and f.kind == C.DASH_ATK
                       and f.sf < C.ATTACKS[C.DASH_ATK].startup + C.ATTACKS[C.DASH_ATK].active)
            if ((f.state == St.DASH and f.sf < C.DASH.duration) or lunging) and self.t % 2 == 0:
                gh.append([j, 10, f.facing])
            for g in gh:
                g[1] -= 1
            gh[:] = [g for g in gh if g[1] > 0]

            hp = f.hp
            self.hp_disp[i] = hp if self.hp_disp[i] < hp else \
                self.hp_disp[i] + (hp - self.hp_disp[i]) * (0.04 if world.hitstop == 0 else 0)
            c, life = self.combo[i]
            self.combo[i] = (c, max(0, life - 1))

        for e in self.fx:
            e["t"] += 1
        self.fx = [e for e in self.fx if e["t"] < e["life"]]
        for fl in self.flashes:
            fl[2] -= 1
        self.flashes = [fl for fl in self.flashes if fl[2] > 0]
        if self.rng.random() < 0.035:
            self._flash()
        self.stage_flash = [max(0, v - 1) for v in self.stage_flash]
        self.lock_pop = [max(0, v - 1) for v in self.lock_pop]
        if self.break_text:
            self.break_text[1] -= 1
            if self.break_text[1] <= 0:
                self.break_text = None
        for i, f in enumerate(world.f):
            g = f.break_gauge
            self.gauge_disp[i] = g if g < self.gauge_disp[i] else \
                self.gauge_disp[i] + (g - self.gauge_disp[i]) * 0.15   # 줄어들 땐 바로, 찰 땐 부드럽게
        self.shake *= 0.82

    # ------------------------------------------------------------ 그리기
    def draw(self, world, debug=False, overlay=None):
        cv = self.canvas
        if self.shake > 0.5:
            self.ox = math.sin(self.t * 2.7) * self.shake
            self.oy = math.cos(self.t * 3.3) * self.shake * 0.6
        else:
            self.ox = self.oy = 0
        cv.blit(self.bg, (int(self.ox * self.ss * 0.4), int(self.oy * self.ss * 0.4)))   # 배경은 덜 흔들림
        for x, y, life in self.flashes:
            arena.draw_flash(cv, self.ss, x, y, life)
        for f in world.f:                          # 선수 발밑 조명
            cx, cy = self.S((f.x, 0))
            k = max(0.0, 1 - f.y / 220)
            if k > 0:
                g = self.glow
                if k < 1:
                    g = g.copy()
                    g.fill((int(255 * k),) * 3, special_flags=pygame.BLEND_MULT)
                cv.blit(g, g.get_rect(center=(int(cx), int(cy))), special_flags=pygame.BLEND_ADD)

        for f in world.f:
            self._shadow(f)
        for i in (1, 0):
            for j, life, fc in self.ghosts[i]:
                col = _mix(BG_BOT, self.cols[i], 0.12 + 0.03 * life)
                self._figure(j, col, col, thin=True, facing=fc)
        for i in (1, 0):
            self._telegraph(world.f[i], i)
            self._smear(i)
        order = sorted((0, 1), key=lambda k: world.f[k].state == St.ATTACK)
        for i in order:
            f = world.f[i]
            col = self.cols[i]
            if f.state == St.HITSTUN and f.sf < 3:
                col = WHITE
            elif f.state == St.PARRIED:
                col = _mix(col, (255, 230, 120), 0.5 + 0.5 * math.sin(f.sf * 0.6))
            elif f.state == St.KO:
                col = _dark(col, 0.7)
            elif f.state == St.BREAK:
                col = _mix(col, BREAK_COL, 0.75 if f.sf < 6 else 0.4)
            if f.state == St.BREAK:
                self._break_aura(self.joints[i], f)
            glove = self.gloves[i] if col == self.cols[i] else _mix(self.gloves[i], col, 0.6)
            self._figure(self.joints[i], col, _dark(col), facing=f.facing, glove=glove)
            if f.break_locked:
                self._lock_mark(self.joints[i], i)
            if self.mode == "training" and i == 1:
                self._dummy_mark(self.joints[i], f)
            if f.parry_active():
                self._parry_aura(self.joints[i], f)
        for p in world.proj:
            self._projectile(p)
        self._effects()
        if debug:
            self._debug(world)

        pygame.transform.smoothscale(cv, (SCREEN_W, SCREEN_H), self.screen)
        self.screen.blit(self.vignette, (0, 0))
        if any(e["type"] == "flash" for e in self.fx):
            fl = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            fe = next(e for e in self.fx if e["type"] == "flash")
            fl.fill((255, 255, 255, int(70 * (1 - fe["t"] / fe["life"]))))
            self.screen.blit(fl, (0, 0))
        if self.mode != "menu":
            self._hud(world, overlay)
        if debug:
            self.debug_labels(world)

    def _dummy_mark(self, j, f):
        """허수아비 표시: 몸통에 과녁, 머리에 바느질 X."""
        c = self.S(((j["neck"][0] * 0.55 + j["hip"][0] * 0.45),
                    (j["neck"][1] * 0.55 + j["hip"][1] * 0.45)))
        red = (214, 72, 64) if not (f.state == St.HITSTUN and f.sf < 3) else WHITE
        self._disc(red, c, 13 * self.ss)
        self._disc(DUMMY_COL, c, 9 * self.ss)
        self._disc(red, c, 5 * self.ss)
        hx, hy = self.S(j["head"])
        r = poses.HEAD_R * 0.45 * self.ss
        st = _dark(DUMMY_COL, 0.45)
        self._sline(st, (hx - r, hy - r), (hx + r, hy + r), 2.2 * self.ss)
        self._sline(st, (hx - r, hy + r), (hx + r, hy - r), 2.2 * self.ss)

    def _break_aura(self, j, f):
        """브레이커 발동: 몸 주변으로 퍼지는 빛."""
        c = self.S(((j["neck"][0] + j["hip"][0]) / 2, (j["neck"][1] + j["hip"][1]) / 2))
        t = f.sf / max(1, f.stun_max)
        r = (40 + 50 * poses.ease_out(t)) * self.ss
        col = _mix(BG_BOT, BREAK_COL, 1 - t)
        for k in range(int(3 * self.ss)):
            gfxdraw.aacircle(self.canvas, int(c[0]), int(c[1]), max(1, int(r - k)), col)
        if t < 0.5:
            self._disc(_mix(BG_BOT, BREAK_COL, 0.35 * (1 - 2 * t)), c, r * 0.7)

    def _lock_mark(self, j, i):
        """브레이커 잠김: 머리 위 빨간 X."""
        hx, hy = self.S((j["head"][0], j["head"][1] + 28))
        r = (7 + 6 * self.lock_pop[i] / 18) * self.ss
        w = 3.2 * self.ss
        self._sline(LOCK_COL, (hx - r, hy - r), (hx + r, hy + r), w)
        self._sline(LOCK_COL, (hx - r, hy + r), (hx + r, hy - r), w)

    def _shadow(self, f):
        h = min(1.0, f.y / 150)
        cx, cy = self.S((f.x, 0))
        w = (28 - 12 * h) * self.ss
        col = (16, 17, 24) if h < 0.5 else (20, 22, 30)
        gfxdraw.filled_ellipse(self.canvas, int(cx), int(cy + 2 * self.ss), int(w), int(5 * self.ss), col)

    # ------------------------------------------------------------ 캐릭터 (v5: 격투가)
    # V자 상체 + 큰 글러브 + 머리띠. 빛은 화면 왼쪽 위에서 고정으로 들어온다고 보고
    # 팔다리는 원통, 머리·글러브는 공처럼 명암을 넣어 입체감을 준다.
    def _poly(self, color, pts):
        s = [(round(x), round(y)) for x, y in (self.S(p) for p in pts)]
        gfxdraw.aapolygon(self.canvas, s, color)
        gfxdraw.filled_polygon(self.canvas, s, color)

    def _taper(self, color, a, b, wa, wb):
        """월드 좌표 a→b, 두께가 wa에서 wb로 줄어드는 둥근 막대."""
        p1, p2 = self.S(a), self.S(b)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / L, dx / L
        ra, rb = wa * self.ss / 2, wb * self.ss / 2
        pts = [(p1[0] + nx * ra, p1[1] + ny * ra), (p2[0] + nx * rb, p2[1] + ny * rb),
               (p2[0] - nx * rb, p2[1] - ny * rb), (p1[0] - nx * ra, p1[1] - ny * ra)]
        pts = [(round(x), round(y)) for x, y in pts]
        gfxdraw.aapolygon(self.canvas, pts, color)
        gfxdraw.filled_polygon(self.canvas, pts, color)
        self._disc(color, p1, ra)
        self._disc(color, p2, rb)

    @staticmethod
    def _lit_offset(a, b, k):
        """막대 a→b에서 빛 쪽(왼쪽 위)으로 k만큼 옮기는 벡터 (월드 좌표)."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / L, dx / L
        d = nx * LIGHT[0] + ny * LIGHT[1]
        return nx * d * k, ny * d * k

    def _limb3d(self, c, a, b, wa, wb):
        """원통 명암: 그림자색 전체 → 본색(빛 쪽으로) → 하이라이트 줄."""
        self._taper(_dark(c, 0.68), a, b, wa, wb)
        ox, oy = self._lit_offset(a, b, (wa + wb) * 0.07)
        self._taper(c, (a[0] + ox, a[1] + oy), (b[0] + ox, b[1] + oy), wa * 0.7, wb * 0.7)
        hx, hy = self._lit_offset(a, b, (wa + wb) * 0.15)
        self._taper(_mix(c, WHITE, 0.42), (a[0] + hx, a[1] + hy), (b[0] + hx, b[1] + hy),
                    wa * 0.22, wb * 0.22)

    def _ball3d(self, c, center, r):
        """공 명암 (머리·글러브)."""
        x, y = self.S(center)
        ss = self.ss
        self._disc(OUTLINE, (x, y), (r + 2.2) * ss)
        self._disc(_dark(c, 0.68), (x, y), r * ss)
        # 화면 좌표는 y가 아래로 → 빛 쪽 = (LIGHT[0], -LIGHT[1])
        self._disc(c, (x + LIGHT[0] * r * 0.18 * ss, y - LIGHT[1] * r * 0.18 * ss), r * 0.78 * ss)
        self._disc(_mix(c, WHITE, 0.55), (x + LIGHT[0] * r * 0.42 * ss, y - LIGHT[1] * r * 0.42 * ss),
                   r * 0.26 * ss)

    def _boot(self, c, foot, s):
        fx, fy = foot
        pts = [(fx - 4 * s, fy + 6), (fx + 2 * s, fy + 6), (fx + 11 * s, fy + 2.5),
               (fx + 11 * s, fy - 1), (fx - 4 * s, fy - 1)]
        cx = sum(p[0] for p in pts) / 5
        cy = sum(p[1] for p in pts) / 5
        out = [(x + (x - cx) * 0.35, y + (y - cy) * 0.45) for x, y in pts]
        self._poly(OUTLINE, out)
        self._poly(c, pts)
        self._poly(_dark(c, 0.6), [(fx - 4 * s, fy + 1), (fx + 11 * s, fy + 1),
                                   (fx + 11 * s, fy - 1), (fx - 4 * s, fy - 1)])   # 밑창

    def _figure(self, j, col, back_col, thin=False, facing=1, glove=None):
        s = facing
        if thin:          # 대쉬 잔상: 명암 없이 실루엣만
            for a, b, w in (("shoulder", "elbow_b", 8), ("elbow_b", "hand_b", 6.5),
                            ("hip", "knee_b", 9), ("knee_b", "foot_b", 7), ("hip", "knee_f", 9),
                            ("knee_f", "foot_f", 7), ("shoulder", "elbow_f", 8), ("elbow_f", "hand_f", 6.5)):
                self._taper(col, j[a], j[b], w, w * 0.8)
            self._poly(col, self._vtorso(j))
            self._disc(col, self.S(j["head"]), (poses.HEAD_R + 0.5) * self.ss)
            self._disc(col, self.S(j["hand_f"]), 7 * self.ss)
            return
        glove = glove or col
        back = _mix(col, BG_BOT, 0.42)
        gback = _mix(glove, BG_BOT, 0.38)
        O = 3.6        # 외곽선 두께

        def limb(c, a, b, wa, wb):
            self._taper(OUTLINE, j[a], j[b], wa + O, wb + O)
            self._limb3d(c, j[a], j[b], wa, wb)

        # 뒤쪽 팔다리 (어둡게 → 앞뒤 구분)
        limb(back, "hip", "knee_b", 11, 8)
        limb(back, "knee_b", "foot_b", 8, 6.5)
        self._boot(back, j["foot_b"], s)
        limb(back, "shoulder", "elbow_b", 9, 7)
        limb(back, "elbow_b", "hand_b", 7, 6)
        self._ball3d(gback, j["hand_b"], 7)
        # 상체 (V자)
        v = self._vtorso(j)
        cx = sum(p[0] for p in v) / 4
        cy = sum(p[1] for p in v) / 4
        self._poly(OUTLINE, [(x + (x - cx) * 0.16, y + (y - cy) * 0.1) for x, y in v])
        self._poly(_dark(col, 0.68), v)
        lit = [(cx + (x - cx) * 0.78 + LIGHT[0] * 2.2, cy + (y - cy) * 0.9 + LIGHT[1] * 1.2) for x, y in v]
        self._poly(col, lit)
        hip, neck = j["hip"], j["neck"]
        ux, uy = neck[0] - hip[0], neck[1] - hip[1]
        L = math.hypot(ux, uy) or 1e-6
        ux, uy = ux / L, uy / L
        # 가슴 하이라이트 (빛 쪽 어깨 근처)
        cht = (neck[0] - ux * 9 + LIGHT[0] * 5, neck[1] - uy * 9 + LIGHT[1] * 5)
        self._disc(_mix(col, WHITE, 0.3), self.S(cht), 4.5 * self.ss)
        # 허리띠
        b0 = (hip[0] + ux * 4, hip[1] + uy * 4)
        self._line(OUTLINE, (b0[0] + uy * 9, b0[1] - ux * 9), (b0[0] - uy * 9, b0[1] + ux * 9), 7)
        self._line(BAND_COL, (b0[0] + uy * 8, b0[1] - ux * 8), (b0[0] - uy * 8, b0[1] + ux * 8), 4.5)
        # 앞쪽 다리
        limb(col, "hip", "knee_f", 11, 8)
        limb(col, "knee_f", "foot_f", 8, 6.5)
        self._boot(col, j["foot_f"], s)
        # 머리 + 머리띠 + 눈
        hc = j["head"]
        self._ball3d(col, hc, poses.HEAD_R + 0.5)
        hx, hy = hc[0] - neck[0], hc[1] - neck[1]
        Lh = math.hypot(hx, hy) or 1e-6
        hx, hy = hx / Lh, hy / Lh                 # 목 → 머리
        px, py = hy * s, -hx * s                  # 얼굴 앞쪽
        ba = (hc[0] - px * 11 + hx * 4, hc[1] - py * 11 + hy * 4)
        bb = (hc[0] + px * 11 + hx * 4, hc[1] + py * 11 + hy * 4)
        self._line(OUTLINE, ba, bb, 6)
        self._line(BAND_COL, ba, bb, 3.8)
        wave = math.sin(self.t * 0.3) * 2
        t1 = (ba[0] - px * 7 - hx * 3, ba[1] - py * 7 - hy * 3 + wave)
        t2 = (t1[0] - px * 6 - hx * 2, t1[1] - py * 6 - hy * 2 - wave * 1.5)
        self._taper(BAND_COL, ba, t1, 4, 3)
        self._taper(BAND_COL, t1, t2, 3, 1.8)
        self._disc(OUTLINE, self.S((hc[0] + px * 6 - hx, hc[1] + py * 6 - hy)), 2.2 * self.ss)
        # 앞팔 (가장 마지막 = 맨 앞)
        limb(col, "shoulder", "elbow_f", 9, 7)
        limb(col, "elbow_f", "hand_f", 7, 6)
        self._ball3d(glove, j["hand_f"], 8)

    def _vtorso(self, j):
        hip, neck = j["hip"], j["neck"]
        dx, dy = neck[0] - hip[0], neck[1] - hip[1]
        L = math.hypot(dx, dy) or 1e-6
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        a = (hip[0] - ux * 2, hip[1] - uy * 2)
        b = (neck[0] + ux * 2, neck[1] + uy * 2)
        wa, wb = 6.0, 15.0
        return [(a[0] + nx * wa, a[1] + ny * wa), (b[0] + nx * wb, b[1] + ny * wb),
                (b[0] - nx * wb, b[1] - ny * wb), (a[0] - nx * wa, a[1] - ny * wa)]

    def _heavy_glow(self, f, i, lv):
        """강공격 경고빛: 단계가 오를수록 밝고 굵어진다. 3단계는 깜빡임, 단계가 바뀌면 번쩍."""
        if lv <= 0:
            return
        j = self.joints[i]
        k = STAGE_GLOW[lv]
        col = HEAVY_COL if lv < 3 else _mix(HEAVY_COL, HEAVY_HOT, 0.45)
        if lv == 3 and f.state == St.CHARGE:
            k *= 0.7 + 0.3 * math.sin(f.sf * 1.6)
        if self.stage_flash[i] > 0:
            col = _mix(col, WHITE, 0.75)
            k = 1.0
        c = _mix(BG_BOT, col, k)
        w = STAGE_WIDTH[lv]
        for a, b, e in (("shoulder", "elbow_f", "hand_f"), ("shoulder", "elbow_b", "hand_b")):
            self._line(c, j[a], j[b], w)
            self._line(c, j[b], j[e], w)
        self._disc(c, self.S(((j["hand_f"][0] + j["hand_b"][0]) / 2,
                              (j["hand_f"][1] + j["hand_b"][1]) / 2)), (4 + 5 * lv) * self.ss)

    def _telegraph(self, f, i):
        """예비동작 중 휘두를 팔/다리에 은은한 빛 (종류 구분은 자세로 한다)."""
        j = self.joints[i]
        if f.state == St.CHARGE or (f.state == St.ATTACK and f.kind == Act.HEAVY
                                    and f.sf < C.ATTACKS[Act.HEAVY].startup):
            self._heavy_glow(f, i, f.charge_stage)
        elif f.state == St.ATTACK and f.sf < C.ATTACKS[f.kind].startup:
            t = (f.sf + 1) / C.ATTACKS[f.kind].startup
            col = _mix(BG_BOT, _mix(self.cols[i], WHITE, 0.4), 0.35 + 0.35 * t)
            if f.kind == Act.LOW:
                a, b, c = j["hip"], j["knee_f"], j["foot_f"]
            else:
                a, b, c = j["shoulder"], j["elbow_f"], j["hand_f"]
            self._line(col, a, b, 22)
            self._line(col, b, c, 22)
            self._disc(col, self.S(c), (13 + 4 * t) * self.ss)   # 휘두를 손/발 끝 강조
        elif f.state == St.RANGED and f.sf < C.RANGED.startup:
            t = (f.sf + 1) / C.RANGED.startup
            col = _mix(BG_BOT, _mix(self.cols[i], WHITE, 0.5), 0.3 + 0.5 * t)
            self._disc(col, self.S(j["hand_f"]), (5 + 7 * t) * self.ss)

    def _smear(self, i):
        tr = self.trails[i]
        if len(tr) < 2:
            return
        n = len(tr)
        tint = self.trail_tint[i]
        for k in range(1, n):
            t = k / n
            col = _mix(BG_BOT, _mix(tint, WHITE, 0.6), 0.25 + 0.6 * t)
            self._line(col, tr[k - 1][0], tr[k][0], 3 + 9 * t)
        pts = [self.S(e) for e, _ in tr] + [self.S(tr[-1][1]), self.S(tr[0][1])]
        pts = [(round(x), round(y)) for x, y in pts]
        col = _mix(BG_BOT, tint, 0.28)
        gfxdraw.filled_polygon(self.canvas, pts, col)

    def _parry_aura(self, j, f):
        c = self.S(((j["hand_f"][0] + j["hand_b"][0]) / 2, (j["hand_f"][1] + j["hand_b"][1]) / 2))
        r = (24 + 3 * math.sin(f.sf)) * self.ss
        for k in range(3):
            gfxdraw.aacircle(self.canvas, int(c[0]), int(c[1]), int(r - k * self.ss), (210, 230, 255))

    def _projectile(self, p):
        col = self.cols[p.owner]
        r = C.RANGED.radius
        for k in range(5, 0, -1):
            pos = (p.x - p.vx * k * 1.4, p.y)
            self._disc(_mix(BG_BOT, col, 0.5 - k * 0.08), self.S(pos), (r - k) * self.ss)
        self._disc(_mix(BG_BOT, col, 0.35), self.S((p.x, p.y)), (r + 5) * self.ss)
        self._disc(col, self.S((p.x, p.y)), r * self.ss)
        self._disc(WHITE, self.S((p.x, p.y)), r * 0.45 * self.ss)

    def _effects(self):
        for e in self.fx:
            t = e["t"] / e["life"]
            if e["type"] == "spark":
                c = self.S(e["pos"])
                col = _mix(e["col"], WHITE, 0.5 * (1 - t))
                for k in range(9):
                    ang = k * 2 * math.pi / 9 + e["seed"] * 0.7
                    r0 = (6 + 30 * t) * e["size"] * self.ss
                    r1 = (14 + 46 * t) * e["size"] * self.ss
                    a = (c[0] + math.cos(ang) * r0, c[1] + math.sin(ang) * r0)
                    b = (c[0] + math.cos(ang) * r1, c[1] + math.sin(ang) * r1)
                    self._sline(_mix(BG_BOT, col, 1 - t), a, b, (4 - 3 * t) * self.ss)
                if t < 0.3:
                    self._disc(WHITE, c, (14 * (1 - t / 0.3)) * e["size"] * self.ss)
            elif e["type"] == "break":
                # 패링이 깨짐: 주황 파편이 튀어나감
                c = self.S(e["pos"])
                col = _mix(BG_BOT, e["col"], 1 - t)
                for k in range(12):
                    ang = k * 2 * math.pi / 12 + 0.3
                    r0 = (10 + 50 * poses.ease_out(t)) * e["size"] * self.ss
                    a = (c[0] + math.cos(ang) * r0, c[1] + math.sin(ang) * r0)
                    b = (a[0] + math.cos(ang + 0.6) * 10 * self.ss, a[1] + math.sin(ang + 0.6) * 10 * self.ss)
                    self._sline(col, a, b, (5 - 4 * t) * self.ss)
            elif e["type"] == "ring":
                c = self.S(e["pos"])
                r = (10 + 60 * poses.ease_out(t)) * e["size"] * self.ss
                col = _mix(BG_BOT, e["col"], 1 - t)
                for k in range(int(4 * self.ss)):
                    gfxdraw.aacircle(self.canvas, int(c[0]), int(c[1]), max(1, int(r - k)), col)

    def _debug(self, world):
        ss = self.ss
        for f in world.f:
            x0, x1, y0, y1 = f.hurtbox()
            a, b = self.S((x0, y1)), self.S((x1, y0))
            pygame.draw.rect(self.canvas, (80, 220, 120), (a[0], a[1], b[0] - a[0], b[1] - a[1]), ss)
            if f.state == St.ATTACK:
                x0, x1, y0, y1 = f.hitbox()
                a, b = self.S((x0, y1)), self.S((x1, y0))
                col = (255, 70, 70) if f.attack_active() else (120, 60, 60)
                pygame.draw.rect(self.canvas, col, (a[0], a[1], b[0] - a[0], b[1] - a[1]), ss)
        r = C.RANGED.radius
        for p in world.proj:
            a, b = self.S((p.x - r, p.y + r)), self.S((p.x + r, p.y - r))
            pygame.draw.rect(self.canvas, (255, 70, 70), (a[0], a[1], b[0] - a[0], b[1] - a[1]), ss)

    # ------------------------------------------------------------ 승리 화면 (v15)
    def draw_victory(self, winner, t, x=330):
        """승리 화면 배경 + 승자(승리 모션)를 1.6배 클로즈업해서 그린다. winner가 -1이면 두 선수를 나란히 (무승부)."""
        self.t += 1
        for fl in self.flashes:
            fl[2] -= 1
        self.flashes = [fl for fl in self.flashes if fl[2] > 0]
        if self.rng.random() < (0.5 if t < 40 else 0.12):     # 처음엔 플래시가 쏟아진다
            self._flash(1 + (t < 20))
        self.ox = self.oy = 0
        cv = self.canvas
        cv.blit(self.bg, (0, 0))
        for fx, fy, life in self.flashes:
            arena.draw_flash(cv, self.ss, fx, fy, life)

        class _Pos:          # poses.solve가 읽는 값만 (x, y, facing)
            def __init__(self, x, facing):
                self.x, self.y, self.facing = x, 0.0, facing

        if winner == -1:
            figs = [(0, _Pos(170, 1), poses.idle_pose(self.t)), (1, _Pos(420, -1), poses.idle_pose(self.t))]   # 왼쪽 (오른쪽은 기록창)
        else:
            figs = [(winner, _Pos(x, 1), poses.victory_pose(t))]
        for i, f, p in figs:
            cx, cy = self.S((f.x, 0))
            cv.blit(self.glow, self.glow.get_rect(center=(int(cx), int(cy))), special_flags=pygame.BLEND_ADD)
            self._shadow(f)
            col = self.cols[i]
            self._figure(poses.solve(f, p), col, _dark(col), facing=f.facing, glove=self.gloves[i])
        if winner == -1:
            pygame.transform.smoothscale(cv, (SCREEN_W, SCREEN_H), self.screen)
        else:                                         # 승자 클로즈업 (16:9로 잘라서 키움)
            zw = 600
            zh = zw * SCREEN_H / SCREEN_W
            crop = pygame.Rect(int((x - zw * 0.5) * self.ss), int((FLOOR + 40 - zh) * self.ss),
                               int(zw * self.ss), int(zh * self.ss))
            crop.clamp_ip(cv.get_rect())
            pygame.transform.smoothscale(cv.subsurface(crop), (SCREEN_W, SCREEN_H), self.screen)
        self.screen.blit(self.vignette, (0, 0))

    # ------------------------------------------------------------ HUD (v14: 격투 게임 스타일)
    BAR_W, BAR_H, BAR_Y, SKEW = 400, 22, 18, 10
    HUD_FONT = "display"       # (v16) 게임에 들어 있는 폰트 (game/fonts.py): Anton
    SMALL_FONT = "small"       # 나눔고딕 굵게

    def _font(self, size, names=None):
        names = names or self.HUD_FONT
        if names == "display":
            return fonts.display(size)
        if names == "small":
            return fonts.korean(size, bold=True)
        key = (names, size)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont(names, size)
        return self._fonts[key]

    def _text(self, s, size, col, pos, anchor="topleft", outline=2, names=None, alpha=None):
        f = self._font(size, names)
        img = f.render(s, True, col)
        r = img.get_rect(**{anchor: pos})
        if outline:
            sh = f.render(s, True, (0, 0, 0))
            if alpha is not None:
                sh.set_alpha(alpha)
            for dx, dy in ((-outline, 0), (outline, 0), (0, -outline), (0, outline), (outline, outline)):
                self.screen.blit(sh, r.move(dx, dy))
        if alpha is not None:
            img.set_alpha(alpha)
        self.screen.blit(img, r)
        return r

    @staticmethod
    def _para(a, b, y, h, slant):
        """기울어진 막대 (a~b, 위쪽이 slant만큼 밀림)."""
        return [(a + slant, y), (b + slant, y), (b, y + h), (a, y + h)]

    def _span(self, i, x0, w, v):
        """i쪽 막대에서 비율 v만큼 찬 부분의 x 범위 (P1은 바깥(왼쪽)부터, P2는 바깥(오른쪽)부터)."""
        v = max(0.0, min(1.0, v))
        return (x0, x0 + w * v) if i == 0 else (x0 + w * (1 - v), x0 + w)

    def _hud(self, world, overlay):
        scr = self.screen
        W, H, Y, K = self.BAR_W, self.BAR_H, self.BAR_Y, self.SKEW
        for i, f in enumerate(world.f):
            x0 = 24 if i == 0 else SCREEN_W - 24 - W
            sl = K if i == 0 else -K
            # 틀
            pygame.draw.polygon(scr, (6, 6, 10), self._para(x0 - 4, x0 + W + 4, Y - 4, H + 8, sl))
            pygame.draw.polygon(scr, (46, 18, 24), self._para(x0, x0 + W, Y, H, sl))
            # 깎인 체력 (천천히 따라옴) → 지금 체력
            a, b = self._span(i, x0, W, self.hp_disp[i] / C.MAX_HP)
            if b - a > 1:
                pygame.draw.polygon(scr, (250, 210, 150), self._para(a, b, Y, H, sl))
            a, b = self._span(i, x0, W, f.hp / C.MAX_HP)
            col = self.cols[i]
            if f.hp <= 25 and f.hp > 0:                       # 체력이 적으면 붉게 깜빡
                col = _mix(col, (255, 60, 60), 0.5 + 0.5 * math.sin(self.t * 0.25))
            if b - a > 1:
                pygame.draw.polygon(scr, _dark(col, 0.8), self._para(a, b, Y, H, sl))
                hi = sl * 0.45                              # 위쪽 절반 밝게 (광택)
                pygame.draw.polygon(scr, _mix(col, WHITE, 0.3),
                                    self._para(a + sl - hi, b + sl - hi, Y, H * 0.45, hi))
            for k in range(1, 10):                           # 10% 눈금
                tx = x0 + W * k / 10
                pygame.draw.line(scr, (0, 0, 0), (tx + sl, Y), (tx, Y + H), 1)
            pygame.draw.polygon(scr, (176, 150, 96), self._para(x0 - 4, x0 + W + 4, Y - 4, H + 8, sl), 2)
            # 이름
            nm = self.names[i]
            anchor = "topleft" if i == 0 else "topright"
            nx = 30 if i == 0 else SCREEN_W - 30
            r = self._text(nm, 22, WHITE, (nx, Y + H + 6), anchor)
            if self.mode == "training" and i == 1:
                self._gauge(scr, i, f, x0, W, Y + H + 40)
                continue
            # 쿨타임 (원거리, 대쉬). 패링 쿨타임은 일부러 보여주지 않는다 (설계 결정)
            base = r.right + 16 if i == 0 else r.left - 16
            for k, (cd, mx, label) in enumerate(((f.ranged_cd, C.RANGED.cooldown, "SHOT"),
                                                 (f.dash_cd, C.DASH.cooldown, "DASH"))):
                bw = 46
                bx = base + k * 92 if i == 0 else base - bw - k * 92
                by = Y + H + 16
                pygame.draw.polygon(scr, (24, 26, 36), self._para(bx, bx + bw, by, 6, 3))
                ready = cd == 0
                fill = 1.0 if ready else 1 - cd / mx
                pygame.draw.polygon(scr, self.cols[i] if ready else (100, 104, 124),
                                    self._para(bx, bx + bw * fill, by, 6, 3))
                lc = (210, 214, 230) if ready else (110, 114, 132)
                self._text(label, 11, lc, (bx + bw + 4, by + 3) if i == 0 else (bx - 4, by + 3),
                           "midleft" if i == 0 else "midright", outline=0, names=self.SMALL_FONT)
            self._gauge(scr, i, f, x0, W, Y + H + 40)
            # 콤보
            c, life = self.combo[i]
            if c >= 2 and life > 0:
                al = min(255, life * 8)
                pop = 1 + 0.35 * max(0, (life - 62) / 8)
                num = self._font(56).render(str(c), True, self.cols[i])
                if pop > 1:
                    num = pygame.transform.rotozoom(num, 0, pop)
                nx = 34 if i == 0 else SCREEN_W - 34 - num.get_width() - 70
                rr = self._text(str(c), int(56 * pop), self.cols[i], (nx, 104), outline=3, alpha=al)
                self._text("HITS", 22, WHITE, (rr.right + 6, rr.bottom - 8), "bottomleft", alpha=al)
        if self.break_text:
            who, life = self.break_text
            al = min(255, life * 10)
            if self.mode == "training":
                self._text("COMBO BREAKER!", 32, BREAK_COL, (SCREEN_W // 2, 104), "midtop", 3, alpha=al)
            elif who == 0:
                self._text("COMBO BREAKER!", 32, BREAK_COL, (34, 170), outline=3, alpha=al)
            else:
                self._text("COMBO BREAKER!", 32, BREAK_COL, (SCREEN_W - 34, 170), "topright", 3, alpha=al)
        # 가운데 타이머
        box = pygame.Rect(0, 0, 72, 54)
        box.midtop = (SCREEN_W // 2, 8)
        pygame.draw.rect(scr, (8, 8, 14), box.inflate(6, 6), border_radius=10)
        pygame.draw.rect(scr, (20, 20, 30), box, border_radius=8)
        pygame.draw.rect(scr, (176, 150, 96), box.inflate(6, 6), 2, border_radius=10)
        if self.mode == "match":
            remain = max(0, C.ROUND_FRAMES - world.frame) // C.FPS
            tc = WHITE
            if remain <= 10 and not world.done:
                tc = _mix(WHITE, (255, 80, 70), 0.5 + 0.5 * math.sin(self.t * 0.3))
            self._text(str(remain), 40, tc, box.center, "center", outline=0)
        else:
            self._text("∞", 34, (200, 204, 220), box.center, "center", outline=0, names="small")
        # 슬로우/정지 표시 (타이머 아래)
        for k, line in enumerate(overlay or []):
            img = self._font(13, self.SMALL_FONT).render(line, True, (220, 224, 240))
            chip = img.get_rect(midtop=(SCREEN_W // 2, box.bottom + 10 + 22 * k)).inflate(14, 6)
            pygame.draw.rect(scr, (20, 20, 30), chip, border_radius=6)
            scr.blit(img, img.get_rect(center=chip.center))
        if self.mode != "match":
            return
        # 시작: FIGHT!
        fr = world.frame
        if fr < 60 and not world.done:
            k = min(1.0, fr / 6)
            al = 255 if fr < 40 else int(255 * (60 - fr) / 20)
            img = self._font(110).render("FIGHT!", True, (255, 214, 120))
            sh = self._font(110).render("FIGHT!", True, (0, 0, 0))
            z = 1.6 - 0.6 * k
            img, sh = pygame.transform.rotozoom(img, 0, z), pygame.transform.rotozoom(sh, 0, z)
            img.set_alpha(al)
            sh.set_alpha(al)
            c = (SCREEN_W // 2, 210)
            for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4), (4, 4)):
                scr.blit(sh, sh.get_rect(center=(c[0] + dx, c[1] + dy)))
            scr.blit(img, img.get_rect(center=c))
        # 끝: K.O. / TIME 배너
        if world.done:
            band = pygame.Surface((SCREEN_W, 150), pygame.SRCALPHA)
            band.fill((0, 0, 0, 160))
            scr.blit(band, (0, 150))
            pygame.draw.line(scr, (176, 150, 96), (0, 150), (SCREEN_W, 150), 2)
            pygame.draw.line(scr, (176, 150, 96), (0, 299), (SCREEN_W, 299), 2)
            msg = "K.O." if any(f.hp <= 0 for f in world.f) else "TIME"
            self._text(msg, 96, (255, 214, 120), (SCREEN_W // 2, 214), "center", outline=4)
            w = world.winner
            sub = "DRAW" if w == -1 else f"{self.names[w]}  WINS"
            self._text(sub, 30, WHITE, (SCREEN_W // 2, 272), "center", outline=2)
            self._text("R  REMATCH      ESC  MENU", 14, (170, 174, 196), (SCREEN_W // 2, 318), "center",
                       outline=0, names=self.SMALL_FONT)

    def _gauge(self, scr, i, f, x, bw, y):
        """콤보 브레이커 게이지. 가득 차면 빛나고 MAX(데미지 증가) 표시, 잠기면 빨간색."""
        gw, gh = 170, 8
        gx = x if i == 0 else x + bw - gw
        sl = 4 if i == 0 else -4
        pygame.draw.polygon(scr, (6, 6, 10), self._para(gx - 2, gx + gw + 2, y - 2, gh + 4, sl))
        pygame.draw.polygon(scr, (28, 36, 44), self._para(gx, gx + gw, y, gh, sl))
        full = f.gauge_full
        bk = C.BREAKER
        left = bk.decay_delay - f.break_idle          # 줄어들기 시작할 때까지 남은 프레임
        if full:
            col = _mix(BREAK_COL, WHITE, 0.25 + 0.25 * math.sin(self.t * 0.25))
            if left < 60 and (self.t // 4) % 2:      # 1초 뒤 줄어들기 시작 → 깜빡임
                col = _mix((70, 110, 110), BREAK_COL, 0.5)
        else:
            col = _mix((70, 110, 110), BREAK_COL, 0.5)
            if left < 0 and f.break_gauge > 0:        # 줄어드는 중: 어둡게
                col = _mix((50, 70, 74), BREAK_COL, 0.3)
        if f.break_locked:
            col = LOCK_COL
        a, b = self._span(i, gx, gw, self.gauge_disp[i] / bk.gauge_max)
        if b - a > 1:
            pygame.draw.polygon(scr, col, self._para(a, b, y, gh, sl))
        # 브레이커 사용 가능선 (60) 눈금
        mk = gw * bk.use_min / bk.gauge_max
        mx = gx + mk if i == 0 else gx + gw - mk
        pygame.draw.line(scr, WHITE if f.can_break else (120, 126, 140), (mx + sl, y - 3), (mx, y + gh + 2), 2)
        ok = f.can_break and not f.break_locked
        if full:
            tag = f"BREAK  MAX +{round(bk.full_bonus * 100)}%"
        elif ok:
            tag = "BREAK OK"
        else:
            tag = "BREAK"
        if f.break_locked:
            tag = "LOCKED"
        lc = col if (ok or f.break_locked) else (130, 134, 152)
        if i == 0:
            self._text(tag, 12, lc, (gx + gw + 12, y + gh // 2), "midleft", outline=0, names=self.SMALL_FONT)
        else:
            self._text(tag, 12, lc, (gx - 12, y + gh // 2), "midright", outline=0, names=self.SMALL_FONT)

    def debug_labels(self, world):
        for f in world.f:
            j = self.joints[f.idx]
            if not j:
                continue
            hx, hy = j["head"]
            lab = C.LABELS[f.label]
            extra = f" Lv{f.charge_stage}" if f.charge_stage else ""
            s = self.small.render(f"{lab} {f.sf}{extra}", True, (200, 255, 200))
            self.screen.blit(s, (hx + self.ox - s.get_width() // 2, FLOOR - hy - 40 + self.oy))
