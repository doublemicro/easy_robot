"""직접 플레이. 실행하면 메뉴가 뜬다 (마우스 클릭 또는 ↑/↓ + Enter).

    python play.py                     # 메뉴
    python play.py --start training    # 바로 훈련장
    python play.py --start match       # 바로 봇 대전 (P2 = 중간 봇)
    python play.py --start match --p2 hard    # 상위 봇 (easy / normal / hard)
    python play.py --start match --p2 human   # 2인 플레이
    python play.py --start match --p2 idle    # 가만히 있는 상대
    python play.py --start match --p2 ai      # 학습된 AI (기본 models/ai.npz, --ai 로 바꿈)

메뉴
  훈련장   허수아비 상대로 기술·콤보 데미지 확인 (넉백/경직은 실전과 같음, 체력 무한)
  봇 대전  하위 / 중간 / 상위 봇 중에서 골라 1:1
  AI 대전  강화학습으로 훈련한 AI와 1:1 (models/ai.npz, numpy로 돌아감)
  조작법   P1/P2 키와 기술 설명

P1 조작: A/D 이동, W 점프, S 숙이기(누르는 동안), Space 대쉬 (뒤 방향키를 누른 채면 백대쉬)
         U 상단, J 정면, M 하단, K 원거리, L 패링, I 길게 누르기 = 강공격
P2 조작: ←/→ 이동, ↑ 점프, ↓ 숙이기, 오른쪽 Shift 대쉬
         Numpad 8 상단, 5 정면, 2 하단, 6 원거리, 4 패링, 9 길게 누르기 = 강공격
공통:    F1 판정박스, F2 슬로우(1/4), P 일시정지, R 위치 초기화/재시작,
         Esc 일시정지 메뉴 (계속 / 다시 시작 / 조작법 / 메뉴로)  ← (v14) 화면 아래 버튼·설명 대신
훈련장:  T 기록 초기화, B 허수아비 브레이커(끔/찍기/반응), G 내 브레이커 게이지 항상 MAX, Tab 기록창 (간단 → 전체 → 끔)

콤보 브레이커: 맞을수록 게이지가 찬다. 가득 찬 상태에서 콤보 2타째부터,
              맞고 있는 공격과 같은 버튼(상단/정면/하단)을 누르면 콤보를 끊는다.
              틀리면 그 콤보 동안 잠김(머리 위 X). 게이지가 가득 찬 동안엔 내 공격 데미지 +15%.
"""
import argparse
import importlib.util
import math
import os
import random

import pygame

from game import config as C
from game import ui
from game.config import Act, St
from game.core import World
from game.render import BREAK_COL, FLOOR, HEAVY_COL, SCREEN_H, SCREEN_W, Renderer
from game.training import REFILL_AFTER, DamageMeter, kind_label
from ai.bots import BOTS, TIERS
from ai.obs import build_obs
from ai.perception import Perception, Reaction, SeenWorld

K = pygame
P1_KEYS = dict(left=[K.K_a], right=[K.K_d], jump=[K.K_w], dash=[K.K_SPACE],
               high=[K.K_u], mid=[K.K_j], low=[K.K_m], ranged=[K.K_k], parry=[K.K_l],
               heavy=[K.K_i], crouch=[K.K_s])
P2_KEYS = dict(left=[K.K_LEFT], right=[K.K_RIGHT], jump=[K.K_UP], dash=[K.K_RSHIFT, K.K_KP0],
               high=[K.K_KP8], mid=[K.K_KP5], low=[K.K_KP2], ranged=[K.K_KP6], parry=[K.K_KP4],
               heavy=[K.K_KP9], crouch=[K.K_DOWN])
PRIORITY = (("parry", Act.PARRY), ("high", Act.HIGH), ("mid", Act.MID), ("low", Act.LOW),
            ("ranged", Act.RANGED))


class HumanController:
    def __init__(self, keys):
        self.k = keys

    def __call__(self, world, me, presses, held):
        k = self.k
        f = world.f[me]
        pressed = lambda name: any(c in presses for c in k[name])
        down = lambda name: any(held[c] for c in k[name])
        for name, act in PRIORITY:
            if pressed(name):
                return act
        if pressed("dash"):
            back = (down("left") and f.facing > 0) or (down("right") and f.facing < 0)
            return Act.DASH_B if back else Act.DASH_F
        if pressed("jump"):
            return Act.JUMP
        if down("heavy"):
            return Act.HEAVY
        if down("crouch"):                 # 숙이는 동안은 걷지 않는다 (공격·점프·대쉬는 바로 나감)
            return Act.CROUCH
        l, r = down("left"), down("right")
        if l and not r:
            return Act.LEFT
        if r and not l:
            return Act.RIGHT
        return Act.IDLE


AI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ai.npz")


def ai_unavailable(path):
    """AI 대전을 못 하는 이유 (할 수 있으면 None). 파일만 확인한다."""
    if not os.path.exists(path):
        return ui.T("AI 파일이 없음 (models/ai.npz)", "AI file missing (models/ai.npz)")
    if path.endswith(".pt") and importlib.util.find_spec("torch") is None:
        return ui.T("PyTorch가 없음 (pip install torch)", "needs PyTorch (pip install torch)")
    return None


def load_ai(path):
    """학습된 AI를 봇처럼 불러온다. 못 불러오면 RuntimeError (메뉴에서 이유를 보여 줌).
    (v16) 기본은 models/ai.npz를 numpy로 돌린다 (PyTorch 불필요, 웹에서도 됨). .pt를 주면 PyTorch로."""
    why = ai_unavailable(path)
    if why:
        raise RuntimeError(why)
    try:
        if path.endswith(".pt"):
            from ai.policy import PolicyBot    # torch는 이때만 불러온다
            return PolicyBot.load(path)
        from ai.policy_np import NumpyPolicyBot
        return NumpyPolicyBot.load(path)
    except ValueError:                 # 규칙이 바뀌어 관측/행동 크기가 다름
        raise RuntimeError(ui.T("AI가 지금 게임 규칙과 안 맞음 → 다시 학습", "AI outdated, retrain"))


class BotController:
    """봇은 학습 환경과 똑같이 4프레임마다 결정하고 반응 지연된 관측을 받는다."""

    def __init__(self, bot, eye: Perception, repeat=4):
        self.bot, self.eye, self.repeat = bot, eye, repeat
        self.reset()

    def reset(self):
        self.k = 0
        self.a = Act.IDLE

    def __call__(self, world, me, presses, held):
        first = self.k % self.repeat == 0
        if first:
            obs = build_obs(world, self.eye)
            self.a = Act(self.bot.act(obs, SeenWorld(world, self.eye), me))
            if getattr(self.bot, "relative_actions", False) and obs[4] * world.f[me].facing < 0:
                # 학습된 AI는 "상대가 오른쪽" 기준으로 행동한다 (env와 같은 변환)
                self.a = {Act.LEFT: Act.RIGHT, Act.RIGHT: Act.LEFT}.get(self.a, self.a)
        self.k += 1
        return self.a if (first or self.a not in C.BUFFERABLE) else Act.IDLE


def _idle(world, me, presses, held):
    return Act.IDLE


class DummyBreaker:
    """훈련장 허수아비의 콤보 브레이커 (내 콤보가 끊기는지 연습용).

    mode 0 끔 / 1 찍기: 콤보 2타째가 보이면 아무 버튼이나 누름 (1/3 확률로 맞음)
         2 반응: 2타째가 보이고 8~14프레임(약 130~240ms) 뒤에 그때 본 공격 버튼을 누름.
                 그 사이에 다른 공격으로 또 캔슬하면 틀려서 잠긴다.
    """
    MODES = (("끔", "OFF"), ("찍기", "GUESS"), ("반응", "REACT"))

    def __init__(self, seed=None):
        self.mode = 0
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.plan = None         # [남은 프레임, 누를 버튼]
        self.seen_chain = 0

    def __call__(self, world, me, presses, held):
        d, a = world.f[me], world.f[1 - me]
        if self.mode == 0:
            return Act.IDLE
        follow = a.state == St.ATTACK and a.chain >= 2
        if not follow:
            self.seen_chain = 0
        elif a.chain != self.seen_chain:
            self.seen_chain = a.chain
            if self.plan is None and d.state == St.HITSTUN and not d.break_locked:
                kind = a.kind if self.mode == 2 else self.rng.choice(C.CANCELABLE)
                self.plan = [self.rng.randint(8, 14), kind]
        if self.plan is not None:
            self.plan[0] -= 1
            if self.plan[0] <= 0:
                kind = self.plan[1]
                self.plan = None
                if d.state == St.HITSTUN:
                    return kind
        return Act.IDLE


# ====================================================================== 장면
class Scene:
    def __init__(self, app):
        self.app = app

    def handle(self, ev):
        pass

    def update(self):
        pass

    def draw(self, scr):
        pass


# ---------------------------------------------------------------------- 메뉴
class MenuScene(Scene):
    def __init__(self, app):
        super().__init__(app)
        self.world = World(110, 850, training=True)
        self.renderer = Renderer(app.screen, mode="menu")
        cx, w = SCREEN_W // 2, 340
        x = cx - w // 2
        ai_why = ai_unavailable(app.args.ai)
        half = (w - 10) // 2
        self.buttons = [
            ("training", ui.Button((x, 184, w, 62), ui.T("훈련장", "TRAINING"),
                                   ui.T("허수아비로 기술·콤보 데미지 확인", "test moves on a dummy"),
                                   key_hint="1")),
            ("match", ui.Button((x, 254, w, 62), ui.T("봇 대전", "VS BOT"),
                                ui.T("하위 · 중간 · 상위 봇 중에서 선택", "pick easy / normal / hard"),
                                accent=(255, 132, 112), key_hint="2")),
            ("ai", ui.Button((x, 324, w, 62), ui.T("AI 대전", "VS AI"),
                             ai_why or ui.T("강화학습으로 훈련한 AI · 반응 속도는 사람과 같음",
                                            "trained AI, human reaction time"),
                             enabled=ai_why is None, accent=AI_INFO[2], key_hint="3")),
            ("controls", ui.Button((x, 400, half, 40), ui.T("조작법", "CONTROLS"), size=17,
                                   accent=(140, 210, 190), key_hint="4")),
            ("quit", ui.Button((x + w - half, 400, half, 40), ui.T("종료", "QUIT"), size=17,
                               accent=(150, 154, 172))),
        ]
        self.focus = self._enabled()[0]     # 처음부터 첫 버튼이 선택돼 있다 → 바로 Enter 가능

    def _enabled(self):
        return [i for i, (_, b) in enumerate(self.buttons) if b.enabled]

    def _go(self, name):
        if name == "training":
            self.app.goto(TrainingScene(self.app))
        elif name == "match":
            self.app.goto(BotSelectScene(self.app))
        elif name == "ai":
            try:
                self.app.goto(MatchScene(self.app, "ai"))
            except RuntimeError as e:          # 못 불러오면 버튼을 끄고 이유를 보여 준다
                b = dict(self.buttons)["ai"]
                b.enabled, b.sub = False, str(e)
                self.focus = self._enabled()[0]
        elif name == "controls":
            self.app.goto(ControlsScene(self.app))
        elif name == "quit":
            self.app.running = False

    def handle(self, ev):
        self.focus = ui.mouse_focus(self.buttons, ev, self.focus)
        for name, b in self.buttons:
            if b.handle(ev):
                self._go(name)
                return
        if ev.type == pygame.KEYDOWN:
            en = self._enabled()
            if ev.key in (pygame.K_DOWN, pygame.K_s, pygame.K_UP, pygame.K_w):
                step = 1 if ev.key in (pygame.K_DOWN, pygame.K_s) else -1
                cur = en.index(self.focus) if self.focus in en else -1
                self.focus = en[(cur + step) % len(en)]
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and self.focus is not None:
                self._go(self.buttons[self.focus][0])
            elif ev.key == pygame.K_1:
                self._go("training")
            elif ev.key == pygame.K_2:
                self._go("match")
            elif ev.key == pygame.K_3 and dict(self.buttons)["ai"].enabled:
                self._go("ai")
            elif ev.key == pygame.K_4:
                self._go("controls")
            elif ev.key == pygame.K_ESCAPE:
                self.app.running = False

    def update(self):
        self.renderer.update(self.world, self.world.step(Act.IDLE, Act.IDLE))
        ui.sync_focus(self.buttons, self.focus)

    def draw(self, scr):
        self.renderer.draw(self.world)
        shade = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        shade.fill((10, 11, 16, 150))
        scr.blit(shade, (0, 0))
        ui.text(scr, "STICK FIGHT", (SCREEN_W // 2, 104), 64, ui.WHITE, True, "center")
        ui.text(scr, ui.T("1:1 졸라맨 대전", "1v1 stick figure fighting"),
                (SCREEN_W // 2, 152), 18, ui.DIM, anchor="center")
        for _, b in self.buttons:
            b.draw(scr)
        ui.text(scr, ui.T("마우스 클릭 · ↑/↓ + Enter", "click · arrows + Enter"),
                (SCREEN_W // 2, SCREEN_H - 22), 13, (110, 114, 132), anchor="center")


# ---------------------------------------------------------------------- 봇 난이도 선택
TIER_INFO = {
    "easy": (ui.T("하위 봇", "EASY"), ui.T("반응이 느리고 자주 머뭇거림 · 처음 해볼 때", "slow, hesitant"),
             (120, 210, 140)),
    "normal": (ui.T("중간 봇", "NORMAL"), ui.T("다가와서 때리고 콤보·패링·브레이커를 씀", "rushes in, combos"),
               (255, 196, 92)),
    "hard": (ui.T("상위 봇", "HARD"), ui.T("거리 싸움 · 헛친 공격 벌하기 · 습관 읽기", "spacing, punishes, reads"),
             (255, 112, 112)),
}

AI_INFO = (ui.T("AI", "AI"), ui.T("강화학습 AI", "trained AI"), (176, 140, 255))


class BotSelectScene(MenuScene):
    """봇 대전 → 하위 / 중간 / 상위 고르기. 메뉴와 같은 배경·조작 (클릭, ↑/↓ + Enter, 1·2·3, Esc 뒤로)."""

    def __init__(self, app):
        super().__init__(app)
        cx, w = SCREEN_W // 2, 340
        x = cx - w // 2
        self.buttons = []
        for i, tier in enumerate(TIERS):
            name, sub, col = TIER_INFO[tier]
            self.buttons.append((tier, ui.Button((x, 196 + 78 * i, w, 66), name, sub,
                                                 accent=col, key_hint=str(i + 1))))
        self.buttons.append(("back", ui.Button((x + 90, 436, w - 180, 40), ui.T("뒤로", "BACK"),
                                               size=17, accent=(150, 154, 172), key_hint="Esc")))
        self.focus = 0

    def _go(self, name):
        if name == "back":
            self.app.goto(MenuScene(self.app))
        else:
            self.app.goto(MatchScene(self.app, name))

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self._go(TIERS[ev.key - pygame.K_1])
            return
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self._go("back")
            return
        super().handle(ev)

    def draw(self, scr):
        self.renderer.draw(self.world)
        shade = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        shade.fill((10, 11, 16, 150))
        scr.blit(shade, (0, 0))
        ui.text(scr, ui.T("봇 대전", "VS BOT"), (SCREEN_W // 2, 104), 52, ui.WHITE, True, "center")
        ui.text(scr, ui.T("상대 난이도를 고르세요", "choose a difficulty"),
                (SCREEN_W // 2, 152), 18, ui.DIM, anchor="center")
        for _, b in self.buttons:
            b.draw(scr)
        ui.text(scr, ui.T("마우스 클릭 · ↑/↓ + Enter · 1/2/3 · Esc 뒤로", "click · arrows + Enter · 1/2/3 · Esc"),
                (SCREEN_W // 2, SCREEN_H - 22), 13, (110, 114, 132), anchor="center")


# ---------------------------------------------------------------------- 조작법
CONTROL_ROWS = (   # (동작, P1, P2)
    (("이동", "Move"), "A / D", "← / →"),
    (("점프", "Jump"), "W", "↑"),
    (("숙이기 (누르는 동안)", "Crouch (hold)"), "S", "↓"),
    (("대쉬", "Dash"), "Space", "R-Shift / Num0"),
    (("백대쉬", "Back dash"), ("뒤 + Space", "back+Space"), ("뒤 + R-Shift", "back+R-Shift")),
    (("상단 공격", "High"), "U", "Num 8"),
    (("정면 공격", "Mid"), "J", "Num 5"),
    (("하단 공격", "Low"), "M", "Num 2"),
    (("원거리", "Shot"), "K", "Num 6"),
    (("패링", "Parry"), "L", "Num 4"),
    (("강공격 (길게 눌렀다 떼기)", "Heavy (hold, release)"), "I", "Num 9"),
)
CONTROL_TIPS = (
    (("콤보", "Combo"), ("맞힌 직후 다른 공격 버튼(상단·정면·하단)으로 캔슬, 최대 4타",
                         "after a hit, cancel into another attack (max 4)")),
    (("대쉬 공격", "Dash attack"), ("앞 대쉬 중 정면 → 미끄러지며 찌름. 숙인 상대도 맞힘",
                                   "Mid while dashing. hits crouchers, punishable on whiff")),
    (("숙이기", "Crouch"), ("상단·정면·투사체를 피함. 하단·대쉬 공격·강공격은 맞음",
                           "dodges high, mid & shots. low/dash attack/heavy hit")),
    (("패링", "Parry"), ("맞기 직전에 누르면 상대가 크게 휘청임. 강공격은 못 받음",
                         "press just before a hit. can't parry heavy")),
    (("콤보 브레이커", "Combo breaker"), ("게이지 MAX일 때 콤보 2타째부터, 맞고 있는 공격과 같은 버튼",
                                         "gauge MAX, 2nd hit+: press the same attack")),
    (("강공격", "Heavy"), ("오래 모을수록 3단계까지 강해짐. 패링 불가, 헛치면 크게 반격당함",
                          "charge up to 3 stages. unparryable, punishable")),
)


def _t(v):
    return ui.T(*v) if isinstance(v, tuple) else v


class ControlsScene(MenuScene):
    """메뉴 → 조작법. P1/P2 키 표와 기술 설명. 메뉴와 같은 배경, Esc/Enter/클릭으로 뒤로."""

    def __init__(self, app, back=None):
        super().__init__(app)
        self.back = back             # 돌아갈 장면 (대전 중 일시정지 메뉴에서 열었으면 그 대전)
        self.buttons = [("back", ui.Button((SCREEN_W // 2 - 80, 482, 160, 36), ui.T("뒤로", "BACK"),
                                           size=16, accent=(150, 154, 172), key_hint="Esc"))]
        self.focus = 0

    def _go(self, name):
        self.app.goto(self.back or MenuScene(self.app))

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_KP_ENTER,
                                                     pygame.K_SPACE, pygame.K_BACKSPACE):
            self._go("back")
            return
        for _, b in self.buttons:
            if b.handle(ev):
                self._go("back")
                return

    def draw(self, scr):
        self.renderer.draw(self.world)
        shade = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        shade.fill((10, 11, 16, 190))
        scr.blit(shade, (0, 0))
        ui.text(scr, ui.T("조작법", "CONTROLS"), (SCREEN_W // 2, 44), 34, ui.WHITE, True, "center")
        # 왼쪽: 키 표
        ui.panel(scr, (28, 76, 452, 396))
        x0, x1, x2, y = 46, 236, 352, 90
        ui.text(scr, "P1", (x1, y), 15, (255, 140, 120), True)
        ui.text(scr, "P2", (x2, y), 15, (120, 180, 255), True)
        y += 28
        for k, (name, p1, p2) in enumerate(CONTROL_ROWS):
            if k % 2 == 0:
                band = pygame.Surface((436, 26), pygame.SRCALPHA)
                band.fill((255, 255, 255, 10))
                scr.blit(band, (36, y - 4))
            hot = name[0].startswith("숙이기") or name[1].startswith("Crouch")
            ui.text(scr, _t(name), (x0, y), 14, (190, 240, 210) if hot else ui.DIM)
            ui.text(scr, _t(p1), (x1, y), 14, ui.WHITE, True)
            ui.text(scr, _t(p2), (x2, y), 14, ui.WHITE, True)
            y += 28
        y += 6
        ui.text(scr, ui.T("공통: F1 판정박스 · F2 슬로우 · P 일시정지 · R 재시작 · Esc 메뉴",
                          "F1 hitbox · F2 slow · P pause · R restart · Esc menu"), (x0, y), 13, (130, 134, 152))
        # 오른쪽: 기술 설명
        ui.panel(scr, (492, 76, 440, 396))
        y = 92
        for title, desc in CONTROL_TIPS:
            ui.text(scr, _t(title), (510, y), 15, ui.ACCENT, True)
            ui.text(scr, _t(desc), (510, y + 22), 13, ui.DIM)
            y += 62
        for _, b in self.buttons:
            b.draw(scr)


# ---------------------------------------------------------------------- 대전 공통
class FightScene(Scene):
    """키 입력 → World.step → 렌더러. 훈련장과 봇 대전이 같이 쓴다."""

    renderer_mode = "match"

    def __init__(self, app, world, p1, p2):
        super().__init__(app)
        self.world = world
        self.p1, self.p2 = p1, p2
        self.renderer = Renderer(app.screen, mode=self.renderer_mode)
        self.debug, self.slow, self.paused = app.args.debug, False, False
        self.pending = set()
        self.tick = 0
        self.buttons = []           # (이름, Button)
        self.menu_open = False      # (v14) Esc 일시정지 메뉴
        T = ui.T
        x, w = SCREEN_W // 2 - 140, 280
        self.pause_buttons = [
            ("resume", ui.Button((x, 196, w, 44), T("계속", "RESUME"), size=19, key_hint="Esc")),
            ("reset", ui.Button((x, 248, w, 44), T("다시 시작", "RESTART"), size=19,
                                accent=(255, 196, 92), key_hint="R")),
            ("controls", ui.Button((x, 300, w, 44), T("조작법", "CONTROLS"), size=19,
                                   accent=(140, 210, 190))),
            ("menu", ui.Button((x, 352, w, 44), T("메뉴로", "MAIN MENU"), size=19,
                               accent=(150, 154, 172))),
        ]
        self.pause_focus = 0

    # 하위 클래스에서 구현
    def reset_round(self):
        self.world.reset()
        self.renderer.reset()

    def on_button(self, name):
        if name == "reset":
            self.reset_round()
        elif name == "hitbox":
            self.debug = not self.debug
        elif name == "menu":
            self.app.goto(MenuScene(self.app))

    # 일시정지 메뉴에 보이는 키 설명 (훈련장은 따로)
    PAUSE_HINT = (("P 일시정지 · F1 판정박스 · F2 슬로우(1/4)", "P pause · F1 hitbox · F2 slow"),)

    def _pause_pick(self, name):
        if name == "resume":
            self.menu_open = False
        elif name == "reset":
            self.menu_open = False
            self.on_button("reset")
        elif name == "controls":
            self.app.goto(ControlsScene(self.app, back=self))   # 뒤로 → 이 대전 (메뉴 열린 채)
        elif name == "menu":
            self.menu_open = False
            self.on_button("menu")

    def _pause_handle(self, ev):
        self.pause_focus = ui.mouse_focus(self.pause_buttons, ev, self.pause_focus)
        for name, b in self.pause_buttons:
            if b.handle(ev):
                self._pause_pick(name)
                return
        if ev.type != pygame.KEYDOWN:
            return
        if ev.key == pygame.K_ESCAPE:
            self.menu_open = False
        elif ev.key == pygame.K_r:
            self._pause_pick("reset")
        elif ev.key in (pygame.K_DOWN, pygame.K_s, pygame.K_UP, pygame.K_w):
            step = 1 if ev.key in (pygame.K_DOWN, pygame.K_s) else -1
            n = len(self.pause_buttons)
            self.pause_focus = 0 if self.pause_focus is None else (self.pause_focus + step) % n
        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and self.pause_focus is not None:
            self._pause_pick(self.pause_buttons[self.pause_focus][0])

    def draw_pause(self, scr):
        if not self.menu_open:
            return
        shade = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        shade.fill((6, 6, 12, 175))
        scr.blit(shade, (0, 0))
        hint_w = max(ui.font(13).size(ui.T(ko, en))[0] for ko, en in self.PAUSE_HINT)
        pw = max(340, hint_w + 40)                     # 키 설명이 길면(훈련장) 패널을 넓힌다
        ui.panel(scr, (SCREEN_W // 2 - pw // 2, 120, pw, 300 + 20 * len(self.PAUSE_HINT)), alpha=225)
        ui.text(scr, ui.T("일시정지", "PAUSED"), (SCREEN_W // 2, 156), 28, ui.WHITE, True, "center")
        ui.sync_focus(self.pause_buttons, self.pause_focus)
        for _, b in self.pause_buttons:
            b.draw(scr)
        for k, (ko, en) in enumerate(self.PAUSE_HINT):
            ui.text(scr, ui.T(ko, en), (SCREEN_W // 2, 414 + 20 * k), 13, (140, 144, 162), anchor="center")

    def after_step(self, events):
        pass

    def handle(self, ev):
        if self.menu_open:
            self._pause_handle(ev)
            return
        for name, b in self.buttons:
            if b.handle(ev):
                self.on_button(name)
                return
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.menu_open = True
                self.pause_focus = 0
                self.pending.clear()
            elif ev.key == pygame.K_F1:
                self.debug = not self.debug
            elif ev.key == pygame.K_F2:
                self.slow = not self.slow
            elif ev.key == pygame.K_p:
                self.paused = not self.paused
            elif ev.key == pygame.K_r:
                self.on_button("reset")
            else:
                self.pending.add(ev.key)

    def update(self):
        if self.menu_open:
            return
        held = pygame.key.get_pressed()
        self.tick += 1
        if not self.paused and (not self.slow or self.tick % 4 == 0):
            w = self.world
            a1 = self.p1(w, 0, self.pending, held)
            a2 = self.p2(w, 1, self.pending, held)
            self.pending.clear()
            events = w.step(a1, a2)
            self.after_step(events)
            self.renderer.update(w, events)

    def status_lines(self):
        out = []
        if self.slow:
            out.append("SLOW x0.25")
        if self.paused:
            out.append("PAUSED")
        return out

    def draw_buttons(self, scr):
        for name, b in self.buttons:
            if name == "hitbox":
                b.focus = self.debug
            b.draw(scr)


def _button_row(specs, right=SCREEN_W - 16, y=FLOOR + 14, h=38, gap=6):
    """오른쪽 정렬 버튼 줄. specs = [(이름, 글자, 단축키, 너비)]"""
    out = []
    x = right - sum(s[3] for s in specs) - gap * (len(specs) - 1)
    for name, label, key, w in specs:
        out.append((name, ui.Button((x, y, w, h), label, size=15, key_hint=key)))
        x += w + gap
    return out


# ---------------------------------------------------------------------- 훈련장
class TrainingScene(FightScene):
    renderer_mode = "training"
    PANEL = (SCREEN_W - 24 - 214, 94, 214, 204)     # 전체 기록창 (x, y, w, h)
    COMPACT_H = 30                                   # 간단 기록창 높이 (같은 y에서 시작)
    # 기록창: 간단(한 줄) → 전체 → 끔 순서로 바뀐다. 장면을 나갔다 와도 유지된다 (클래스 변수)
    PANEL_MODES = ("compact", "full", "off")
    PANEL_NAMES = {"compact": ("간단", "mini"), "full": ("전체", "full"), "off": ("끔", "off")}
    panel_mode = "compact"
    PAUSE_HINT = (("R 위치 초기화 · T 기록 초기화 · Tab 기록창 · F1 판정박스",
                   "R reset · T clear · Tab stats · F1 hitbox"),
                  ("B 허수아비 브레이커 · G 내 게이지 MAX(데미지 +15%)",
                   "B dummy breaker · G my gauge MAX (+15% dmg)"),
                  ("P 일시정지 · F2 슬로우(1/4)", "P pause · F2 slow"))

    def __init__(self, app):
        self.dummy = DummyBreaker()
        super().__init__(app, World(training=True), HumanController(P1_KEYS), self.dummy)
        self.meter = DamageMeter(attacker=0)
        self.gauge_max = False       # G: 내 게이지 항상 가득 (데미지 증가 확인용)
        self.breaks = [0, 0]         # 허수아비 브레이커 [성공, 실패]
        self.popups = []
        self.combo_banner = None     # (hits, dmg, life)
        self._was_live = False
        T = ui.T
        # (v14) 화면 아래 버튼 줄은 없앰 → 키(R/T/Tab/F1)와 Esc 일시정지 메뉴로.
        # 브레이커/게이지 버튼은 작게, 오른쪽 위 기록창 바로 아래에 붙는다 (_place_side_buttons)
        self.buttons = _button_row([
            ("breaker", "", "B", 122),
            ("gauge", T("게이지 MAX", "Gauge MAX"), "G", 112),
        ], right=SCREEN_W - 24, y=0, h=28)
        for name, b in self.buttons:
            if name in ("breaker", "gauge"):
                b.size = 13
        self._label_breaker()
        self._place_side_buttons()

    def _place_side_buttons(self):
        """기록창 모드에 따라 브레이커/게이지 버튼 높이를 옮긴다."""
        top = self.PANEL[1]
        y = {"full": top + self.PANEL[3] + 6, "compact": top + self.COMPACT_H + 6,
             "off": top}[self.panel_mode]
        for name, b in self.buttons:
            if name in ("breaker", "gauge"):
                b.rect.y = y

    def _label_breaker(self):
        ko, en = DummyBreaker.MODES[self.dummy.mode]
        b = dict(self.buttons)["breaker"]
        b.label = ui.T(f"브레이커 {ko}", f"Breaker {en}")

    def reset_round(self):
        super().reset_round()
        self.popups.clear()
        self.combo_banner = None
        self.dummy.reset()

    def on_button(self, name):
        if name == "panel":
            modes = self.PANEL_MODES
            TrainingScene.panel_mode = modes[(modes.index(self.panel_mode) + 1) % len(modes)]
            self._place_side_buttons()
        elif name == "clear":
            self.meter.reset()
            self.popups.clear()
            self.combo_banner = None
            self.breaks = [0, 0]
        elif name == "breaker":
            self.dummy.mode = (self.dummy.mode + 1) % len(DummyBreaker.MODES)
            self.dummy.reset()
            self._label_breaker()
            if self.dummy.mode == 0:
                self.world.f[1].break_gauge = 0.0
        elif name == "gauge":
            self.gauge_max = not self.gauge_max
            if not self.gauge_max:
                self.world.f[0].break_gauge = 0.0
        else:
            super().on_button(name)

    def handle(self, ev):
        keys = {pygame.K_t: "clear", pygame.K_b: "breaker", pygame.K_g: "gauge", pygame.K_TAB: "panel"}
        if ev.type == pygame.KEYDOWN and ev.key in keys:
            self.on_button(keys[ev.key])
            return
        super().handle(ev)

    def draw_buttons(self, scr):
        b = dict(self.buttons)
        b["breaker"].focus = self.dummy.mode != 0
        b["gauge"].focus = self.gauge_max
        super().draw_buttons(scr)

    def after_step(self, events):
        w, m = self.world, self.meter
        m.step(w, events)
        bonus = any(e[0] == "hit_bonus" and e[1] == 0 for e in events)
        for e in events:
            if e[0] == "hit" and e[1] == 0:
                _, att, dfn, kind, dealt, combo, pt = e
                heavy = kind == Act.HEAVY
                self.popups.append(dict(x=pt[0] + 14 * ((combo - 1) % 3 - 1), y=pt[1] + 24,
                                        dmg=dealt, t=0, heavy=heavy, combo=combo, bonus=bonus,
                                        drift=0.6 * (1 if (len(self.popups) % 2) else -1)))
            elif e[0] == "break" and e[1] == 1:
                self.breaks[0] += 1
            elif e[0] == "break_fail" and e[1] == 1:
                self.breaks[1] += 1
        # 게이지 유지: G를 켜면 내 게이지 가득, 허수아비 브레이커가 켜져 있으면 허수아비 게이지 가득
        if self.gauge_max:
            w.f[0].break_gauge = C.BREAKER.gauge_max
        if self.dummy.mode != 0 and w.f[1].state == St.IDLE:
            w.f[1].break_gauge = C.BREAKER.gauge_max
        # 콤보가 끝나는 순간 (2타 이상) 총합을 크게 보여준다
        if self._was_live and not m.combo_live and m.combo_hits >= 2:
            self.combo_banner = [m.combo_hits, m.combo_dmg, 80]
        self._was_live = m.combo_live
        for p in self.popups:
            p["t"] += 1
        self.popups = [p for p in self.popups if p["t"] < 55]
        if self.combo_banner:
            self.combo_banner[2] -= 1
            if self.combo_banner[2] <= 0:
                self.combo_banner = None
        # 한동안 안 맞으면 허수아비 체력이 다시 찬다
        d = w.f[1]
        if m.idle_frames() > REFILL_AFTER and d.state == St.IDLE and d.hp < C.MAX_HP:
            d.hp = C.MAX_HP

    def draw(self, scr):
        self.renderer.draw(self.world, debug=self.debug, overlay=self.status_lines())
        self._draw_popups(scr)
        if self.panel_mode == "full":
            self._draw_panel(scr)
        elif self.panel_mode == "compact":
            self._draw_compact(scr)
        self.draw_buttons(scr)
        self.draw_pause(scr)

    def _screen(self, x, y):
        r = self.renderer
        return (x + r.ox, FLOOR - y + r.oy)

    def _draw_popups(self, scr):
        for p in self.popups:
            t = p["t"]
            rise = 46 * (1 - (1 - min(1.0, t / 40)) ** 3)
            x, y = self._screen(p["x"] + p["drift"] * t, p["y"] + rise)
            alpha = 255 if t < 32 else int(255 * (1 - (t - 32) / 23))
            pop = 1.0 + 0.5 * max(0.0, 1 - t / 6)          # 처음 몇 프레임 크게 튀어나옴
            size = int((30 if p["heavy"] else 22 + min(p["combo"], 4) * 2) * pop)
            col = HEAVY_COL if p["heavy"] else (255, 240, 200) if p["combo"] > 1 else ui.WHITE
            if p.get("bonus"):
                col = BREAK_COL                         # 게이지 MAX 데미지 증가가 들어간 타격
            ui.text(scr, str(p["dmg"]), (x + 2, y + 2), size, (0, 0, 0), True, "center",
                    alpha=alpha // 2)
            ui.text(scr, str(p["dmg"]), (x, y), size, col, True, "center", alpha=alpha)
        if self.combo_banner:
            hits, dmg, life = self.combo_banner
            d = self.world.f[1]
            x, y = self._screen(d.x, d.y + 190)
            right = self.PANEL[0] - 130 if self.panel_mode == "full" else SCREEN_W - 150
            x = max(130, min(right, x))   # 오른쪽 기록창에 가리지 않게
            a = min(255, life * 8)
            ui.text(scr, f"{hits} HIT  {dmg}", (x, y), 30, (255, 214, 120), True, "center", alpha=a)

    def _draw_compact(self, scr):
        """간단 모드: DPS · 현재 콤보 · 마지막 타격만 한 줄로."""
        m = self.meter
        ko = ui.has_korean()
        sep = ("  ·  ", (90, 94, 112), 14, False)
        segs = [("DPS ", ui.DIM, 14, False),
                (f"{m.dps():.1f}", ui.WHITE if m.session_active else (170, 174, 192), 17, True)]
        if m.last:
            segs += [sep,
                     (f"{m.combo_hits} HIT {m.combo_dmg}",
                      (255, 214, 120) if m.combo_live else ui.WHITE, 15, True),
                     sep,
                     (kind_label(m.last, ko) + " ", ui.DIM, 14, False),
                     (str(m.last.dmg), ui.WHITE, 15, True)]
        imgs = [ui.font(sz, b).render(t, True, c) for t, c, sz, b in segs]
        pad, h = 12, self.COMPACT_H
        w = sum(i.get_width() for i in imgs) + 2 * pad
        x, y = SCREEN_W - 24 - w, self.PANEL[1]
        ui.panel(scr, (x, y, w, h), alpha=150, radius=8)
        cx = x + pad
        for img in imgs:
            scr.blit(img, (cx, y + h // 2 - img.get_height() // 2))
            cx += img.get_width()

    def _draw_panel(self, scr):
        m = self.meter
        ko = ui.has_korean()
        L = (lambda a, b: a if ko else b)
        x, y, w, h = self.PANEL
        ui.panel(scr, (x, y, w, h), alpha=170)
        pad = 12
        ui.text(scr, "DPS", (x + pad, y + 10), 13, ui.DIM, True)
        dps_col = ui.WHITE if m.session_active else (170, 174, 192)
        ui.text(scr, f"{m.dps():.1f}", (x + pad, y + 24), 32, dps_col, True)
        secs = m.session_frames() / C.FPS
        state = L("측정 중", "live") if m.session_active else (L("대기", "idle") if m.sess_start is None
                                                               else L("종료", "done"))
        ui.text(scr, f"{secs:.1f}s · {state}", (x + w - pad, y + 12), 12, ui.DIM, anchor="topright")
        ui.text(scr, f"{L('구간', 'sess')} {m.sess_dmg}", (x + w - pad, y + 30), 12, ui.DIM,
                anchor="topright")
        pygame.draw.line(scr, (54, 58, 78), (x + pad, y + 70), (x + w - pad, y + 70))

        last = f"{kind_label(m.last, ko)}  {m.last.dmg}" if m.last else "-"
        cur = f"{m.combo_hits} HIT · {m.combo_dmg}" if m.last else "-"
        best = f"{m.best_hits} HIT · {m.best_dmg}" if m.best_hits else "-"
        rows = ((L("마지막 타격", "Last hit"), last),
                (L("현재 콤보", "Combo"), cur),
                (L("최고 콤보", "Best"), best),
                (L("총 데미지", "Total"), f"{m.total} ({m.n_hits}{L('타', ' hits')})"),
                (L("브레이커", "Dummy breaks"),
                 f"{L('성공', 'ok')} {self.breaks[0]} · {L('실패', 'miss')} {self.breaks[1]}"))
        for k, (name, val) in enumerate(rows):
            ry = y + 80 + 24 * k
            ui.text(scr, name, (x + pad, ry), 13, ui.DIM)
            col = (255, 214, 120) if (k == 1 and m.combo_live) else ui.WHITE
            ui.text(scr, val, (x + w - pad, ry), 14, col, True, anchor="topright")


# ---------------------------------------------------------------------- 경기 기록 (v15)
class MatchStats:
    """승리 화면에 보여줄 경기 기록. 게임 이벤트만 센다."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.dmg = [0, 0]          # 준 데미지
        self.hits = [0, 0]         # 맞힌 횟수
        self.max_combo = [0, 0]    # 최대 콤보 (연속으로 맞힌 수)
        self.parries = [0, 0]      # 패링 성공 (근접 + 투사체)
        self.breaks = [0, 0]       # 콤보 브레이커 성공

    def update(self, events):
        for e in events:
            if e[0] == "hit":
                a = e[1]
                self.dmg[a] += e[4]
                self.hits[a] += 1
                self.max_combo[a] = max(self.max_combo[a], e[5])
            elif e[0] in ("parry", "parry_proj"):
                self.parries[e[1]] += 1
            elif e[0] == "break":
                self.breaks[e[1]] += 1


# ---------------------------------------------------------------------- 봇 대전
HUD_NAMES = {"ai": "AI", "easy": "EASY BOT", "normal": "NORMAL BOT", "hard": "HARD BOT", "human": "P2"}


class MatchScene(FightScene):
    def __init__(self, app, p2="normal"):
        world = World()
        self.p2_name = p2
        if p2 == "ai":
            bot = load_ai(app.args.ai)
        else:
            bot = None if p2 == "human" else BOTS[p2]()
        # P2(봇)가 P1을 보는 눈. 봇마다 반응 시간이 다를 수 있다 (하위 봇은 더 느림, 사람보다 빠른 봇은 없음)
        ms = getattr(bot, "reaction_ms", (130, 180, 240))
        self.eye = Perception(me=1, reaction=Reaction(*ms, rng=random.Random()))
        self.eye.reset(world)
        p2c = HumanController(P2_KEYS) if bot is None else BotController(bot, self.eye)
        super().__init__(app, world, HumanController(P1_KEYS), p2c)
        # (v14) 화면 아래 버튼·설명 없음. HUD 이름으로 상대를 보여 준다
        self.renderer.names = ["P1", HUD_NAMES.get(p2, p2.upper())]
        self.stats = MatchStats()
        self.end_t = 0              # (v15) 경기가 끝난 뒤 지난 프레임 → 슬로우모션 후 승리 화면

    def reset_round(self):
        super().reset_round()
        self.eye.reset(self.world)
        if isinstance(self.p2, BotController):
            self.p2.reset()
        self.stats.reset()
        self.end_t = 0

    def after_step(self, events):
        self.eye.push(self.world)
        self.stats.update(events)

    KO_SLOW = 75        # KO 뒤 이 프레임 동안 1/3 속도 (약 1.2초 → 실제 화면 시간)
    KO_END = 110        # KO 뒤 이만큼 지나면 승리 화면
    TIME_END = 75       # 시간 끝이면 슬로우 없이 이만큼 뒤

    def update(self):
        w = self.world
        if not w.done or self.menu_open:
            return super().update()
        self.end_t += 1
        ko = any(f.hp <= 0 for f in w.f)
        if not (ko and self.end_t < self.KO_SLOW) or self.end_t % 3 == 0:
            super().update()                       # 끝난 뒤에도 쓰러지는 연출은 물리로 진행
        if self.end_t >= (self.KO_END if ko else self.TIME_END):
            self.app.goto(VictoryScene(self.app, self))

    def on_button(self, name):
        if name == "menu" and self.p2_name in TIERS:
            self.app.goto(BotSelectScene(self.app))   # 봇 대전에서 나가면 난이도 선택으로
        else:
            super().on_button(name)

    def draw(self, scr):
        self.renderer.draw(self.world, debug=self.debug, overlay=self.status_lines())
        self.draw_pause(scr)


# ---------------------------------------------------------------------- 승리 화면 (v15)
CONFETTI_COLS = ((255, 214, 120), (240, 242, 248))


class VictoryScene(Scene):
    """경기가 끝나면: 승자 클로즈업 + 승리 모션 + 색종이, 오른쪽에 경기 기록, 재경기 / 상대 바꾸기 / 메뉴로."""

    ROWS = ((("준 데미지", "Damage"), "dmg"), (("맞힌 횟수", "Hits"), "hits"),
            (("최대 콤보", "Max combo"), "max_combo"), (("패링 성공", "Parries"), "parries"),
            (("콤보 브레이커", "Breakers"), "breaks"), (("남은 체력", "HP left"), "hp"))

    def __init__(self, app, match):
        super().__init__(app)
        self.match = match
        w = match.world
        self.winner = w.winner
        self.ko = any(f.hp <= 0 for f in w.f)
        self.secs = w.frame // C.FPS
        st = match.stats
        self.values = {"dmg": st.dmg, "hits": st.hits, "max_combo": st.max_combo,
                       "parries": st.parries, "breaks": st.breaks, "hp": [f.hp for f in w.f]}
        self.names = list(match.renderer.names)
        r = Renderer(app.screen, mode="match")
        r.names = self.names
        self.renderer = r
        self.t = 0
        self.rng = random.Random()
        self.confetti = []
        T = ui.T
        x, w_ = 600, 316
        self.buttons = [
            ("rematch", ui.Button((x, 380, w_, 40), T("재경기", "REMATCH"), size=17,
                                  accent=(255, 196, 92), key_hint="R")),
            ("change", ui.Button((x, 426, w_, 40), T("상대 바꾸기", "CHANGE OPPONENT"), size=17,
                                 accent=(255, 132, 112), key_hint="C")),
            ("menu", ui.Button((x, 472, w_, 40), T("메뉴로", "MAIN MENU"), size=17,
                               accent=(150, 154, 172), key_hint="Esc")),
        ]
        self.focus = 0

    def _go(self, name):
        if name == "rematch":
            self.match.reset_round()
            self.app.goto(self.match)
        elif name == "change":
            self.app.goto(BotSelectScene(self.app))
        else:
            self.app.goto(MenuScene(self.app))

    def handle(self, ev):
        self.focus = ui.mouse_focus(self.buttons, ev, self.focus)
        for name, b in self.buttons:
            if b.handle(ev):
                self._go(name)
                return
        if ev.type != pygame.KEYDOWN or self.t < 20:        # 넘어오자마자 누른 키는 무시
            return
        keys = {pygame.K_r: "rematch", pygame.K_c: "change", pygame.K_ESCAPE: "menu"}
        if ev.key in keys:
            self._go(keys[ev.key])
        elif ev.key in (pygame.K_DOWN, pygame.K_s, pygame.K_UP, pygame.K_w):
            step = 1 if ev.key in (pygame.K_DOWN, pygame.K_s) else -1
            self.focus = 0 if self.focus is None else (self.focus + step) % len(self.buttons)
        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) and self.focus is not None:
            self._go(self.buttons[self.focus][0])

    def update(self):
        self.t += 1
        if self.winner != -1 and self.t < 150:            # 색종이 (승자 색 + 금색 + 흰색)
            for _ in range(3 if self.t < 50 else 1):
                cols = (self.renderer.cols[self.winner],) + CONFETTI_COLS
                self.confetti.append([self.rng.uniform(0, 560), -10, self.rng.uniform(-0.6, 0.6),
                                      self.rng.uniform(1.4, 3.0), self.rng.uniform(0, 6.28),
                                      self.rng.choice(cols)])
        for c in self.confetti:
            c[0] += c[2] + math.sin(c[4]) * 0.6
            c[1] += c[3]
            c[4] += 0.15
        self.confetti = [c for c in self.confetti if c[1] < SCREEN_H + 10]

    def draw(self, scr):
        r = self.renderer
        r.draw_victory(self.winner, self.t, x=330)
        for x, y, _, _, ph, col in self.confetti:
            w = max(1, int(abs(math.cos(ph)) * 7))
            pygame.draw.rect(scr, col, (int(x), int(y), w, 4))
        # 큰 이름 / 결과
        if self.winner == -1:
            r._text("DRAW", 84, (255, 214, 120), (295, 70), "midtop", outline=4)
        else:
            r._text("K.O." if self.ko else "TIME", 26, (255, 214, 120), (32, 24), outline=2)
            r._text(self.names[self.winner], 84, r.cols[self.winner], (30, 48), outline=4)
            r._text("WINS", 40, ui.WHITE, (34, 152), outline=3)
        # 기록 패널
        px, py, pw, ph = 584, 36, 348, 336
        ui.panel(scr, (px, py, pw, ph), alpha=215)
        ui.text(scr, ui.T("경기 기록", "MATCH STATS"), (px + 18, py + 14), 18, ui.WHITE, True)
        ui.text(scr, ui.T(f"{self.secs}초", f"{self.secs}s"), (px + pw - 18, py + 16), 14, ui.DIM,
                anchor="topright")
        c1, c2 = px + 210, px + 300
        y = py + 54
        r._text(self.names[0], 15, r.cols[0], (c1, y), "midtop", outline=0)
        r._text(self.names[1], 15, r.cols[1], (c2, y), "midtop", outline=0)
        y += 30
        for k, ((ko, en), key) in enumerate(self.ROWS):
            v = self.values[key]
            if k % 2 == 0:
                band = pygame.Surface((pw - 20, 34), pygame.SRCALPHA)
                band.fill((255, 255, 255, 10))
                scr.blit(band, (px + 10, y - 7))
            ui.text(scr, ui.T(ko, en), (px + 22, y), 15, ui.DIM)
            for i, cx in ((0, c1), (1, c2)):
                best = v[i] > v[1 - i]
                col = (255, 214, 120) if best else ui.WHITE
                ui.text(scr, str(v[i]), (cx, y - 1), 17, col, best, "midtop")
            y += 40
        ui.sync_focus(self.buttons, self.focus)
        for _, b in self.buttons:
            b.draw(scr)
        if self.t < 24:                                    # 검은 화면에서 밝아짐
            fade = pygame.Surface((SCREEN_W, SCREEN_H))
            fade.set_alpha(int(255 * (1 - self.t / 24)))
            scr.blit(fade, (0, 0))


# ====================================================================== 앱
class App:
    def __init__(self, args):
        self.args = args
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.SCALED)
        pygame.display.set_caption("Stick Fight")
        self.clock = pygame.time.Clock()
        self.running = True
        start = {"menu": MenuScene, "training": TrainingScene,
                 "match": lambda a: MatchScene(a, args.p2)}[args.start]
        try:
            self.scene = start(self)
        except RuntimeError as e:          # --start match --p2 ai 인데 AI를 못 불러옴
            pygame.quit()
            raise SystemExit(f"AI를 불러올 수 없음: {e}")

    def goto(self, scene):
        self.scene = scene

    def frame(self):
        """한 프레임: 입력 → 갱신 → 그리기. 끝내야 하면 False."""
        scene = self.scene
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif self.scene is scene:      # 장면이 바뀐 프레임의 남은 입력은 버린다
                scene.handle(ev)
        if not self.running:
            return False
        self.scene.update()
        self.scene.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(C.FPS)
        return True

    def run(self):
        while self.frame():
            pass
        pygame.quit()

    async def run_async(self):
        """(v16) 브라우저(pygbag)용: 매 프레임 브라우저에 차례를 넘긴다 (main.py에서 씀)."""
        import asyncio
        while self.frame():
            await asyncio.sleep(0)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="menu", choices=["menu", "training", "match"])
    ap.add_argument("--p2", default="normal", choices=["human", "ai", *BOTS.keys()],
                    help="봇 대전의 P2")
    ap.add_argument("--ai", default=AI_PATH, help="AI 대전에 쓸 AI 파일 (tools/export_ai.py로 만든 것 또는 체크포인트)")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args(argv)


def main():
    App(parse_args()).run()


if __name__ == "__main__":
    main()
