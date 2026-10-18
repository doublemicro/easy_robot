"""(v14) 야간 경기장 배경 (렌더링 전용, 게임 규칙과 무관).

한 번만 그려 두는 것 (make_background):
  천장 조명 트러스 → 스포트라이트 원뿔 → 관중석 실루엣 여러 줄 → LED 광고판 → 뒤쪽 링 로프·코너 기둥
  → 링 매트와 앞 에이프런 → 매트 위 조명 웅덩이
매 프레임 따로 그리는 것 (render.py): 선수 발밑 조명(make_glow), 관중 카메라 플래시, 화면 가장자리 어둡게(make_vignette).

벽(WALL_MARGIN) 위치에 코너 기둥이 서서, 벽꽝이 로프에 부딪히는 것처럼 보인다.
캐릭터가 잘 보이도록 배경은 전부 어둡고 채도를 낮게 둔다.
"""
import math
import random

import pygame
from pygame import gfxdraw

from . import config as C
from . import fonts

SCREEN_W, SCREEN_H = 960, 540
FLOOR = 468

SKY_TOP = (6, 7, 13)
SKY_BOT = (20, 18, 30)
MAT_TOP = (36, 40, 62)
MAT_BOT = (24, 27, 42)
APRON = (11, 12, 19)
ROPE = (112, 112, 132)
POST = (64, 68, 86)
CORNER_COLS = ((70, 130, 220), (220, 84, 70))      # 왼쪽 = P1 쪽, 오른쪽 = P2 쪽
LED = (70, 110, 200)
ROPE_H = (36, 66, 96)                              # 바닥에서 로프 높이
SPOTS = ((205, 330), (480, 480), (755, 630))       # 조명 x → 비추는 바닥 x
CROWD_Y = (176, FLOOR - 62)                        # 관중이 보이는 화면 y 범위 (플래시용)
BG_MID = (22, 22, 34)                              # 선수 뒤쪽 평균 색 (잔상·효과가 사라질 때 섞는 색)


def _mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def _add_cone(surf, ss, top_x, top_y, bot_x, bot_y, half_w, layers=7, col=(7, 7, 10)):
    """밝기를 더하는 빛 원뿔 (겹쳐 그려서 가운데가 더 밝고 가장자리는 부드럽게)."""
    light = pygame.Surface(surf.get_size())
    for k in range(layers):
        f = 1.0 - k / layers
        tw, bw = 9 * f + 3, half_w * f
        pts = [((top_x - tw) * ss, top_y * ss), ((top_x + tw) * ss, top_y * ss),
               ((bot_x + bw) * ss, bot_y * ss), ((bot_x - bw) * ss, bot_y * ss)]
        layer = pygame.Surface(surf.get_size())
        pygame.draw.polygon(layer, col, pts)
        light.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)
    surf.blit(light, (0, 0), special_flags=pygame.BLEND_ADD)


def _add_ellipse_glow(surf, ss, cx, cy, rx, ry, col=(9, 9, 12), layers=8):
    light = pygame.Surface(surf.get_size())
    for k in range(layers):
        f = 1.0 - k / layers
        r = pygame.Rect(0, 0, int(2 * rx * f * ss), int(2 * ry * f * ss))
        r.center = (int(cx * ss), int(cy * ss))
        layer = pygame.Surface(surf.get_size())
        pygame.draw.ellipse(layer, col, r)
        light.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)
    surf.blit(light, (0, 0), special_flags=pygame.BLEND_ADD)


def _crowd(bg, ss):
    rng = random.Random(7)
    for r in range(7):                               # 뒷줄 → 앞줄
        y = 186 + r * 27
        s = 0.78 + r * 0.09
        base = (15 + r * 3, 15 + r * 3, 25 + r * 4)
        x = -10 + rng.uniform(0, 10)
        while x < SCREEN_W + 20:
            px = x + rng.uniform(-3, 3)
            py = y + rng.uniform(-3, 3)
            c = _mix(base, (40, 40, 58), rng.uniform(0, 0.25))
            # 몸 (어깨부터 아래로)
            body = pygame.Rect(0, 0, int(21 * s * ss), int(40 * s * ss))
            body.midtop = (int(px * ss), int((py - 4 * s) * ss))
            pygame.draw.ellipse(bg, c, body)
            # 팔을 든 사람 (환호)
            if rng.random() < 0.07:
                side = rng.choice((-1, 1))
                a = (px + side * 8 * s, py)
                b = (px + side * 13 * s, py - 24 * s)
                pygame.draw.line(bg, c, (a[0] * ss, a[1] * ss), (b[0] * ss, b[1] * ss), int(5 * s * ss))
                gfxdraw.filled_circle(bg, int(b[0] * ss), int(b[1] * ss), int(3.4 * s * ss), c)
            # 머리 + 위쪽 테두리 빛 (조명이 위에서 비춤)
            hx, hy, hr = int(px * ss), int((py - 12 * s) * ss), int(7 * s * ss)
            gfxdraw.filled_circle(bg, hx, hy - int(ss), hr, _mix(c, (70, 74, 100), 0.35))
            gfxdraw.filled_circle(bg, hx, hy, hr, c)
            x += 17 * s + rng.uniform(-2, 4)


def make_background(ss):
    W, H = SCREEN_W * ss, SCREEN_H * ss
    bg = pygame.Surface((W, H))
    fy = FLOOR * ss
    # 하늘(경기장 안쪽 어둠)
    for y in range(fy):
        pygame.draw.line(bg, _mix(SKY_TOP, SKY_BOT, (y / fy) ** 1.4), (0, y), (W, y))
    # 관중석
    _crowd(bg, ss)
    # 관중 위쪽을 살짝 어둡게 (멀리 있는 줄은 어둠에 묻힘)
    fade = pygame.Surface((W, int(220 * ss)), pygame.SRCALPHA)
    for y in range(fade.get_height()):
        a = int(170 * (1 - y / fade.get_height()) ** 1.6)
        pygame.draw.line(fade, (*SKY_TOP, a), (0, y), (W, y))
    bg.blit(fade, (0, int(150 * ss)))
    # 천장 트러스 + 조명
    pygame.draw.rect(bg, (12, 12, 20), (0, 0, W, int(16 * ss)))
    for x in range(0, SCREEN_W + 24, 24):
        pygame.draw.line(bg, (30, 31, 46), (x * ss, 0), ((x + 24) * ss, 16 * ss), ss)
        pygame.draw.line(bg, (30, 31, 46), ((x + 24) * ss, 0), (x * ss, 16 * ss), ss)
    pygame.draw.line(bg, (40, 42, 60), (0, 16 * ss), (W, 16 * ss), 2 * ss)
    for sx, tx in SPOTS:
        _add_cone(bg, ss, sx, 18, tx, FLOOR + 20, 150)
    for sx, _ in SPOTS:
        pygame.draw.rect(bg, (28, 30, 42), ((sx - 14) * ss, 10 * ss, 28 * ss, 12 * ss), border_radius=3 * ss)
        gfxdraw.filled_circle(bg, sx * ss, 22 * ss, 6 * ss, (255, 250, 230))
        _add_ellipse_glow(bg, ss, sx, 22, 30, 14, col=(14, 13, 10), layers=6)
    # LED 광고판 (관중석 앞 벽)
    top = FLOOR - 62
    pygame.draw.rect(bg, (13, 14, 22), (0, top * ss, W, (FLOOR - top) * ss))
    pygame.draw.line(bg, LED, (0, top * ss), (W, top * ss), 2 * ss)
    _add_ellipse_glow(bg, ss, SCREEN_W / 2, top, SCREEN_W * 0.7, 6, col=(5, 7, 14), layers=5)
    font = fonts.display(30 * ss)
    word = font.render("STICK FIGHT  ·  NIGHT ARENA  ·  ", True, (24, 28, 48))
    for k in range(-1, 4):
        bg.blit(word, (k * word.get_width() + 40 * ss, (top + 14) * ss))
    # 링 매트
    for y in range(fy, int((FLOOR + 46) * ss)):
        t = (y - fy) / (46 * ss)
        pygame.draw.line(bg, _mix(MAT_TOP, MAT_BOT, t), (0, y), (W, y))
    pygame.draw.ellipse(bg, (46, 52, 80), (int(330 * ss), int((FLOOR + 8) * ss), int(300 * ss), int(30 * ss)), 2 * ss)
    logo = fonts.display(18 * ss).render("SF", True, (46, 52, 80))
    bg.blit(logo, logo.get_rect(center=(SCREEN_W * ss // 2, int((FLOOR + 23) * ss))))
    for sx, tx in SPOTS:                              # 조명이 닿는 매트
        _add_ellipse_glow(bg, ss, tx, FLOOR + 20, 150, 20, col=(6, 6, 9))
    pygame.draw.line(bg, (96, 104, 148), (0, fy), (W, fy), 2 * ss)
    # 앞 에이프런 (링 옆면)
    ay = int((FLOOR + 46) * ss)
    pygame.draw.rect(bg, APRON, (0, ay, W, H - ay))
    pygame.draw.line(bg, (150, 40, 58), (0, ay), (W, ay), 2 * ss)
    for x in range(0, SCREEN_W, 96):
        pygame.draw.line(bg, (18, 19, 30), (x * ss, ay + 4 * ss), (x * ss, H), ss)
    # 뒤쪽 로프 + 코너 기둥 (벽 위치)
    wm = C.WALL_MARGIN
    posts = (wm - 8, SCREEN_W - wm + 8)
    for h in ROPE_H:
        y = (FLOOR - h) * ss
        pygame.draw.line(bg, (40, 40, 52), (posts[0] * ss, y + 2 * ss), (posts[1] * ss, y + 2 * ss), 2 * ss)
        pygame.draw.line(bg, ROPE, (posts[0] * ss, y), (posts[1] * ss, y), 3 * ss)
        pygame.draw.line(bg, (170, 170, 190), (posts[0] * ss, y - ss), (posts[1] * ss, y - ss), ss)
    for k, px in enumerate(posts):
        r = pygame.Rect(0, 0, 10 * ss, (ROPE_H[-1] + 60) * ss)
        r.midbottom = (px * ss, (FLOOR + 40) * ss)
        pygame.draw.rect(bg, POST, r, border_radius=2 * ss)
        pygame.draw.line(bg, (110, 116, 140), (r.x + 2 * ss, r.y), (r.x + 2 * ss, r.bottom), ss)
        for h in ROPE_H:
            pad = pygame.Rect(0, 0, 16 * ss, 14 * ss)
            pad.center = (px * ss, (FLOOR - h) * ss)
            pygame.draw.rect(bg, CORNER_COLS[k], pad, border_radius=4 * ss)
            pygame.draw.rect(bg, _mix(CORNER_COLS[k], (255, 255, 255), 0.35),
                             (pad.x + 3 * ss, pad.y + 2 * ss, 4 * ss, pad.h - 4 * ss), border_radius=2 * ss)
    return bg


def make_glow(ss, rx=70, ry=12, col=(10, 10, 12), layers=7):
    """선수 발밑 조명 (BLEND_ADD로 그린다)."""
    w, h = int(2 * rx * ss), int(2 * ry * ss)
    s = pygame.Surface((w, h))
    for k in range(layers):
        f = 1.0 - k / layers
        r = pygame.Rect(0, 0, int(w * f), int(h * f))
        r.center = (w // 2, h // 2)
        layer = pygame.Surface((w, h))
        pygame.draw.ellipse(layer, col, r)
        s.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)
    return s


def make_vignette():
    """화면 가장자리를 어둡게 (화면 해상도, 알파)."""
    import numpy as np
    v = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    v.fill((0, 0, 0, 255))
    xs = (np.arange(SCREEN_W) - SCREEN_W / 2) / (SCREEN_W / 2)
    ys = (np.arange(SCREEN_H) - SCREEN_H * 0.5) / (SCREEN_H / 2)
    d = np.sqrt(xs[:, None] ** 2 + (ys[None, :] * 0.9) ** 2)
    a = np.clip((d - 0.62) / 0.75, 0, 1) ** 1.5 * 175
    alpha = pygame.surfarray.pixels_alpha(v)
    alpha[:] = a.astype(np.uint8)
    del alpha
    return v


def flash_spot(rng):
    """관중석 어딘가 (카메라 플래시 위치)."""
    return rng.uniform(30, SCREEN_W - 30), rng.uniform(CROWD_Y[0], CROWD_Y[1] - 10)


def draw_flash(canvas, ss, x, y, life, max_life=7):
    t = life / max_life
    r = (1.5 + 3.5 * t) * ss
    c = (int(255 * t), int(255 * t), int(240 * t))
    gfxdraw.filled_circle(canvas, int(x * ss), int(y * ss), max(1, int(r)), c)
    L = (4 + 10 * t) * ss
    for ang in (0, math.pi / 2):
        dx, dy = math.cos(ang) * L, math.sin(ang) * L
        pygame.draw.line(canvas, c, (x * ss - dx, y * ss - dy), (x * ss + dx, y * ss + dy), max(1, ss))
