"""행동별 모션을 한 장의 이미지로 뽑는다 (모션 구분이 잘 되는지 확인용).

    python tools/pose_sheet.py   →  pose_sheet.png
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame

from game import config as C
from game.config import Act
from game.core import World
from game.render import FLOOR, SCREEN_H, SCREEN_W, Renderer

TILE_W, TILE_H = 200, 290


def capture(action, frames, x=400.0, target=None, setup=None, label="", hold=0):
    """P1이 action을 하고 지정한 프레임들에서 P1 주변을 잘라온다."""
    screen = pygame.Surface((SCREEN_W, SCREEN_H))
    r = Renderer(screen)
    w = World(x, target if target else 900.0)
    tiles = []
    want = set(frames)
    last = max(frames)
    for t in range(last + 1):
        a0 = action if (t == 0 or t < hold) else Act.IDLE
        a1 = setup(t) if setup else Act.IDLE
        ev = w.step(a0, a1)
        while w.hitstop > 0:          # 히트스톱은 건너뛰고 찍는다
            r.update(w, ev)
            ev = w.step(Act.IDLE, Act.IDLE)
        r.update(w, ev)
        if t in want:
            r.draw(w)
            fx = w.f[0].x
            rect = pygame.Rect(int(fx - TILE_W / 2 + 20), FLOOR - TILE_H + 30, TILE_W, TILE_H)
            tiles.append((screen.subsurface(rect).copy(), f"{label} f{w.f[0].sf}"))
    return tiles


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))
    font = pygame.font.Font(None, 22)
    big = pygame.font.Font(None, 26)
    rows = []
    for name, act in (("HIGH", Act.HIGH), ("MID", Act.MID), ("LOW", Act.LOW)):
        ad = C.ATTACKS[act]
        fr = [0, ad.startup // 2, ad.startup - 1, ad.startup + 1, ad.startup + ad.active + 4,
              ad.total - 4]
        rows.append((name, capture(act, fr)))
    # 기 모으기 0단계(빛 없음) → 1 → 2 → 3단계, 50프레임에 떼고 → 들어올리기 → 내려찍기
    rows.append(("HEAVY (hold)", capture(Act.HEAVY, [8, 16, 32, 48, 55, 61], hold=50)))
    r = C.RANGED
    rows.append(("RANGED", capture(Act.RANGED, [0, r.startup // 2, r.startup - 1, r.startup + 1,
                                                r.startup + 6, r.total - 3])))
    rows.append(("PARRY", capture(Act.PARRY, [0, 3, 8, 12, 18, 26])))
    rows.append(("DASH F/B", capture(Act.DASH_F, [2, 6, 13], x=300)
                 + capture(Act.DASH_B, [2, 6, 13], x=600)))
    rows.append(("JUMP", capture(Act.JUMP, [1, 6, 14, 20, 28, 35])))
    rows.append(("CROUCH (hold)", capture(Act.CROUCH, [0, 1, 2, 4, 12, 30], hold=31)))
    rows.append(("HIT/PARRIED", capture(Act.IDLE, [3, 12, 20], x=400, target=455,
                                        setup=lambda t: Act.MID if t == 0 else Act.IDLE)
                 + capture(Act.MID, [10, 20, 32], target=455,
                           setup=lambda t: Act.PARRY if t == 3 else Act.IDLE)))

    lw = 130
    sheet = pygame.Surface((lw + TILE_W * 6, TILE_H * len(rows)))
    sheet.fill((14, 15, 20))
    for ri, (name, tiles) in enumerate(rows):
        sheet.blit(big.render(name, True, (230, 232, 240)), (10, ri * TILE_H + 10))
        for ci, (img, lab) in enumerate(tiles):
            x, y = lw + ci * TILE_W, ri * TILE_H
            sheet.blit(img, (x, y))
            pygame.draw.rect(sheet, (14, 15, 20), (x, y, TILE_W, TILE_H), 1)
            sheet.blit(font.render(lab, True, (170, 174, 190)), (x + 6, y + 6))
    out = sys.argv[1] if len(sys.argv) > 1 else "pose_sheet.png"
    pygame.image.save(sheet, out)
    print("saved", out)


if __name__ == "__main__":
    main()
