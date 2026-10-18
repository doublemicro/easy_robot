"""공격 시작 후 프레임별 실제 화면 (어떤 공격인지 몇 프레임째부터 알아볼 수 있는지 확인용).

    python tools/startup_strip.py   →  startup_strip.png
빨간 테두리 = 판정이 나오는 프레임.
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

N = 18
TW, TH = 120, 190


DASH_ATK_SCRIPT = {0: Act.DASH_F, 3: Act.MID}   # 대쉬 공격: 앞 대쉬 3프레임째에 정면


def strip(action):
    screen = pygame.Surface((SCREEN_W, SCREEN_H))
    r = Renderer(screen)
    w = World(400.0 if action != "DASH_ATK" else 250.0, 900.0)
    out = []
    for t in range(N):
        a = DASH_ATK_SCRIPT.get(t, Act.IDLE) if action == "DASH_ATK" else (action if t == 0 else Act.IDLE)
        ev = w.step(a, Act.IDLE)
        r.update(w, ev)
        r.draw(w)
        rect = pygame.Rect(int(w.f[0].x - TW / 2 + 12), FLOOR - TH + 16, TW, TH)
        rect.clamp_ip(screen.get_rect())
        out.append((screen.subsurface(rect).copy(), w.f[0].sf, w.f[0].attack_active()
                    if action != Act.RANGED else False))
    return out


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))
    font = pygame.font.Font(None, 20)
    rows = [("HIGH", Act.HIGH), ("MID", Act.MID), ("LOW", Act.LOW), ("RANGED", Act.RANGED),
            ("DASH ATK", "DASH_ATK")]
    lw = 80
    sheet = pygame.Surface((lw + TW * N, 20 + TH * len(rows)))
    sheet.fill((14, 15, 20))
    for k in range(N):
        sheet.blit(font.render(f"f{k} {k * 1000 // 60}ms", True, (170, 174, 190)), (lw + k * TW + 4, 4))
    for ri, (name, act) in enumerate(rows):
        y = 20 + ri * TH
        su = (C.ATTACKS[C.DASH_ATK].startup if act == "DASH_ATK" else
              C.ATTACKS[act].startup if act in C.ATTACKS else C.RANGED.startup)
        sheet.blit(font.render(name, True, (230, 232, 240)), (8, y + 8))
        sheet.blit(font.render(f"startup {su}", True, (150, 154, 172)), (8, y + 26))
        for k, (img, sf, active) in enumerate(strip(act)):
            x = lw + k * TW
            sheet.blit(img, (x, y))
            pygame.draw.rect(sheet, (230, 70, 70) if active else (14, 15, 20), (x, y, TW, TH), 2)
    sheet = pygame.transform.smoothscale(sheet, (int(sheet.get_width() * 1.25), int(sheet.get_height() * 1.25)))
    out = sys.argv[1] if len(sys.argv) > 1 else "startup_strip.png"
    pygame.image.save(sheet, out)
    print("saved", out)


if __name__ == "__main__":
    main()
