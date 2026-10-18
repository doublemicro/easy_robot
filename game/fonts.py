"""(v16) 게임에 들어 있는 폰트 (assets/fonts, 모두 OFL 무료 폰트). PC와 웹(pygbag)에서 똑같이 보이게 시스템 폰트 대신 쓴다.

- 제목·HUD 숫자: Anton (Impact와 비슷한 굵은 세로 글꼴)
- 한글·작은 글자: 나눔고딕 (tools/subset_fonts.py로 게임에 쓰는 글자만 남김)
파일이 없으면 시스템 폰트로 대신한다.
"""
import os

import pygame

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")


def _path(name):
    p = os.path.join(FONT_DIR, name)
    return p if os.path.exists(p) else None


DISPLAY = _path("Anton-Regular.ttf")
KOREAN = _path("NanumGothic-Regular.ttf")
KOREAN_BOLD = _path("NanumGothic-Bold.ttf")
_cache = {}


def display(size):
    """굵은 제목 글꼴 (HUD 숫자, FIGHT!, K.O. ...)."""
    key = ("display", size)
    if key not in _cache:
        _cache[key] = pygame.font.Font(DISPLAY, size) if DISPLAY else \
            pygame.font.SysFont("impact,arialblack,arial", size)
    return _cache[key]


def korean(size, bold=False):
    key = ("korean", size, bold)
    if key not in _cache:
        path = (KOREAN_BOLD or KOREAN) if bold else KOREAN
        if path:
            _cache[key] = pygame.font.Font(path, size)
        else:
            f = pygame.font.SysFont("malgungothic,applesdgothicneo,notosanscjkkr,nanumgothic", size)
            f.set_bold(bold)
            _cache[key] = f
    return _cache[key]
