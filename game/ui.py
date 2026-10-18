"""메뉴/훈련장용 화면 부품: 한글 폰트, 마우스로 누르는 버튼, 반투명 패널.

게임 규칙과는 무관하고 pygame 화면에만 쓴다.
"""
from __future__ import annotations

import pygame

from . import fonts

WHITE = (240, 242, 248)
DIM = (150, 154, 172)
ACCENT = (118, 196, 255)
PANEL = (16, 18, 26)

# Windows: 맑은 고딕, macOS: Apple SD Gothic Neo, Linux: Noto Sans CJK / 나눔고딕
_KO_FONTS = ("malgungothic", "malgungothicsemilight", "applesdgothicneo", "applegothic",
             "notosanscjkkr", "notosanscjk", "nanumgothic", "nanumbarungothic", "gulim")
_cache = {}
_ko_path = None
_ko_checked = False


def _korean_font_path():
    global _ko_path, _ko_checked
    if not _ko_checked and fonts.KOREAN:       # (v16) 게임에 들어 있는 나눔고딕이 먼저
        _ko_checked, _ko_path = True, fonts.KOREAN
    if not _ko_checked:
        _ko_checked = True
        for name in _KO_FONTS:
            path = pygame.font.match_font(name)
            if path:
                _ko_path = path
                break
    return _ko_path


def has_korean() -> bool:
    return _korean_font_path() is not None


def T(ko: str, en: str) -> str:
    """한글 폰트가 있으면 한글, 없으면 영어 (글자가 네모로 깨지는 것 방지)."""
    return ko if has_korean() else en


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _cache and fonts.KOREAN:
        _cache[key] = fonts.korean(size, bold)
    if key not in _cache:
        path = _korean_font_path()
        if path:
            f = pygame.font.Font(path, size)
        else:
            f = pygame.font.Font(None, int(size * 1.3))
        f.set_bold(bold)
        _cache[key] = f
    return _cache[key]


def text(surf, s, pos, size=18, col=WHITE, bold=False, anchor="topleft", alpha=None):
    img = font(size, bold).render(s, True, col)
    if alpha is not None:
        img.set_alpha(alpha)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def panel(surf, rect, alpha=200, radius=10, border=(54, 58, 78)):
    r = pygame.Rect(rect)
    s = pygame.Surface(r.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*PANEL, alpha), s.get_rect(), border_radius=radius)
    if border:
        pygame.draw.rect(s, (*border, 255), s.get_rect(), 1, border_radius=radius)
    surf.blit(s, r)


class Button:
    """마우스 버튼. 누른 채로 버튼 안에서 떼야 클릭으로 친다.

    handle(event)가 True를 돌려주면 클릭된 것.
    """

    def __init__(self, rect, label, sub=None, enabled=True, size=22, accent=ACCENT, key_hint=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.sub = sub              # 버튼 아래쪽 작은 설명
        self.enabled = enabled
        self.size = size
        self.accent = accent
        self.key_hint = key_hint    # 오른쪽에 작게 표시할 단축키
        self.hover = False
        self.focus = False          # 키보드로 선택된 상태 (메뉴)
        self._down = False
        self._glow = 0.0

    def handle(self, ev) -> bool:
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._down = self.enabled and self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            hit = self._down and self.enabled and self.rect.collidepoint(ev.pos)
            self._down = False
            return hit
        return False

    def draw(self, surf):
        lit = self.enabled and (self.hover or self.focus)
        self._glow += ((1.0 if lit else 0.0) - self._glow) * 0.3
        g = self._glow
        r = self.rect.move(0, 1 if self._down and self.hover else 0)
        base = (30, 33, 46) if self.enabled else (24, 26, 34)
        fill = tuple(int(b + (a * 0.22 - b * 0.0) * g) for a, b in zip(self.accent, base))
        pygame.draw.rect(surf, fill, r, border_radius=8)
        edge = tuple(int(54 + (a - 54) * g) for a in self.accent) if self.enabled else (44, 46, 58)
        pygame.draw.rect(surf, edge, r, 2 if g > 0.5 else 1, border_radius=8)
        if g > 0.05:   # 왼쪽 강조 막대
            bar = pygame.Rect(r.x + 6, r.y + 8, 3, r.h - 16)
            pygame.draw.rect(surf, self.accent, bar, border_radius=2)
        col = WHITE if self.enabled else (96, 100, 116)
        if self.sub:
            text(surf, self.label, (r.centerx, r.centery - 9), self.size, col, True, "center")
            text(surf, self.sub, (r.centerx, r.centery + 15), max(12, self.size - 9),
                 DIM if self.enabled else (80, 84, 98), anchor="center")
        else:   # 단축키 표시가 있으면 글자를 살짝 왼쪽으로
            cx = r.centerx - (9 if self.key_hint else 0)
            text(surf, self.label, (cx, r.centery), self.size, col, True, "center")
        if self.key_hint:
            text(surf, self.key_hint, (r.right - 10, r.centery), 13,
                 (110, 114, 132), anchor="midright")
