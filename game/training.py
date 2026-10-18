"""훈련장 딜 측정기. 화면과 무관한 계산만 한다 (테스트 가능).

World.step()이 돌려준 이벤트를 매 프레임 넣어주면
마지막 타격 / 현재 콤보 / 최고 콤보 / 총 데미지 / DPS 를 계산한다.

DPS 구간: 첫 타격부터 시작해서, SESSION_GAP 프레임 동안 아무것도 안 맞으면 끝난다.
다음 타격부터 새 구간이 시작된다. 구간 중에는 시간이 계속 흐르므로 쉬면 DPS가 떨어진다.
"""
from . import config as C
from .config import Act

KIND_NAME = {Act.HIGH: ("상단", "HIGH"), Act.MID: ("정면", "MID"), Act.LOW: ("하단", "LOW"),
             Act.RANGED: ("원거리", "SHOT"), Act.HEAVY: ("강공격", "HEAVY"),
             C.DASH_ATK: ("대쉬 공격", "DASH ATK")}

SESSION_GAP = 3 * C.FPS      # 3초 동안 안 맞으면 DPS 구간 종료
MIN_SPAN = C.FPS             # 한 대만 쳤을 때 DPS가 무한대가 되지 않게 최소 1초로 계산
REFILL_AFTER = 90            # 허수아비가 이만큼 안 맞으면 체력이 다시 찬다


class Hit:
    __slots__ = ("dmg", "kind", "lv", "combo", "t")

    def __init__(self, dmg, kind, lv, combo, t):
        self.dmg, self.kind, self.lv, self.combo, self.t = dmg, kind, lv, combo, t


class DamageMeter:
    def __init__(self, attacker: int = 0):
        self.attacker = attacker
        self.reset()

    def reset(self):
        self.t = 0                # 측정기 시계 (화면 정지 프레임 포함, 실제 시간)
        self.last = None          # 마지막 타격 Hit
        self.combo_hits = 0
        self.combo_dmg = 0
        self.combo_live = False   # 상대가 아직 경직 중이라 콤보가 이어지는 중
        self.best_dmg = 0
        self.best_hits = 0
        self.total = 0
        self.n_hits = 0
        self.sess_start = None
        self.sess_last = None
        self.sess_dmg = 0
        self.history = []         # 최근 타격 (화면 기록용)

    # ------------------------------------------------------------ 입력
    def step(self, world, events):
        """World.step() 뒤에 매 프레임 호출."""
        self.t += 1
        for e in events:
            if e[0] == "hit" and e[1] == self.attacker:
                _, att, dfn, kind, dealt, combo, pt = e
                lv = world.f[att].heavy_lv if kind == Act.HEAVY else 0
                self._hit(Hit(dealt, kind, lv, combo, self.t))
        target = world.f[1 - self.attacker]
        self.combo_live = target.combo_taken > 0

    def _hit(self, h: Hit):
        if h.combo == 1:
            self.combo_hits = 0
            self.combo_dmg = 0
        self.combo_hits = h.combo
        self.combo_dmg += h.dmg
        if (self.combo_dmg, self.combo_hits) > (self.best_dmg, self.best_hits):
            self.best_dmg, self.best_hits = self.combo_dmg, self.combo_hits
        if self.sess_start is None or self.t - self.sess_last > SESSION_GAP:
            self.sess_start = self.t
            self.sess_dmg = 0
        self.sess_last = self.t
        self.sess_dmg += h.dmg
        self.total += h.dmg
        self.n_hits += 1
        self.last = h
        self.history.append(h)
        del self.history[:-6]

    # ------------------------------------------------------------ 조회
    @property
    def session_active(self) -> bool:
        return self.sess_start is not None and self.t - self.sess_last <= SESSION_GAP

    def session_frames(self) -> int:
        if self.sess_start is None:
            return 0
        end = self.t if self.session_active else self.sess_last
        return end - self.sess_start

    def dps(self) -> float:
        if self.sess_start is None:
            return 0.0
        return self.sess_dmg * C.FPS / max(MIN_SPAN, self.session_frames())

    def idle_frames(self) -> int:
        """마지막 타격 뒤 지난 프레임."""
        return self.t - self.last.t if self.last else 10 ** 9


def kind_label(h: Hit, korean: bool = True) -> str:
    ko, en = KIND_NAME.get(h.kind, ("?", "?"))
    name = ko if korean else en
    if h.kind == Act.HEAVY and h.lv:
        name += f" Lv{h.lv}"
    return name
