"""강공격 사거리·대응 측정 (v8: 휘두르며 한 발 내딛기).

    python -m tools.heavy_reach
"""
from game import config as C
from game.config import Act, St
from game.core import World

I = Act.IDLE


def step_distance():
    w = World(300, 900)
    x0 = w.f[0].x
    out = {}
    ad = C.ATTACKS[Act.HEAVY]
    for t in range(80):
        w.step(Act.HEAVY if t < 14 else I, I)
        f = w.f[0]
        if f.state == St.ATTACK and f.sf == ad.startup + ad.active:
            out["active_end"] = f.x - x0
    out["total"] = w.f[0].x - x0
    return out


def hits_at(dist, held=14):
    w = World(300, 300 + dist)
    for t in range(held + 50):
        w.step(Act.HEAVY if t < held else I, I)
    return w.f[1].hp < 100


def counter(dist, held, react, plan):
    """P1이 held 프레임 차지 후 강공격. P2는 차지 시작 react 프레임 뒤부터 plan대로 대응.
    결과: 'P2 승'(P2가 먼저 맞힘) / '피함'(아무도 안 맞음) / '맞음'"""
    w = World(300, 300 + dist)
    for t in range(held + 60):
        a1 = Act.HEAVY if t < held else I
        a2 = I
        if t >= react:
            f2, f1 = w.f[1], w.f[0]
            d = abs(f1.x - f2.x)
            if plan == "walk_mid":
                a2 = Act.MID if d < 80 else Act.LEFT
            elif plan == "backdash":
                a2 = Act.DASH_B
            elif plan == "back_walk":
                a2 = Act.RIGHT
            elif plan == "jump":
                a2 = Act.JUMP
            elif plan == "low":
                a2 = Act.LOW if d < 94 else Act.LEFT
        w.step(a1, a2)
    h0, h1 = w.f[0].hp < 100, w.f[1].hp < 100
    if h0 and not h1:
        return "끊음"
    if h1 and not h0:
        return "맞음"
    return "피함" if not h0 else "둘다"


def main():
    sd = step_distance()
    ad = C.ATTACKS[Act.HEAVY]
    print(f"내딛는 거리: 판정 끝까지 {sd['active_end']:.0f}px, 후딜까지 합쳐 {sd['total']:.0f}px")
    print(f"히트박스 앞 끝 {ad.box[1]:.0f} → 서 있을 때 사거리 {ad.box[1] + C.BODY_W / 2:.0f}px")
    far = max(d for d in range(40, 260, 2) if hits_at(d))
    print(f"서 있는 상대에게 닿는 최대 거리(시작 거리): {far}px  (v7: 93px, 정면 81, 하단 95)")
    print("\n대응 (가장 빠른 차지 14 / 2단계 28 / 3단계 46), 반응 = 차지 시작을 보고 11프레임(최빈) 뒤")
    plans = (("걸어 들어가 정면", "walk_mid"), ("걸어 들어가 하단", "low"),
             ("백대쉬", "backdash"), ("뒤로 걷기", "back_walk"), ("점프", "jump"))
    for dist in (60, 90, 120, 150):
        print(f" 거리 {dist}px")
        for name, plan in plans:
            res = [counter(dist, held, 11, plan) for held in (14, 28, 46)]
            print(f"   {name:12s} " + "  ".join(f"{h}:{r}" for h, r in zip((14, 28, 46), res)))


if __name__ == "__main__":
    main()
