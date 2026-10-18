"""규칙 테스트. 실행: python -m tests.test_rules  (stickfight 폴더에서)"""
import random
import time

from game import config as C
from game.config import Act, St
from game.core import World
from ai.env import StickFightEnv
from ai.bots import AggroBot, EasyBot, HardBot, NormalBot, RandomBot
from ai.perception import Perception, Reaction, SeenWorld, pick_delay
from ai.obs import build_obs

I = Act.IDLE


def run(world, script0, script1, frames):
    """script: {frame: action}. 지정 안 된 프레임은 IDLE."""
    log = []
    for t in range(frames):
        log += world.step(script0.get(t, I), script1.get(t, I))
    return log


def hits(log):
    return [e for e in log if e[0] == "hit"]


def test_mid_hits_standing():
    w = World(400, 455)
    run(w, {0: Act.MID}, {}, 30)
    assert w.f[1].hp < 100, w.f[1].hp


def test_jump_avoids_low():
    w = World(400, 455)
    run(w, {0: Act.LOW}, {0: Act.JUMP}, 30)
    assert w.f[1].hp == 100


def test_high_hits_jumper():
    w = World(400, 440)
    run(w, {0: Act.HIGH}, {4: Act.JUMP}, 30)
    assert w.f[1].hp < 100


def test_low_ducks_under_high():
    # P1의 상단이 판정 나는 시점에 P2는 하단 자세(낮은 피격 범위)
    w = World(400, 450)
    log = run(w, {0: Act.HIGH}, {6: Act.LOW}, 40)
    hs = hits(log)
    assert hs and all(h[1] == 1 for h in hs), hs   # P2만 때렸다


def test_parry():
    w = World(400, 455)
    log = run(w, {0: Act.MID}, {3: Act.PARRY}, 12)
    assert any(e[0] == "parry" for e in log), log
    assert w.f[0].state == St.PARRIED and w.f[1].hp == 100


def test_parry_whiff_is_punishable():
    w = World(400, 455)
    log = run(w, {12: Act.MID}, {0: Act.PARRY}, 30)
    assert w.f[1].hp < 100


def test_combo_three_hits():
    w = World(400, 455)
    log = run(w, {0: Act.MID, 9: Act.LOW, 22: Act.HIGH}, {}, 70)
    hs = hits(log)
    assert len(hs) == 3, hs
    assert [h[5] for h in hs] == [1, 2, 3], hs   # 연속 콤보로 인정됨


def test_combo_needs_hit():
    # 헛친 공격은 캔슬되지 않는다
    w = World(200, 700)
    run(w, {0: Act.MID, 9: Act.LOW}, {}, 12)
    assert w.f[0].state == St.ATTACK and w.f[0].kind == Act.MID


def _vs_projectile(answer, at=60):
    w = World(300, 700)
    # P2가 원거리 발사 → 투사체가 다가올 때 P1이 answer
    for t in range(80):
        p = w.proj[0] if w.proj else None
        a0 = answer if (p and abs(p.x - w.f[0].x) < at and w.f[0].state == St.IDLE
                        and w.f[0].grounded) else I
        w.step(a0, Act.RANGED if t == 0 else I)
    return w.f[0].hp


def test_projectile_answers():
    # v6: 대쉬 무적이 없어져서 대쉬로는 투사체를 못 뚫는다. 점프·하단 자세로 피한다
    assert _vs_projectile(Act.DASH_F) < 100
    assert _vs_projectile(Act.JUMP, at=110) == 100
    assert _vs_projectile(Act.LOW) == 100


def test_parry_projectile():
    w = World(300, 700)
    log = []
    for t in range(80):
        p = w.proj[0] if w.proj else None
        a0 = Act.PARRY if (p and abs(p.x - w.f[0].x) < 60 and w.f[0].state == St.IDLE) else I
        log += w.step(a0, Act.RANGED if t == 0 else I)
    assert any(e[0] == "parry_proj" for e in log) and w.f[0].hp == 100


def hold(action, start, end):
    """start~end-1 프레임 동안 누르고 있기 (강공격용)."""
    return {t: action for t in range(start, end)}


def test_parry_cooldown():
    w = World(200, 700)
    run(w, {0: Act.PARRY}, {}, 31)          # 패링 끝 (총 30프레임)
    w.step(Act.PARRY, I)
    assert w.f[0].state == St.IDLE, "쿨타임 중엔 패링이 안 나가야 함"
    run(w, {}, {}, 10)
    w.step(Act.PARRY, I)
    assert w.f[0].state == St.PARRY, "쿨타임이 끝나면 다시 나가야 함"


def heavy_active_frame(held):
    """held 프레임 누르고 뗐을 때 강공격 판정이 처음 나는 프레임 (0 = 누르기 시작)."""
    w = World(400, 900)
    for t in range(held + 40):
        w.step(Act.HEAVY if t < held else I, I)
        if w.f[0].attack_active():
            return t
    return None


def test_heavy_breaks_parry():
    w = World(400, 460)
    act = heavy_active_frame(20)
    log = run(w, hold(Act.HEAVY, 0, 20), {act - 3: Act.PARRY}, act + 10)
    assert any(e[0] == "parry_break" for e in log), log
    assert w.f[1].hp < 100 and w.f[0].hp == 100


def test_heavy_min_charge_is_feint():
    # 13프레임까지 누르고 떼면 페인트: 공격이 안 나가고 짧게 경직
    w = World(400, 460)
    run(w, hold(Act.HEAVY, 0, 13), {}, 14)
    assert w.f[0].state == St.RECOVER
    run(w, {}, {}, 40)
    assert w.f[1].hp == 100


def test_heavy_stage_damage():
    for held, dmg in ((14, 12), (27, 12), (28, 16), (43, 16), (44, 22), (59, 22)):
        w = World(400, 460)
        log = run(w, hold(Act.HEAVY, 0, held), {}, held + 40)
        hs = hits(log)
        assert hs and hs[0][4] == dmg, (held, hs)


def test_heavy_auto_release():
    # 60프레임(1초)이 되면 떼지 않아도 3단계로 나간다
    w = World(400, 460)
    log = run(w, hold(Act.HEAVY, 0, 200), {}, 200)
    rel = [e for e in log if e[0] == "heavy_release"]
    assert rel and rel[0][2] == 3, rel
    assert hits(log)[0][4] == 22
    assert heavy_active_frame(60) == heavy_active_frame(200) == 69


def test_heavy_charge_stage_visible():
    w = World(400, 900)
    stages = []
    for t in range(60):
        w.step(Act.HEAVY, I)
        stages.append(w.f[0].charge_stage)
    assert stages[12] == 0 and stages[13] == 1 and stages[27] == 2 and stages[43] == 3, stages


def test_heavy_reactable():
    # 가장 빠른 강공격(14프레임 차지)도, 240ms(약 14프레임) 뒤에 정면 공격으로 반응하면 먼저 맞힌다
    assert heavy_active_frame(14) == 24
    w = World(400, 455)
    log = run(w, hold(Act.HEAVY, 0, 14), {14: Act.MID}, 40)
    assert w.f[0].hp < 100 and w.f[1].hp == 100, hits(log)


def test_mid_interrupts_charge():
    w = World(400, 455)
    log = run(w, hold(Act.HEAVY, 0, 40), {2: Act.MID}, 40)
    assert w.f[0].hp < 100 and w.f[1].hp == 100


def test_heavy_not_in_combo():
    w = World(400, 455)
    run(w, {0: Act.MID, 9: Act.LOW}, {}, 10)
    w2 = World(400, 455)
    log = run(w2, {0: Act.MID, **hold(Act.HEAVY, 8, 60)}, {}, 60)
    assert [h[5] for h in hits(log)][:2] != [1, 2], hits(log)


def test_determinism():
    def play(seed):
        rng = random.Random(seed)
        w = World()
        for _ in range(3000):
            w.step(rng.randrange(12), rng.randrange(12))
        return w.snapshot(), w.frame
    assert play(1) == play(1)


def test_env_episode():
    env = StickFightEnv(opponent=AggroBot(seed=0), seed=0)
    obs, _ = env.reset()
    assert obs.shape == (env.observation_size,)
    total, done, n = 0.0, False, 0
    me = RandomBot(seed=1)
    while not done:
        obs, r, done, _, info = env.step(me.act(obs, env.world, 0))
        total += r
        n += 1
    assert info["winner"] in (0, 1, -1)


# ---------------------------------------------------------------- 콤보 브레이커
def _breaker_world(full=True):
    w = World(400, 455)
    if full:
        w.f[1].break_gauge = C.BREAKER.gauge_max
    return w


def _mid_low(w, p2_script, frames=40):
    """P1: 정면 → (히트 후) 하단 캔슬. P2는 p2_script = {프레임: 행동}."""
    return run(w, {0: Act.MID, 9: Act.LOW}, p2_script, frames)


def test_breaker_correct_button():
    w = _breaker_world()
    log = _mid_low(w, {15: Act.LOW})        # 2타째(하단)를 보고 하단 버튼
    assert any(e[0] == "break" for e in log), log
    assert len(hits(log)) == 1, hits(log)   # 2타째는 안 들어감
    a, d = w.f
    assert d.break_gauge == 0.0
    assert abs(a.x - d.x) > 70, (a.x, d.x)  # 공격자가 밀려남


def test_breaker_wrong_button_locks():
    w = _breaker_world()
    log = _mid_low(w, {15: Act.HIGH, 17: Act.LOW})   # 틀림 → 잠겨서 바로 다시 눌러도 안 됨
    assert any(e[0] == "break_fail" for e in log)
    assert not any(e[0] == "break" for e in log)
    assert len(hits(log)) == 2, hits(log)
    assert w.f[1].break_gauge == C.BREAKER.gauge_max   # 게이지는 남는다
    run(w, {}, {}, 60)
    assert not w.f[1].break_locked                     # 콤보가 끝나면 잠김 풀림


def test_breaker_needs_use_min():
    w = _breaker_world(full=False)
    w.f[1].break_gauge = C.BREAKER.use_min - 18    # 1타 맞고도 59.5 → 60 미만이라 못 씀
    log = _mid_low(w, {15: Act.LOW})
    assert not any(e[0] in ("break", "break_fail") for e in log)
    assert len(hits(log)) == 2
    w = _breaker_world(full=False)
    w.f[1].break_gauge = C.BREAKER.use_min - 10    # 1타(정면 7 → +17.5) 맞고 60 이상이 되면 씀 (전부 소모)
    log = _mid_low(w, {15: Act.LOW})
    assert any(e[0] == "break" for e in log)
    assert w.f[1].break_gauge == 0.0


def test_breaker_not_on_first_hit():
    w = _breaker_world()
    log = run(w, {0: Act.MID}, {9: Act.MID}, 40)       # 1타째 동안 눌러도 판정 안 함
    assert not any(e[0] in ("break", "break_fail") for e in log)
    assert not w.f[1].break_locked


def test_breaker_no_guaranteed_punish():
    # 끊은 뒤 바로 정면을 내밀어도 공격자가 패링으로 받을 수 있다 (확정 반격 없음)
    w = _breaker_world()
    log = _mid_low(w, {15: Act.LOW}, 16)
    assert any(e[0] == "break" for e in log)
    t0 = None
    for t in range(80):
        a, d = w.f
        act1 = Act.PARRY if a.state == St.IDLE and t0 is None else I
        if act1 == Act.PARRY:
            t0 = t
        act2 = Act.MID if d.state == St.IDLE else I
        log += w.step(act1, act2)
    assert not any(e[0] == "hit" and e[1] == 1 for e in log[-200:]), hits(log)


def test_gauge_fills_and_bonus():
    w = World(400, 455)
    run(w, {0: Act.MID}, {}, 30)
    taken = C.MAX_HP - w.f[1].hp
    assert w.f[1].break_gauge == taken * C.BREAKER.gain_per_damage
    w2 = World(400, 455)
    w2.f[0].break_gauge = C.BREAKER.gauge_max
    log = run(w2, {0: Act.MID}, {}, 30)
    bonus = C.MAX_HP - w2.f[1].hp
    assert bonus == round(taken * (1 + C.BREAKER.full_bonus)) and bonus > taken, (taken, bonus)
    assert any(e[0] == "hit_bonus" for e in log)


def test_gauge_decays_after_delay():
    w = World(400, 455)
    run(w, {0: Act.MID}, {}, 30)
    g = w.f[1].break_gauge
    bk = C.BREAKER
    run(w, {}, {}, bk.decay_delay - 30)          # 마지막 피격 뒤 아직 3초 안 됨
    assert w.f[1].break_gauge == g
    run(w, {}, {}, 60)                           # 3초 지나면 줄어든다
    assert w.f[1].break_gauge < g
    run(w, {}, {}, 600)
    assert w.f[1].break_gauge == 0.0


def test_full_gauge_bonus_expires():
    w = World(400, 455)
    w.f[0].break_gauge = C.BREAKER.gauge_max
    run(w, {}, {}, C.BREAKER.decay_delay + 2)    # 가득 찼어도 안 맞으면 MAX가 풀린다
    assert not w.f[0].gauge_full
    log = run(w, {0: Act.MID}, {}, 30)
    assert not any(e[0] == "hit_bonus" for e in log)


def test_hit_resets_decay_timer():
    w = World(400, 455)
    w.f[1].break_gauge = 50.0
    run(w, {}, {}, C.BREAKER.decay_delay + 30)
    low = w.f[1].break_gauge
    assert low < 50.0
    run(w, {0: Act.MID}, {}, 30)                 # 맞으면 다시 차고 타이머 처음부터
    g = w.f[1].break_gauge
    assert g > low
    run(w, {}, {}, 100)
    assert w.f[1].break_gauge == g


def test_breaker_bot_uses_it():
    # 공격형 봇끼리 많이 싸우면 브레이커 성공·실패가 둘 다 나온다
    env = StickFightEnv(opponent=AggroBot(seed=3), seed=3)
    obs, _ = env.reset()
    me = AggroBot(seed=4)
    kinds = set()
    for ep in range(6):
        done = False
        while not done:
            from ai.perception import SeenWorld
            obs, r, done, _, info = env.step(me.act(obs, SeenWorld(env.world, env.eyes[0]), 0))
            kinds |= {e[0] for e in info["events"]}
        obs, _ = env.reset()
    assert "break" in kinds, kinds


# ---------------------------------------------------------------- 반응 지연
def test_pick_delay_lands_on_tick():
    for s in range(0, 40):
        for T in (7.8, 9.0, 10.6, 11.0, 13.3, 14.4):
            d = pick_delay(s, T, 4, 8)
            assert (s + d) % 4 == 0, (s, T, d)            # 판단 시점에 처음 보인다
            assert d >= 8, (s, T, d)                      # 130ms보다 빠르지 않다
            if T >= 10:
                assert abs(d - T) <= 2.0, (s, T, d)      # 목표에서 ±2프레임 안


def effective_reactions(n=4000, seed=0):
    """상대가 무작위 시점에 새 행동을 할 때, AI가 판단 시점에서 그 행동을 처음 보기까지 걸린 프레임."""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        w = World(300, 700)
        eye = Perception(0, 4, Reaction(rng=random.Random(rng.random())))
        eye.reset(w)
        start = rng.randrange(20, 40)
        seen_at = None
        for t in range(1, start + 40):
            w.step(I, Act.MID if t == start else I)
            eye.push(w)
            if eye.t % 4 == 0:
                o, _, v = eye.view()
                if seen_at is None and v >= start:
                    seen_at = eye.t
        out.append(seen_at - start)
    return out


def test_reaction_distribution():
    r = effective_reactions()
    ms = [x * 1000 / 60 for x in r]
    mean = sum(ms) / len(ms)
    assert min(r) >= 8, min(r)                 # 130ms보다 빠른 반응은 없다
    assert max(r) <= 16, max(r)                # 240ms + 판단 시점 보정(최대 2프레임)
    assert 175 <= mean <= 192, mean            # 삼각분포 평균 183ms 근처


def test_view_never_goes_back():
    rng = random.Random(3)
    w = World()
    eye = Perception(0, 4, Reaction(rng=random.Random(1)))
    eye.reset(w)
    last = -1
    for t in range(1, 3000):
        w.step(rng.randrange(12), rng.randrange(12))
        eye.push(w)
        if w.done:
            break
        if eye.t % 4 == 0:
            _, _, v = eye.view()
            assert v >= last
            last = v


def test_opponent_parry_cooldown_hidden():
    # 상대 패링 쿨타임만 다른 두 상황에서 AI가 보는 값은 같아야 한다
    def obs_with(pcd):
        w = World(300, 600)
        eye = Perception(0, 4, Reaction(rng=random.Random(0)))
        eye.reset(w)
        for _ in range(16):
            w.step(I, I)
            w.f[1].parry_cd = pcd
            eye.push(w)
        return build_obs(w, eye)
    assert (obs_with(0) == obs_with(35)).all()


def bench():
    rng = random.Random(0)
    w = World()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        if w.done:
            w.reset()
        w.step(rng.randrange(12), rng.randrange(12))
        n += 1
    fps = n / (time.perf_counter() - t0)
    print(f"시뮬레이션 속도: {fps:,.0f} 프레임/초 (실시간의 {fps / 60:,.0f}배, 코어 1개)")


# ---------------------------------------------------------------- 훈련장
def test_training_no_ko_no_timeout():
    """훈련장: 체력이 0이 되면 바로 가득 차고, 시간이 지나도 라운드가 끝나지 않는다."""
    from game import config as C
    w = World(400, 455, training=True)
    w.f[1].hp = 3
    run(w, {0: Act.MID}, {}, 30)
    assert w.f[1].hp == C.MAX_HP and w.f[1].state != St.KO and not w.done
    for _ in range(C.ROUND_FRAMES + 10):
        w.step(I, I)
    assert not w.done


def test_training_knockback_same_as_match():
    """훈련장도 넉백/경직은 실전과 같다."""
    a, b = World(400, 455), World(400, 455, training=True)
    run(a, {0: Act.HIGH}, {}, 40)
    run(b, {0: Act.HIGH}, {}, 40)
    assert a.snapshot() == b.snapshot()
    assert b.f[1].x > 455


def test_damage_meter_combo_and_dps():
    from game import config as C
    from game.training import DamageMeter, SESSION_GAP
    w = World(400, 450, training=True)
    m = DamageMeter(0)
    script = {0: Act.MID, 12: Act.LOW, 28: Act.HIGH}
    for t in range(60):
        m.step(w, w.step(script.get(t, I), I))
    dealt = C.MAX_HP - w.f[1].hp
    assert m.n_hits == 3 and m.best_hits == 3, (m.n_hits, m.best_hits)
    assert m.total == m.combo_dmg == m.best_dmg == dealt, (m.total, dealt)
    assert m.dps() > 0 and m.session_active
    for _ in range(SESSION_GAP + 1):          # 3초 쉬면 구간이 끝나고 DPS가 고정된다
        m.step(w, w.step(I, I))
    d = m.dps()
    assert not m.session_active
    for _ in range(100):
        m.step(w, w.step(I, I))
    assert m.dps() == d


# ---------------------------------------------------------------- 대쉬 공격 (v6)
def test_dash_attack_comes_out_and_reaches_far():
    # 앞 대쉬 중 정면 → 대쉬 공격. 서 있는 상대를 180px 밖에서 13프레임 만에 맞힌다
    w = World(300, 480)
    log = run(w, {0: Act.DASH_F, 3: Act.MID}, {}, 30)
    hs = hits(log)
    assert hs and hs[0][3] == C.DASH_ATK, hs


def test_dash_attack_faster_than_dash_then_mid():
    def first_hit(script):
        w = World(300, 480)
        for t in range(60):
            if hits(w.step(script.get(t, I), I)):
                return t
    da = first_hit({0: Act.DASH_F, 3: Act.MID})
    plain = first_hit({0: Act.DASH_F, 18: Act.MID})
    assert da is not None and plain is not None and da + 8 <= plain, (da, plain)


def test_no_dash_attack_from_back_dash_or_air():
    w = World(300, 600)
    run(w, {0: Act.DASH_B, 3: Act.MID}, {}, 6)
    assert w.f[0].kind != C.DASH_ATK
    w = World(300, 600)
    run(w, {0: Act.JUMP, 2: Act.DASH_F, 5: Act.MID}, {}, 8)
    assert w.f[0].kind != C.DASH_ATK


def test_dash_attack_jumpable_and_parryable():
    # 점프는 대쉬를 보고 일찍 뛰어야 피한다 (판정 위쪽 96)
    w = World(300, 480)
    run(w, {0: Act.DASH_F, 3: Act.MID}, {4: Act.JUMP}, 30)
    assert w.f[1].hp == 100
    # 180px에서 대쉬 3프레임째 찌르면 13프레임에 맞음 → 11프레임째 패링(반응)이 맞는다. 더 가까우면 미리 눌러야 함
    w = World(300, 480)
    log = run(w, {0: Act.DASH_F, 3: Act.MID}, {11: Act.PARRY}, 20)
    assert any(e[0] == "parry" for e in log) and w.f[0].state == St.PARRIED
    w = World(300, 480)
    log = run(w, {0: Act.DASH_F, 3: Act.MID}, {8: Act.PARRY}, 20)
    assert any(e[0] == "parry" for e in log) and w.f[0].state == St.PARRIED


def test_dash_attack_cancels_into_combo():
    w = World(300, 430)
    log, t_hit = [], None
    for t in range(60):
        a = {0: Act.DASH_F, 3: Act.MID}.get(t, I)
        if t_hit is not None and t == t_hit + 3:
            a = Act.LOW
        ev = w.step(a, I)
        if t_hit is None and hits(ev):
            t_hit = t
        log += ev
    kinds = [h[3] for h in hits(log)]
    assert kinds == [C.DASH_ATK, Act.LOW], kinds


def test_dash_attack_whiff_is_punishable():
    # 헛친 대쉬 공격의 후딜 동안 상대 정면 공격이 들어간다
    w = World(300, 560)
    run(w, {0: Act.DASH_F, 3: Act.MID}, {}, 14)
    assert w.f[0].state == St.ATTACK and w.f[0].kind == C.DASH_ATK and w.f[1].hp == 100
    gap = w.f[1].x - w.f[0].x
    for t in range(40):
        w.step(I, Act.LEFT if w.f[1].x - w.f[0].x > 60 else Act.MID)
        if w.f[0].hp < 100:
            break
    assert w.f[0].hp < 100, gap


def test_dash_attack_label_seen_by_ai():
    w = World(300, 600)
    run(w, {0: Act.DASH_F, 3: Act.MID}, {}, 5)
    assert C.LABELS[w.f[0].label] == "DASH_ATK"
    from ai.perception import SeenFighter
    sf = SeenFighter(0, w.f[0].snapshot(), w.f[0].hp)
    assert sf.state == St.ATTACK and sf.kind == C.DASH_ATK


# ---------------------------------------------------------------- 백대쉬 후딜·벽꽝 (v6)
def test_back_dash_recovery_longer():
    w = World(400, 600)
    for t in range(40):
        w.step(Act.DASH_B if t == 0 else I, I)
        if w.f[0].state == St.IDLE:
            break
    back = t
    w = World(400, 900)
    for t in range(40):
        w.step(Act.DASH_F if t == 0 else I, I)
        if w.f[0].state == St.IDLE:
            break
    assert back - t == C.DASH.back_recovery - C.DASH.recovery, (back, t)


def _wall_world():
    lo = C.WALL_MARGIN + C.BODY_W / 2
    return World(lo + 75, lo + 2)     # P2가 왼쪽 벽에 붙어 있고 P1이 오른쪽에서 때림


def _wall_world_flush():
    """P2가 벽에 완전히 붙어 있음 (정면 넉백 1.5가 마찰로 줄기 전에 벽에 닿게)."""
    lo = C.WALL_MARGIN + C.BODY_W / 2
    return World(lo + 70, lo)


def test_wall_splat_extends_hitstun_once_per_combo():
    # (v11) 하단 넉백이 0.5가 되어 하단으로는 벽꽝이 안 난다 → 정면(넉백 1.5)으로 확인
    w = _wall_world_flush()
    log = run(w, {0: Act.MID}, {}, 20)
    assert [e for e in log if e[0] == "wall_splat"], "벽꽝 안 됨"
    d = w.f[1]
    assert d.wall_hit and d.state == St.HITSTUN
    # 같은 콤보 안에서는 다시 안 생긴다
    log = run(w, {0: Act.LOW}, {}, 30)
    assert not [e for e in log if e[0] == "wall_splat"]
    w = _wall_world()
    log = run(w, {0: Act.LOW}, {}, 20)
    assert hits(log) and not [e for e in log if e[0] == "wall_splat"]   # 하단은 벽꽝 없음


def test_wall_splat_lets_new_combo_connect():
    # 벽꽝이 있으면 맞은 쪽이 풀리는 시점이 extra_stun만큼 늦어져, 후딜이 끝난 뒤 새 공격이 이어진다
    def free_at(wall_on):
        old = C.WALL
        if not wall_on:
            C.WALL = C.WallData(min_speed=1e9)
        try:
            w = _wall_world_flush()
            log = []
            for t in range(80):
                log += w.step(Act.MID if t == 0 else I, I)
                if hits(log) and w.f[1].state == St.IDLE:
                    return t
        finally:
            C.WALL = old
    on, off = free_at(True), free_at(False)
    assert on - off >= C.WALL.extra_stun, (on, off)


def test_breaker_push_no_wall_splat():
    w = World(200, 100 + C.WALL_MARGIN)
    w.f[0].break_gauge = 100
    w.f[1].facing = 1
    log = []
    for t in range(80):
        a1 = {0: Act.MID}.get(t, I)
        if w.f[1].state == St.ATTACK and w.f[1].can_cancel():
            a1 = Act.LOW if w.f[1].kind == Act.MID else Act.MID
        a0 = w.f[1].kind if (w.f[0].state == St.HITSTUN and w.f[1].chain >= 2) else I
        log += w.step(a0 or I, a1)
    assert any(e[0] == "break" for e in log), "브레이커가 안 나감"
    assert not any(e[0] == "wall_splat" and e[1] == 1 for e in log)


def test_back_dash_no_invuln_in_corner():
    # 구석에서 백대쉬는 움직이지 못하고 무적도 없어서 맞는다
    lo = C.WALL_MARGIN + C.BODY_W / 2
    for r in range(8):
        w = World(lo + 55, lo + 1)
        run(w, {0: Act.MID}, {r: Act.DASH_B}, 40)
        assert w.f[1].hp < 100, r
    # 앞 대쉬도 무적 없음: 상대 공격 속으로 대쉬하면 언제 눌러도 맞는다
    for r in range(8):
        w = World(400, 455)
        run(w, {0: Act.MID}, {r: Act.DASH_F}, 30)
        assert w.f[1].hp < 100, r


def test_back_dash_still_escapes_in_open():
    w = World(400, 455)
    run(w, {0: Act.MID}, {3: Act.DASH_B}, 30)
    assert w.f[1].hp == 100


def test_dash_spam_through_attack_not_free():
    # 상대 정면 공격을 대쉬 무적으로 지나쳐 대쉬 공격으로 때리는 패턴: 대쉬가 판정보다 3프레임 이상 먼저면 맞는다
    for r in range(0, 4):
        w = World(300, 400)
        run(w, {r: Act.DASH_F, r + 5: Act.MID}, {0: Act.MID}, 40)
        assert w.f[0].hp < 100, r


def test_ranged_cooldown_120():
    w = World(300, 900)
    shots = []
    for t in range(200):
        for e in w.step(Act.RANGED, I):
            if e[0] == "shoot":
                shots.append(t)
    assert C.RANGED.cooldown == 120 and len(shots) >= 2 and 120 <= shots[1] - shots[0] <= 126, shots


def test_tier_bot_reaction():
    """난이도 봇: 하위 봇만 반응이 느리고(180~360ms), 중간·상위는 사람 기준(130~240ms). 더 빠른 봇은 없음."""
    for bot, lo, hi in ((EasyBot(seed=0), 180, 360), (NormalBot(seed=0), 130, 240), (HardBot(seed=0), 130, 240)):
        env = StickFightEnv(opponent=bot, seed=0)
        rx = env.eyes[1].reaction
        assert (rx.low, rx.high) == (lo, hi) and env.eyes[0].reaction.low == 130, (bot, rx.low, rx.high)
        assert rx.low >= 130


def test_tier_bots_order():
    """상위 봇이 중간 봇을, 중간 봇이 하위 봇을 이긴다 (각 20판, 넉넉한 기준)."""
    def wins(a, b, n=20):
        env = StickFightEnv(opponent=b, seed=5, reaction_ms=a.reaction_ms)
        k = 0
        for _ in range(n):
            obs, _ = env.reset()
            done = False
            while not done:
                obs, _, done, _, info = env.step(a.act(obs, SeenWorld(env.world, env.eyes[0]), 0))
            k += info["winner"] == 0
        return k / n
    assert wins(NormalBot(seed=1), EasyBot(seed=2)) >= 0.7
    assert wins(HardBot(seed=1), NormalBot(seed=2)) >= 0.7


def test_heavy_step_reach():
    """(v8) 강공격은 내려치며 한 발 내딛는다: 기 모으는 동안은 제자리, 판정 끝까지 20~32px 전진, 135px 거리도 닿는다."""
    w = World(300, 900)
    for t in range(14):
        w.step(Act.HEAVY, I)
    assert w.f[0].x == 300
    x0 = w.f[0].x
    for t in range(40):
        w.step(I, I)
    assert 20 <= w.f[0].x - x0 <= 32, w.f[0].x - x0
    for dist, should in ((135, True), (150, False)):
        w = World(300, 300 + dist)
        run(w, hold(Act.HEAVY, 0, 14), {}, 60)
        assert (w.f[1].hp < 100) == should, dist


# ---------------------------------------------------------------- v11: 숙이기, 하단 경직
def _crouch_vs(kind, dist, frames=40):
    """P1은 계속 숙이고, P2가 dist 거리에서 kind로 공격. P1이 맞았나."""
    w = World(300, 300 + dist)
    log = []
    for t in range(frames):
        log += w.step(Act.CROUCH, kind if t == 0 else I)
    return any(e[0] == "hit" and e[2] == 0 for e in log)


def test_crouch_dodges_high_and_projectile():
    assert not _crouch_vs(Act.HIGH, 55)
    assert not _crouch_vs(Act.RANGED, 200, 60)
    # 서 있으면 둘 다 맞는다
    for kind, dist, n in ((Act.HIGH, 55, 40), (Act.RANGED, 200, 60)):
        w = World(300, 300 + dist)
        log = run(w, {}, {0: kind}, n)
        assert any(e[0] == "hit" and e[2] == 0 for e in log), kind


def test_crouch_hit_by_low_mid_dash_attack():
    assert _crouch_vs(Act.LOW, 70)
    assert not _crouch_vs(Act.MID, 55)        # (v13) 정면은 숙이면 피한다 (숙이기 64 < 정면 판정 68)
    w = World(300, 480)                       # 대쉬 → 대쉬 공격 (판정 높이 44~96)
    log = []
    for t in range(40):
        a1 = Act.DASH_F if t == 0 else (Act.MID if t == 3 else I)
        log += w.step(Act.CROUCH, a1)
    assert any(e[0] == "hit" and e[2] == 0 and e[3] == C.DASH_ATK for e in log)


def test_crouch_hold_release_and_act():
    w = World(300, 700)
    for _ in range(5):
        w.step(Act.CROUCH, I)
    f = w.f[0]
    assert f.state == St.CROUCH and f.hurtbox()[3] == C.BODY_H_CROUCH
    x = f.x
    w.step(Act.CROUCH, I)
    assert f.x == x                           # 숙인 채로는 안 걷는다
    w.step(I, I)
    assert f.state == St.IDLE and f.hurtbox()[3] == C.BODY_H     # 떼면 바로 일어남
    for _ in range(3):
        w.step(Act.CROUCH, I)
    w.step(Act.MID, I)
    assert f.state == St.ATTACK and f.kind == Act.MID and f.sf <= 1   # 숙인 채 공격 → 바로 나감
    w = World(300, 700)
    for _ in range(3):
        w.step(Act.CROUCH, I)
    w.step(Act.JUMP, I)
    w.step(Act.CROUCH, I)
    assert w.f[0].y > 0 and w.f[0].state == St.IDLE   # 점프도 바로, 공중에선 안 숙임


def test_crouch_label_in_obs():
    w = World(300, 700)
    for _ in range(3):
        w.step(Act.CROUCH, I)
    assert w.f[0].label == C.LABELS.index("CROUCH")


def _low_then(dist, follow=None, delay=2):
    """하단을 맞힌 뒤 (공격자 자유 프레임, 방어자 자유 프레임, follow 캔슬이 맞았나)."""
    w = World(300, 300 + dist)
    w.step(Act.LOW, I)
    n, t_hit, fa, fd, hits = 0, None, None, None, 0
    while n < 100:
        a = follow if (follow is not None and t_hit is not None and n - t_hit == delay) else I
        ev = w.step(a, I)
        n += 1
        for e in ev:
            if e[0] == "hit":
                hits += 1
                if t_hit is None:
                    t_hit = n
        if t_hit is not None and follow is None:
            if fa is None and w.f[0].state == St.IDLE:
                fa = n - t_hit
            if fd is None and w.f[1].state == St.IDLE:
                fd = n - t_hit
            if fa is not None and fd is not None:
                break
    return fa, fd, hits >= 2


def test_low_on_hit_advantage_and_cancel():
    """(v11) 하단 경직 30·넉백 0.5: 맞히면 +3 (전에는 -5), 85px까지 정면 캔슬이 이어진다."""
    fa, fd, _ = _low_then(70)
    assert fd - fa == 3, (fa, fd)
    for dist in (60, 70, 80, 85):
        assert _low_then(dist, Act.MID)[2], dist
    assert not _low_then(90, Act.MID)[2]          # 사거리 끝에서 맞히면 안 이어짐
    assert _low_then(70, Act.HIGH)[2]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, e)
    bench()
    raise SystemExit(1 if failed else 0)
