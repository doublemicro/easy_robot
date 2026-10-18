"""1단계: PPO로 AI 학습 (PyTorch, 직접 구현). 상대는 하위 → 중간 → 상위 봇 순서 (커리큘럼).

    python train.py                          # 기본 1천만 판단 (이 PC 기준 약 7분)
    python train.py --steps 2000000 --name test   # 짧게 돌려 보기
    python train.py --resume runs/stage1/latest.pt # 이어서 학습
    (또는 train.bat 더블클릭)

구조
- 작업 프로세스(기본 = 코어 수 - 1)가 각자 게임 8판을 동시에 돌린다. 매 반복마다 최신 정책을 받아
  판마다 256번 판단한 기록을 돌려준다 → 한 번에 약 2만 개 기록으로 PPO 업데이트.
- 판이 끝날 때마다 새 판: 내 자리(P1/P2)는 무작위, 상대는 현재 단계의 비율대로 뽑는다.
- 보상은 env 그대로: 준 데미지 - 받은 데미지 (×0.5, 체력 비율) + 이기면 +1 / 지면 -1.

커리큘럼 (최근 300판 승률로 판단, 확률대로 고르는 정책 그대로)
  단계 1  하위 100%                         → 하위 상대 80% 이상이면 다음
  단계 2  중간 80% + 하위 20%               → 중간 상대 75% 이상이면 다음  (1단계 목표: 중간 봇 확실히 이기기)
  단계 3  상위 70% + 중간 20% + 하위 10%    → 상위 상대 50% 이상이면 다음
  단계 4  (v10) 리그: 과거의 나 35% + 거리 조절 20% + 상위 20% + 숙이기·하단 15% (v13) + 중간 10%  → 끝까지
          과거의 나 = 이 단계에서 snap-every 반복마다 찍어 둔 내 사본 (최근 20개 중 무작위).
          봇만 상대하면 봇이 못 하는 것(예: 헛친 대쉬 공격 벌하기)을 막는 법을 안 배운다 → 나와 싸우며 배운다.

    python train.py --resume runs/stage1/latest.pt --stage 4 --name league   # 1단계 AI에서 리그만 이어서

반복 동작 감지 (v10, ai/repeat.py)
  AI가 실제로 낸 기술 순서를 반복마다 모아서, 기술 하나가 --repeat-move(40%)를 넘거나
  같은 n연속 순서가 --repeat-grams(3연속 20%, 4연속 12%, 5연속 8%, 6연속 6%)을 넘는 상태가
  --repeat-patience(10)번 연속이면 멈추고 repeat_stop.pt를 남긴다. 기록에는 넘은 것 중 가장 긴 순서가 대표로 남는다.
  (참고: 사람이 만든 봇들 27~36% / 5~24%, 대쉬 공격·정면 연타였던 1단계 AI 42% / 26%)
  --no-repeat-stop 이면 멈추지 않고 기록만.

저장 (runs/<name>/)
  latest.pt              주기적으로 / 끝날 때 / Ctrl+C
  clear_<봇>.pt          단계를 통과한 순간
  best_<봇>.pt           그 봇 상대 최근 300판 승률이 최고일 때
  repeat_stop.pt         반복 동작으로 멈춘 순간
  log.csv                반복마다 진행 기록
"""
import argparse
import csv
import multiprocessing as mp
import os
import random
import time
from collections import deque

import numpy as np
import torch
from torch.distributions import Categorical

from ai.bots import CrouchLowBot, EasyBot, HardBot, NormalBot, SpacerBot
from ai.env import StickFightEnv
from ai.policy import ActorCritic, PolicyBot
from ai.repeat import (GRAM_LIMITS, MOVE_LIMIT, MoveTracker, by_opponent, check, parse_limits, record,
                       summarize)

OPPONENTS = {"easy": EasyBot, "normal": NormalBot, "hard": HardBot, "spacer": SpacerBot,
             "crouchlow": CrouchLowBot}
KOR = {"easy": "하위", "normal": "중간", "hard": "상위", "spacer": "거리", "crouchlow": "숙하단", "self": "과거나"}
STAGES = (
    {"mix": {"easy": 1.0}, "goal": ("easy", 0.80)},
    {"mix": {"normal": 0.8, "easy": 0.2}, "goal": ("normal", 0.75)},
    {"mix": {"hard": 0.7, "normal": 0.2, "easy": 0.1}, "goal": ("hard", 0.50)},
    {"mix": {"self": 0.35, "spacer": 0.2, "hard": 0.2, "crouchlow": 0.15, "normal": 0.1}, "goal": ("spacer", 0.90)},
)
WINDOW = 300
POOL_MAX = 20


# ====================================================================== 작업 프로세스
def worker(rank, conn, n_envs, seed, draw_reward):
    torch.set_num_threads(1)
    rng = random.Random(seed)
    net = ActorCritic()
    net.eval()
    envs = [StickFightEnv(seed=seed * 1000 + i, draw_reward=draw_reward) for i in range(n_envs)]
    obs, names, ret = [None] * n_envs, [None] * n_envs, [0.0] * n_envs
    mix = None
    pool, pool_ver = [], -1                             # 과거의 나 (셀프플레이 상대)
    trackers = [MoveTracker() for _ in range(n_envs)]  # 내가 실제로 낸 기술 순서 (반복 동작 감지)

    def new_episode(i):
        name = rng.choices(list(mix), weights=list(mix.values()))[0]
        if name == "self" and not pool:
            name = "hard"
        env = envs[i]
        env.me = rng.randrange(2)                       # 양쪽 자리 모두 학습
        if name == "self":
            env.set_opponent(PolicyBot(rng.choice(pool), seed=rng.randrange(1 << 30)))
        else:
            env.set_opponent(OPPONENTS[name](seed=rng.randrange(1 << 30)))
        names[i], ret[i] = name, 0.0
        obs[i], _ = env.reset()
        trackers[i].reset()

    while True:
        msg = conn.recv()
        if msg[0] == "close":
            break
        _, state, mix, T, ver, states = msg
        net.load_state_dict(state)
        if ver != pool_ver and states is not None:
            pool_ver = ver
            pool = []
            for st in states:
                pn = ActorCritic()
                pn.load_state_dict(st)
                pn.eval()
                pool.append(pn)
        for i in range(n_envs):
            if obs[i] is None:
                new_episode(i)
        D = len(obs[0])
        O = np.zeros((T, n_envs, D), np.float32)
        A = np.zeros((T, n_envs), np.int64)
        LP = np.zeros((T, n_envs), np.float32)
        V = np.zeros((T, n_envs), np.float32)
        R = np.zeros((T, n_envs), np.float32)
        DN = np.zeros((T, n_envs), np.float32)
        results, seqs = [], []
        with torch.no_grad():
            for t in range(T):
                ob = np.stack(obs)
                logits, v = net(torch.from_numpy(ob))
                dist = Categorical(logits=logits)
                a = dist.sample()
                O[t], A[t], LP[t], V[t] = ob, a.numpy(), dist.log_prob(a).numpy(), v.numpy()
                for i, env in enumerate(envs):
                    o, r, done, _, info = env.step(int(A[t, i]))
                    R[t, i], DN[t, i] = r, float(done)
                    ret[i] += r
                    trackers[i].update(env.world.f[env.me])
                    if done:
                        seqs.append((names[i], trackers[i].moves))
                        w = info["winner"]
                        out = 1.0 if w == env.me else (0.0 if w == 1 - env.me else 0.5)
                        hp = env.world.f[env.me].hp - env.world.f[1 - env.me].hp
                        results.append((names[i], out, ret[i], hp))
                        new_episode(i)
                    else:
                        obs[i] = o
            _, last_v = net(torch.from_numpy(np.stack(obs)))
        for i, tr in enumerate(trackers):
            seqs.append((names[i], tr.moves))
            tr.moves = []
        conn.send((O, A, LP, V, R, DN, last_v.numpy(), results, seqs))


# ====================================================================== PPO
def gae(R, V, DN, last_v, gamma, lam):
    """GAE(λ). DN[t]=1 이면 t번째 행동으로 판이 끝남 → 다음 가치는 0."""
    T = len(R)
    adv = np.zeros_like(R)
    nxt_adv = np.zeros_like(last_v)
    nxt_v = last_v
    for t in reversed(range(T)):
        alive = 1.0 - DN[t]
        delta = R[t] + gamma * nxt_v * alive - V[t]
        nxt_adv = delta + gamma * lam * alive * nxt_adv
        adv[t] = nxt_adv
        nxt_v = V[t]
    return adv, adv + V


def ppo_update(net, opt, batch, a, device):
    O, A, LP, ADV, RET = (torch.as_tensor(x, device=device) for x in batch)
    n = len(O)
    stats = []
    for _ in range(a.epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, a.minibatch):
            idx = perm[s:s + a.minibatch]
            logits, v = net(O[idx])
            dist = Categorical(logits=logits)
            lp = dist.log_prob(A[idx])
            ratio = torch.exp(lp - LP[idx])
            adv = ADV[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv).mean()
            vl = 0.5 * ((v - RET[idx]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + a.vf_coef * vl - a.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad)
            opt.step()
            with torch.no_grad():
                kl = (LP[idx] - lp).mean()
                clipfrac = ((ratio - 1).abs() > a.clip).float().mean()
            stats.append([pg.item(), vl.item(), ent.item(), kl.item(), clipfrac.item()])
    return np.mean(stats, axis=0)


def save(path, net, opt, a, stage, steps, extra=None):
    torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()},
                "optimizer": opt.state_dict(), "net_config": net.config(),
                "stage": stage, "steps": steps, "args": vars(a), "extra": extra or {},
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}, path)


# ====================================================================== 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="stage1")
    ap.add_argument("--steps", type=float, default=10e6, help="총 판단 횟수 (4프레임마다 1번)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--envs", type=int, default=8, help="작업 프로세스 하나당 동시에 돌리는 판 수")
    ap.add_argument("--rollout", type=int, default=256, help="한 번에 모으는 판단 수 (판마다)")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--draw-reward", type=float, default=0.0)
    ap.add_argument("--stage", type=int, default=None, help="시작 단계 (1~4). --resume과 같이 쓰면 저장된 단계 대신")
    ap.add_argument("--snap-every", type=int, default=10, help="단계 4에서 몇 반복마다 내 사본을 리그에 넣나")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="GPU가 있어도 CPU로 업데이트")
    ap.add_argument("--repeat-move", type=float, default=MOVE_LIMIT, help="기술 하나의 비율이 이보다 크면 반복")
    ap.add_argument("--repeat-grams", default=",".join(f"{k}:{v}" for k, v in GRAM_LIMITS.items()),
                    help="n연속 순서의 기준, 예: 3:0.2,4:0.12,5:0.08,6:0.06")
    ap.add_argument("--repeat-patience", type=int, default=10, help="반복이 몇 번 연속이면 멈추나")
    ap.add_argument("--repeat-warmup", type=int, default=10, help="처음 몇 반복은 감지하지 않음")
    ap.add_argument("--no-repeat-stop", action="store_true", help="반복을 감지해도 멈추지 않고 기록만")
    a = ap.parse_args()
    a.gram_limits = parse_limits(a.repeat_grams)

    out = os.path.join("runs", a.name)
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    torch.manual_seed(a.seed)
    net = ActorCritic().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    stage, steps = (a.stage or 1) - 1, 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        stage, steps = ck["stage"] if a.stage is None else a.stage - 1, ck["steps"]
        pool_saved = ck.get("extra", {}).get("pool", [])
        print(f"이어서 학습: {a.resume} (단계 {stage + 1}, {steps:,} 판단)")

    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for r in range(a.workers):
        p_conn, c_conn = ctx.Pipe()
        p = ctx.Process(target=worker, args=(r, c_conn, a.envs, a.seed * 100 + r + 1, a.draw_reward),
                        daemon=True)
        p.start()
        conns.append(p_conn)
        procs.append(p)

    names = [*OPPONENTS, "self"]
    hist = {k: deque(maxlen=WINDOW) for k in names}
    best = {k: -1.0 for k in names}
    pool = pool_saved if a.resume else []
    pool_ver, sent_ver = 0, -1
    new_log = not os.path.exists(os.path.join(out, "log.csv"))
    logf = open(os.path.join(out, "log.csv"), "a", newline="", encoding="utf-8")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["steps", "minutes", "stage", "sps", *(f"win_{k}" for k in names),
                      *(f"n_{k}" for k in names), "ep_return", "hp_diff",
                      "pg_loss", "v_loss", "entropy", "kl", "clipfrac",
                      "top_move", "move_share", "top_gram", "gram_share", "gram5", "gram5_share",
                      "flag", "repeat_streak"])
    per_iter = a.workers * a.envs * a.rollout
    print(f"작업 프로세스 {a.workers}개 × 판 {a.envs}개, 반복당 {per_iter:,} 판단, 업데이트 장치 {device}")
    t0 = time.time()
    it = 0
    streak = 0
    try:
        while steps < a.steps:
            ti = time.time()
            st = STAGES[stage]
            state = {k: v.cpu() for k, v in net.state_dict().items()}
            if "self" in st["mix"] and (not pool or it % a.snap_every == 0):
                pool = (pool + [state])[-POOL_MAX:]
                pool_ver += 1
            fresh = pool if pool_ver != sent_ver else None       # 바뀌었을 때만 보낸다
            sent_ver = pool_ver
            for c in conns:
                c.send(("roll", state, st["mix"], a.rollout, pool_ver, fresh))
            parts = [c.recv() for c in conns]
            O, A, LP, V, R, DN = (np.concatenate([p[i] for p in parts], axis=1) for i in range(6))
            last_v = np.concatenate([p[6] for p in parts])
            results = [r for p in parts for r in p[7]]
            tagged = [q for p in parts for q in p[8]]
            rep = summarize([q for _, q in tagged])
            adv, ret = gae(R, V, DN, last_v, a.gamma, a.lam)
            flat = lambda x: x.reshape(-1, *x.shape[2:])
            pg, vl, ent, kl, cf = ppo_update(net, opt, (flat(O), flat(A), flat(LP), flat(adv), flat(ret)),
                                             a, device)
            steps += per_iter
            it += 1

            for name, res, _, _ in results:
                hist[name].append(1.0 if res == 1.0 else 0.0)
            rate = {k: (sum(h) / len(h) if h else float("nan")) for k, h in hist.items()}
            for k, h in hist.items():
                if k != "self" and len(h) == WINDOW and rate[k] > best[k]:
                    best[k] = rate[k]
                    save(os.path.join(out, f"best_{k}.pt"), net, opt, a, stage, steps, {"win": rate[k]})
            g_name, g_thr = st["goal"]
            cleared = len(hist[g_name]) == WINDOW and rate[g_name] >= g_thr
            if cleared and not os.path.exists(os.path.join(out, f"clear_{g_name}.pt")):
                save(os.path.join(out, f"clear_{g_name}.pt"), net, opt, a, stage, steps, {"win": rate[g_name]})
                print(f"*** 단계 {stage + 1} 통과: {KOR[g_name]} 봇 상대 승률 {rate[g_name]:.0%} ({steps:,} 판단)")
            if cleared and stage + 1 < len(STAGES):
                stage += 1
                for h in hist.values():
                    h.clear()

            rep["flag"] = check(rep, a.repeat_move, a.gram_limits)
            repeating = rep["moves"] >= 200 and it > a.repeat_warmup and rep["flag"] is not None
            streak = streak + 1 if repeating else 0

            mins = (time.time() - t0) / 60
            sps = per_iter / (time.time() - ti)
            er = np.mean([r[2] for r in results]) if results else float("nan")
            hd = np.mean([r[3] for r in results]) if results else float("nan")
            log.writerow([steps, f"{mins:.2f}", stage + 1, int(sps),
                          *(f"{rate[k]:.3f}" for k in names), *(len(hist[k]) for k in names),
                          f"{er:.3f}", f"{hd:.1f}", f"{pg:.4f}", f"{vl:.4f}", f"{ent:.3f}",
                          f"{kl:.4f}", f"{cf:.3f}", rep["top_move"], f"{rep['move_share']:.3f}",
                          rep["top_gram"], f"{rep['gram_share']:.3f}",
                          *rep["long"].get("5", ["-", 0]),
                          rep["flag"]["pattern"] if rep["flag"] else "", streak])
            logf.flush()
            if it % 5 == 0 or it == 1:
                wr = "  ".join(f"{KOR[k]} {rate[k]:4.0%}({len(hist[k])})" for k in names if hist[k])
                print(f"[{mins:5.1f}분] {steps / 1e6:6.2f}M  단계{stage + 1}  {sps:6.0f}/s  승률 {wr}  "
                      f"보상 {er:+.2f}  체력차 {hd:+.0f}  엔트로피 {ent:.2f}  "
                      f"반복 {rep['top_move']} {rep['move_share']:.0%} / {rep['top_gram']} {rep['gram_share']:.0%}"
                      + (f" / 걸림: {rep['flag']['pattern']} ({rep['flag']['len']}연속 {rep['flag']['share']:.0%})"
                         if rep["flag"] else "")
                      + (f" ({streak}/{a.repeat_patience})" if streak else ""), flush=True)
            if streak == a.repeat_patience:              # 밸런스 패치용 기록 (멈추지 않아도 남긴다)
                path = os.path.join(out, "repeat_stop.pt")
                save(path, net, opt, a, stage, steps, {"pool": pool, "repeat": rep})
                record(f"train.py --name {a.name}", rep, by_opponent(tagged),
                       {k: round(rate[k], 3) for k in names if hist[k]}, path, steps, stage + 1)
                print(f"!!! 반복 동작 기록 → balance_reports/사기기술.md", flush=True)
            if streak >= a.repeat_patience and not a.no_repeat_stop:
                save(os.path.join(out, "repeat_stop.pt"), net, opt, a, stage, steps, {"pool": pool, "repeat": rep})
                print(f"!!! 반복 동작 감지 → 학습 멈춤 ({steps:,} 판단, 단계 {stage + 1})\n"
                      f"    패턴 {rep['flag']['pattern']} ({rep['flag']['len']}연속 {rep['flag']['share']:.0%}), "
                      f"{streak}번 연속\n    → runs/{a.name}/repeat_stop.pt", flush=True)
                break
            if it % 20 == 0:
                save(os.path.join(out, "latest.pt"), net, opt, a, stage, steps, {"pool": pool})
    except KeyboardInterrupt:
        print("\n중단 — 지금까지 저장합니다")
    finally:
        save(os.path.join(out, "latest.pt"), net, opt, a, stage, steps, {"pool": pool})
        logf.close()
        for c in conns:
            try:
                c.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in procs:
            p.join(timeout=5)
    print(f"끝: {steps:,} 판단, {(time.time() - t0) / 60:.1f}분 → {out}")


if __name__ == "__main__":
    main()
