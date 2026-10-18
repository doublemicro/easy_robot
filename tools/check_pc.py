"""이 PC에서 AI 학습을 돌릴 준비가 됐는지 확인한다 (v9, 0단계).

    python -m tools.check_pc        (또는 check_pc.bat 더블클릭)

- CPU 코어 수 / 램 → 게임을 동시에 몇 판 돌릴지 정하는 기준
- PyTorch 설치 여부, GPU(CUDA) 사용 가능 여부
- 시뮬레이션 속도: 코어 1개에서 상위 봇 상대로 초당 몇 번 판단(4프레임)하는지
결과는 화면에 나오고 03_결과물에 넣기 좋게 check_pc_result.txt로도 저장된다.
"""
import os
import platform
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "check_pc_result.txt")
lines = []


def say(s=""):
    """한 줄 쓸 때마다 바로 파일에 저장 → 중간에 에러가 나도 그때까지 결과는 남는다."""
    print(s)
    lines.append(s)
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


say(f"Python {sys.version.split()[0]} / {platform.platform()}")
say(f"CPU: {platform.processor() or '?'}")
say(f"논리 코어 수: {os.cpu_count()}")
try:
    import psutil
    say(f"물리 코어 수: {psutil.cpu_count(logical=False)} / 램: {psutil.virtual_memory().total / 2**30:.1f} GB")
except ImportError:
    say("(psutil 없음 — 물리 코어 수·램은 생략. pip install psutil 하면 나옴)")

try:
    import numpy
    say(f"numpy {numpy.__version__}")
except ImportError:
    say("numpy 없음 → pip install numpy")

try:
    import torch
    say(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        say(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)} "
            f"(VRAM {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")
    else:
        say("CUDA 사용 불가 — CPU 버전 PyTorch가 깔렸거나 드라이버 문제")
except ImportError:
    say("PyTorch 없음 → 설치 필요 (README의 v9 설치 안내 참고)")

try:
    say()
    say("시뮬레이션 속도 측정 (10초, 코어 1개, 무작위 행동 vs 상위 봇)...")
    from ai.bots import BOTS
    from ai.env import StickFightEnv

    env = StickFightEnv(opponent=BOTS["hard"](), seed=0)
    env.reset(seed=0)
    rng = random.Random(0)
    steps, t = 0, time.time()
    while time.time() - t < 10:
        _, _, done, _, _ = env.step(rng.randrange(env.n_actions))
        steps += 1
        if done:
            env.reset()
    dt = time.time() - t
    say(f"코어 1개: 초당 판단 {steps / dt:,.0f}회 (게임 프레임 {steps * 4 / dt:,.0f})")
    n = os.cpu_count() or 1
    say(f"코어 {n}개를 다 쓰면 대략 초당 {steps / dt * max(1, n // 2):,.0f} ~ {steps / dt * n:,.0f}회 (예상)")

except Exception as e:
    import traceback
    say("시뮬레이션 측정 중 에러:")
    say(traceback.format_exc())

print(f"\n저장: {OUT}")
