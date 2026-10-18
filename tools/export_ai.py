"""학습된 체크포인트를 게임에 넣을 AI 파일로 내보낸다 (v12).

    python -m tools.export_ai runs/v11_league/latest.pt              # → models/ai.pt
    python -m tools.export_ai runs/v11_league/best_hard.pt --out models/ai.pt

- 신경망 가중치와 설정만 남기고 옵티마이저·셀프플레이 풀 등 학습용 데이터는 뺀다 (파일이 작아짐).
- 어디서 왔는지(원본 경로, 학습 판단 수, 단계, 저장 시각)를 같이 적어 둔다.
- 게임(play.py)의 AI 대전은 models/ai.npz를 쓴다 (v16: numpy로 돌려서 PyTorch 없이도, 웹에서도 됨).
  ai.pt도 같이 만든다 (eval_ai 등 PyTorch 도구용). 규칙을 바꿔 다시 학습했으면 이걸로 교체하면 된다.
"""
import argparse
import os
import time

import torch

from ai.policy import load_net
from ai.policy_np import export_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=os.path.join("models", "ai.pt"))
    a = ap.parse_args()
    net, ck = load_net(a.ckpt)           # 지금 게임과 관측/행동 크기가 다르면 여기서 막힌다
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "net_config": net.config(),
                "steps": ck.get("steps"), "stage": ck.get("stage"),
                "time": ck.get("time"), "source": a.ckpt,
                "exported": time.strftime("%Y-%m-%d %H:%M:%S")}, a.out)
    npz = os.path.splitext(a.out)[0] + ".npz"
    export_npz(net, npz)
    print(f"{a.ckpt} ({ck.get('steps', 0):,} 판단) → {a.out} ({os.path.getsize(a.out) / 1024:.0f} KB), "
          f"{npz} ({os.path.getsize(npz) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
