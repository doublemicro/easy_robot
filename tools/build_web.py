"""웹(브라우저) 버전을 만든다 → web/ (Vercel에 그대로 올리는 정적 사이트) (v16).

    pip install pygbag
    python -m tools.build_web

1. 게임에 필요한 파일만 임시 폴더(stickfight_web_src)로 모은다
   (main.py, play.py, game/, ai/ (PyTorch 쓰는 학습 코드 빼고), models/ai.npz, assets/fonts/*.ttf)
   → 학습 결과(runs/), 원본 폰트, 테스트·도구는 안 들어감
2. pygbag로 빌드 (파이썬과 pygame을 WebAssembly로 돌리는 index.html + 게임 묶음 .apk)
3. 결과를 web/ 로 옮긴다

로컬에서 확인: python -m http.server -d web 8000 → 브라우저에서 http://localhost:8000
(파이썬·pygame 실행 파일은 pygbag CDN에서 받아 오므로 인터넷이 필요하다)
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 빌드용 임시 폴더는 시스템 임시 폴더에 (OneDrive 안에 두면 동기화가 파일을 잡고 있어 지우기가 실패함)
SRC = os.path.join(tempfile.gettempdir(), "stickfight_web_src")
OUT = os.path.join(ROOT, "web")
SKIP_AI = {"policy.py"}          # torch를 쓰는 학습용 (게임은 policy_np.py)


def copy(rel):
    dst = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, rel), dst)


def main():
    shutil.rmtree(SRC, ignore_errors=True)
    os.makedirs(SRC)
    for rel in ("main.py", "play.py", os.path.join("models", "ai.npz")):
        copy(rel)
    for p in glob.glob(os.path.join(ROOT, "game", "*.py")):
        copy(os.path.relpath(p, ROOT))
    for p in glob.glob(os.path.join(ROOT, "ai", "*.py")):
        if os.path.basename(p) not in SKIP_AI:
            copy(os.path.relpath(p, ROOT))
    for p in glob.glob(os.path.join(ROOT, "assets", "fonts", "*")):
        if os.path.isfile(p):
            copy(os.path.relpath(p, ROOT))
    cmd = [sys.executable, "-m", "pygbag", "--build", "--ume_block", "0", "--title", "Stick Fight",
           "--app_name", "stickfight", SRC]
    print(" ".join(cmd))
    env = dict(os.environ, PYTHONUTF8="1")       # 윈도우에서 pygbag가 한글 소스를 cp949로 읽다 깨지는 것 방지
    subprocess.run(cmd, check=True, env=env)
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(os.path.join(SRC, "build", "web"), OUT)
    for p in sorted(glob.glob(os.path.join(OUT, "*"))):
        print(f"  {os.path.relpath(p, ROOT)}  {os.path.getsize(p) // 1024} KB")


if __name__ == "__main__":
    main()
