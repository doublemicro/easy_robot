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


# (v18) 화면 선명하게: pygbag 기본은 960x540 그림을 창 크기에 맞춰 늘려서(소수 배율) 흐릿하다.
# → 캔버스 표시 크기를 CSS(!important)로 덮어쓴다 (pygbag의 크기 조절은 그대로 두되 결과만 바꿈). 실제 화면 픽셀 기준으로 정수 배율(1배, 2배, ...)만 쓰고,
#   2배 이상은 픽셀을 그대로 키운다(pixelated) → 창이 커져도 흐려지지 않음. 창이 960x540보다 작을 때만 줄여서 맞춤.
CANVAS_JS = """
<style>
  html, body { background: #07080c !important; overflow: hidden; }
  #canvas { position: fixed !important; left: 50% !important; top: 50% !important;
            margin: 0 !important; transform: translate(-50%, -50%);
            width: var(--gw, 960px) !important; height: var(--gh, 540px) !important; }
  #canvas:focus { outline: none; }
  #canvas.crisp { image-rendering: pixelated; image-rendering: crisp-edges; }
</style>
<script>
(function () {
  const GW = __GW__, GH = __GH__;
  function fit() {
    const dpr = window.devicePixelRatio || 1;
    const k = Math.floor(Math.min(window.innerWidth * dpr / GW, window.innerHeight * dpr / GH));
    let w, h;
    if (k >= 1) { w = GW * k / dpr; h = GH * k / dpr; }      // 화면 픽셀 정수 배
    else {                                                   // 창이 더 작음 → 맞춰 줄임
      const s = Math.min(window.innerWidth / GW, window.innerHeight / GH);
      w = GW * s; h = GH * s;
    }
    document.documentElement.style.setProperty("--gw", w + "px");
    document.documentElement.style.setProperty("--gh", h + "px");
    const c = document.getElementById("canvas");
    if (c) c.classList.toggle("crisp", k >= 2);
  }
  window.addEventListener("resize", fit);
  setInterval(fit, 500);       // 게임이 캔버스를 처음 만들 때·모니터를 옮길 때(배율 변경)도 맞춘다
  fit();
})();
</script>
"""


def patch_index(path):
    from game.render import SCREEN_H, SCREEN_W
    with open(path, encoding="utf-8") as fp:
        html = fp.read()
    html = html.replace('platform.document.body.style.background = "#7f7f7f"',
                        'platform.document.body.style.background = "#07080c"')
    html = html.replace("</body>", CANVAS_JS.replace("__GW__", str(SCREEN_W)).replace("__GH__", str(SCREEN_H)) + "</body>")
    assert "#07080c" in html and "const GW" in html, "pygbag index.html 형식이 바뀜"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(html)


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
    # web/ 폴더 자체는 지우지 않고 안의 파일만 바꾼다 (OneDrive가 폴더를 잡고 있으면 폴더 삭제가 실패함)
    os.makedirs(OUT, exist_ok=True)
    for p in glob.glob(os.path.join(OUT, "*")):
        if os.path.isfile(p):
            os.remove(p)
    shutil.copytree(os.path.join(SRC, "build", "web"), OUT, dirs_exist_ok=True)
    patch_index(os.path.join(OUT, "index.html"))
    for p in sorted(glob.glob(os.path.join(OUT, "*"))):
        print(f"  {os.path.relpath(p, ROOT)}  {os.path.getsize(p) // 1024} KB")


if __name__ == "__main__":
    main()
