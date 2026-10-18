"""게임에 넣는 한글 폰트를 실제로 쓰는 글자만 남겨 작게 만든다 (v16, 웹 배포용).

    python -m tools.subset_fonts

- 원본: assets/fonts/src/NanumGothic-Regular.ttf, NanumGothic-Bold.ttf (Google Fonts, OFL)
- 결과: assets/fonts/NanumGothic-Regular.ttf, NanumGothic-Bold.ttf (약 2MB → 수십 KB)
- 남기는 글자: 영문·숫자·기호(ASCII) + play.py / game/ / ai/ 소스에 나오는 모든 한글·특수문자.
  게임 글자를 바꿨으면 다시 실행한다.
"""
import glob
import os

from fontTools import subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.path.join(ROOT, "assets", "fonts")


def used_chars():
    chars = set(chr(c) for c in range(32, 127))
    files = [os.path.join(ROOT, "play.py")] + glob.glob(os.path.join(ROOT, "game", "*.py")) \
        + glob.glob(os.path.join(ROOT, "ai", "*.py"))
    for p in files:
        with open(p, encoding="utf-8") as fp:
            chars |= {ch for ch in fp.read() if ord(ch) > 126}
    return chars


def main():
    text = "".join(sorted(used_chars()))
    for name in ("NanumGothic-Regular.ttf", "NanumGothic-Bold.ttf"):
        src = os.path.join(FONTS, "src", name)
        out = os.path.join(FONTS, name)
        opts = subset.Options()
        opts.drop_tables += ["DSIG"]
        font = subset.load_font(src, opts)
        sub = subset.Subsetter(opts)
        sub.populate(text=text)
        sub.subset(font)
        subset.save_font(font, out, opts)
        print(f"{name}: {os.path.getsize(src) // 1024} KB → {os.path.getsize(out) // 1024} KB ({len(text)}자)")


if __name__ == "__main__":
    main()
