# /// script
# dependencies = [
#     "numpy",
# ]
# ///
"""(v16) 웹(브라우저) 실행 진입점 — pygbag가 이 파일을 실행한다. PC에서는 play.py를 쓴다.

    python -m tools.build_web      # → web/ (Vercel에 올리는 정적 사이트)
"""
import asyncio

import play


async def main():
    app = play.App(play.parse_args([]))
    await app.run_async()


asyncio.run(main())
