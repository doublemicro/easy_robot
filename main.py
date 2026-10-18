# /// script
# dependencies = [
#     "numpy",
# ]
# ///
"""(v16) 웹(브라우저) 실행 진입점 — pygbag가 이 파일을 실행한다. PC에서는 play.py를 쓴다.

    python -m tools.build_web      # → web/ (Vercel에 올리는 정적 사이트)

pygbag는 이 파일의 import 줄을 보고 필요한 패키지(pygame, numpy)를 미리 준비한다.
그래서 게임 코드(play.py)를 불러오기 전에 여기서 직접 import 해야 한다.
"""
import asyncio

import numpy  # noqa: F401  (pygbag가 미리 받게)
import pygame

pygame.init()

import play  # noqa: E402


async def main():
    app = play.App(play.parse_args([]))
    await app.run_async()


asyncio.run(main())
