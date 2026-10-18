"""AI 쪽: 학습 환경(env), 반응 지연(perception), AI가 보는 정보(obs), 규칙 기반 봇(bots).
게임 본체(game)를 가져다 쓴다."""
from .env import StickFightEnv
from .obs import OBS_DIM, build_obs
from .perception import Perception, Reaction, pick_delay
