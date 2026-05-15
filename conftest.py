"""
pytest 설정 — 프로젝트 루트를 sys.path에 추가해서 `from src.xxx import yyy` 가 동작하게 함.
conftest.py 는 pytest가 자동으로 인식하는 특수 파일.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
