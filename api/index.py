"""
api/index.py - Vercel Serverless Function 진입점
"""

import sys
import os

# 프로젝트 루트 디렉토리를 파이썬 모듈 검색 경로에 추가
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import app

# Vercel이 'app' 객체를 ASGI 핸들러로 자동 인식합니다.
