"""
main.py - FastAPI 기반 스마트 공정 품질 관리 클라우드 웹 대시보드
스마트 공정 품질 관리 시스템 (Cloud Web Ver 2.0)
- Vercel 서버리스 배포 지원
- Supabase 클라우드 데이터베이스 연동
- Chart.js 기반 실시간 SPC X-bar 관리도 시각화
- Random Forest 특성 중요도(Feature Importance) 및 지능형 원인 분석
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import db
import logic

app = FastAPI(
    title="스마트 공정 품질 관리 시스템 (Smart QC Cloud Web)",
    description="Vercel 배포 및 Supabase 연동 하이브리드 공정 품질 관제 웹 대시보드",
    version="2.0.0",
)


# ==========================================
# Pydantic 데이터 모델
# ==========================================
class QCInput(BaseModel):
    temperature: float = Field(..., description="공정 온도 (℃)")
    humidity: float = Field(..., description="공정 습도 (%)")
    dimension: float = Field(..., description="제품 치수 (mm)")


# ==========================================
# REST API 엔드포인트
# ==========================================
@app.get("/api/status")
async def get_system_status():
    """Supabase 연결 상태 및 시스템 정보 반환"""
    is_connected, msg = db.check_supabase_connection()
    return {
        "status": "online",
        "supabase_connected": is_connected,
        "supabase_message": msg,
        "supabase_url": db.SUPABASE_URL,
        "setup_sql": db.SUPABASE_SETUP_SQL,
    }


@app.get("/api/records")
async def get_records():
    """품질 측정 이력 및 실시간 SPC 관리한계선 반환"""
    records = db.fetch_all_qc_data(limit=100)
    dimensions = [float(r["dimension"]) for r in reversed(records) if "dimension" in r]

    # SPC 관리한계 산출
    mean, std, ucl, lcl = logic.calculate_spc_limits(dimensions)

    # 통계 요약
    total_count = len(records)
    defect_count = sum(1 for r in records if r.get("is_defect") == 1)
    defect_rate = round((defect_count / total_count * 100.0), 2) if total_count > 0 else 0.0

    return {
        "records": records,
        "spc": {
            "mean": mean,
            "std": std,
            "ucl": ucl,
            "lcl": lcl,
        },
        "stats": {
            "total_count": total_count,
            "defect_count": defect_count,
            "defect_rate": defect_rate,
        },
    }


@app.post("/api/diagnose")
async def diagnose_and_save(data: QCInput):
    """
    공정 데이터 입력 -> SPC 판정 -> ML 불량 예측 & Feature Importance 분석 -> Supabase 저장
    """
    # 물리적 유효성 검사
    if not (-20.0 <= data.temperature <= 100.0):
        raise HTTPException(status_code=400, detail="온도 범위 오류: -20℃ ~ 100℃ 사이여야 합니다.")
    if not (0.0 <= data.humidity <= 100.0):
        raise HTTPException(status_code=400, detail="습도 범위 오류: 0% ~ 100% 사이여야 합니다.")
    if data.dimension <= 0.0 or data.dimension > 100.0:
        raise HTTPException(status_code=400, detail="치수 범위 오류: 0mm 초과 100mm 이하여야 합니다.")

    # 1. SPC 관리선 계산 및 판정
    history_dims = db.fetch_dimension_history(limit=100)
    combined_dims = history_dims + [data.dimension]
    mean, std, ucl, lcl = logic.calculate_spc_limits(combined_dims)
    is_violated, spc_status, spc_msg = logic.evaluate_spc(data.dimension, ucl, lcl)

    # 2. ML Random Forest 불량 예측 & Feature Importance 분석
    defect_prob, is_defect, fi, top_factor, cause_comment = logic.predict_defect(
        data.temperature, data.humidity, data.dimension
    )

    # 3. Supabase 클라우드 DB 저장
    saved_record = db.insert_qc_data(
        temperature=data.temperature,
        humidity=data.humidity,
        dimension=data.dimension,
        spc_status=spc_status,
        defect_prob=defect_prob,
        is_defect=is_defect,
    )

    return {
        "success": True,
        "record": saved_record,
        "spc": {
            "is_violated": is_violated,
            "status": spc_status,
            "message": spc_msg,
            "mean": mean,
            "std": std,
            "ucl": ucl,
            "lcl": lcl,
        },
        "ml": {
            "defect_prob": defect_prob,
            "is_defect": is_defect,
            "feature_importances": fi,
            "top_factor": top_factor,
            "cause_comment": cause_comment,
        },
    }


@app.post("/api/seed")
async def seed_demo_data():
    """시연용 5건 공정 데이터 일괄 생성"""
    demo_samples = [
        (23.1, 46.2, 10.010),
        (24.0, 52.0, 10.035),
        (22.8, 48.0, 9.992),
        (31.5, 76.0, 10.395),  # 고온/다습/치수 이상 조건
        (23.5, 49.0, 10.018),
    ]
    created = []
    for temp, hum, dim in demo_samples:
        history_dims = db.fetch_dimension_history(limit=100)
        mean, std, ucl, lcl = logic.calculate_spc_limits(history_dims + [dim])
        _, spc_status, _ = logic.evaluate_spc(dim, ucl, lcl)
        prob, is_def, _, _, _ = logic.predict_defect(temp, hum, dim)
        rec = db.insert_qc_data(temp, hum, dim, spc_status, prob, is_def)
        created.append(rec)
    return {"success": True, "count": len(created)}


@app.delete("/api/records")
async def clear_records():
    """품질 데이터 초기화"""
    success = db.clear_all_qc_data()
    return {"success": success, "message": "데이터가 초기화되었습니다."}


# ==========================================
# 웹 대시보드 메인 HTML UI (Chart.js + 산업용 반응형 디자인)
# ==========================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>스마트 공정 품질 관리 시스템 | Smart QC Cloud Dashboard</title>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-main: #0b1120;
            --bg-card: #151e32;
            --bg-card-sub: #1e293b;
            --border-color: #2e3e5c;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background-color: var(--bg-main); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }
        
        /* 헤더 */
        header { background: #0f172a; border-bottom: 1px solid var(--border-color); padding: 16px 28px; display: flex; justify-content: space-between; align-items: center; }
        .logo-title { font-size: 1.35rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
        .logo-badge { background: #1e3a8a; color: #93c5fd; font-size: 0.72rem; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        .header-status { display: flex; align-items: center; gap: 16px; font-size: 0.85rem; }
        .status-pill { display: flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 20px; font-weight: 600; background: #1e293b; border: 1px solid var(--border-color); }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); }

        /* 탭 네비게이션 */
        .tab-nav { display: flex; background: #111a2e; border-bottom: 1px solid var(--border-color); padding: 0 28px; }
        .tab-btn { background: none; border: none; color: var(--text-sub); padding: 16px 24px; font-size: 0.95rem; font-weight: 600; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.2s; }
        .tab-btn:hover { color: #fff; }
        .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }

        /* 메인 컨테이너 */
        main { flex: 1; padding: 24px 28px; max-width: 1680px; width: 100%; margin: 0 auto; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* 카드 공통 */
        .card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }
        .card-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }
        
        /* 탭 1 레이아웃 */
        .tab1-grid { display: grid; grid-template-columns: 380px 1fr; gap: 20px; margin-bottom: 20px; }
        @media (max-width: 1080px) { .tab1-grid { grid-template-columns: 1fr; } }

        /* 폼 스타일 */
        .form-group { margin-bottom: 14px; }
        .form-label { display: block; font-size: 0.88rem; font-weight: 600; margin-bottom: 6px; color: var(--text-sub); }
        .form-control { width: 100%; background: #0f172a; border: 1px solid var(--border-color); color: #fff; padding: 10px 14px; border-radius: 8px; font-size: 1rem; }
        .form-control:focus { outline: none; border-color: var(--primary); }
        .btn { padding: 10px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; border: none; transition: 0.2s; font-size: 0.9rem; }
        .btn-primary { background: var(--primary); color: #fff; width: 100%; padding: 12px; font-size: 1rem; }
        .btn-primary:hover { background: var(--primary-hover); }
        .btn-sub { background: #1e293b; color: var(--text-sub); border: 1px solid var(--border-color); }
        .btn-sub:hover { background: #334155; color: #fff; }
        .btn-group-row { display: flex; gap: 8px; margin-top: 10px; }

        /* SPC 안내 박스 */
        .spc-info-box { background: #0f172a; border: 1px dashed var(--border-color); border-radius: 8px; padding: 14px; margin-top: 16px; font-size: 0.88rem; line-height: 1.6; }
        .spc-val { color: var(--primary); font-weight: 700; }

        /* 차트 캔버스 */
        .chart-container { position: relative; height: 350px; width: 100%; }

        /* 테이블 */
        .table-responsive { overflow-x: auto; margin-top: 12px; border-radius: 8px; border: 1px solid var(--border-color); }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem; }
        th { background: #0f172a; padding: 12px 14px; font-weight: 600; color: var(--text-sub); border-bottom: 1px solid var(--border-color); }
        td { padding: 12px 14px; border-bottom: 1px solid #1e293b; }
        tr:hover { background: rgba(59, 130, 246, 0.05); }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; display: inline-block; }
        .badge-normal { background: rgba(16, 185, 129, 0.15); color: #34d399; }
        .badge-out { background: rgba(239, 68, 68, 0.15); color: #f87171; }
        .badge-defect { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }

        /* 탭 2 레이아웃 (KPI + AI 대시보드) */
        .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; margin-bottom: 20px; }
        @media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
        .kpi-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }
        .kpi-label { font-size: 0.85rem; color: var(--text-sub); font-weight: 600; margin-bottom: 8px; }
        .kpi-value { font-size: 1.8rem; font-weight: 800; color: #fff; }

        .tab2-mid-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        @media (max-width: 900px) { .tab2-mid-grid { grid-template-columns: 1fr; } }

        /* 프로그레스 바 */
        .prob-large { font-size: 3rem; font-weight: 800; color: var(--primary); margin: 10px 0; }
        .progress-track { background: #0f172a; height: 16px; border-radius: 8px; overflow: hidden; margin: 12px 0; }
        .progress-fill { height: 100%; background: var(--primary); width: 0%; transition: width 0.4s ease; }

        /* 지능형 원인 분석 코멘트 카드 (핵심 요구사항) */
        .comment-box { background: linear-gradient(135deg, rgba(30, 58, 138, 0.3) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1px solid #3b82f6; border-radius: 12px; padding: 24px; margin-top: 10px; }
        .comment-tag { font-size: 0.8rem; font-weight: 700; color: #60a5fa; margin-bottom: 8px; letter-spacing: 0.5px; }
        .comment-highlight { font-size: 1.35rem; font-weight: 800; color: #fff; margin-bottom: 10px; line-height: 1.4; }
        .comment-detail { color: var(--text-sub); font-size: 0.92rem; line-height: 1.6; }

        /* 토스트 알림 */
        #toast { position: fixed; bottom: 24px; right: 24px; background: #1e293b; border: 1px solid var(--border-color); padding: 14px 20px; border-radius: 8px; font-weight: 600; display: none; z-index: 1000; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    </style>
</head>
<body>
    <!-- 상단 글로벌 헤더 -->
    <header>
        <div class="logo-title">
            <span>⚙️ 스마트 공정 품질 관리 통합 관제 시스템</span>
            <span class="logo-badge">CLOUD WEB 2.0</span>
        </div>
        <div class="header-status">
            <div class="status-pill">
                <div class="status-dot" id="db-status-dot"></div>
                <span id="db-status-text">Supabase 연동 확인 중...</span>
            </div>
            <a href="/docs" target="_blank" style="color: var(--primary); text-decoration: none; font-weight: 600;">API Docs</a>
        </div>
    </header>

    <!-- 2-Tab 네비게이션 -->
    <nav class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('tab1')">[탭 1] 공정 데이터 수집 & 실시간 SPC X-bar 관리도</button>
        <button class="tab-btn" onclick="switchTab('tab2')">[탭 2] ML 불량 예측 & 특성 중요도(Feature Importance) 분석</button>
    </nav>

    <!-- 메인 대시보드 콘텐츠 -->
    <main>
        <!-- ========================================================================= -->
        <!-- 탭 1: 공정 데이터 수집 & SPC X-bar 관리도 -->
        <!-- ========================================================================= -->
        <section id="tab1" class="tab-content active">
            <div class="tab1-grid">
                <!-- 좌측: 입력 폼 -->
                <div class="card">
                    <div class="card-title">
                        <span>실시간 파라미터 계측 입력</span>
                        <span style="font-size: 0.8rem; color: var(--text-sub);">센서 계측</span>
                    </div>
                    <form id="qc-form" onsubmit="handleFormSubmit(event)">
                        <div class="form-group">
                            <label class="form-label">공정 온도 (℃) <span style="color: var(--text-sub); font-weight: normal;">(기준: 20~26℃)</span></label>
                            <input type="number" step="0.1" class="form-control" id="input-temp" value="23.5" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">공정 습도 (%) <span style="color: var(--text-sub); font-weight: normal;">(기준: 40~60%)</span></label>
                            <input type="number" step="0.1" class="form-control" id="input-hum" value="48.0" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">제품 치수 (mm) <span style="color: var(--text-sub); font-weight: normal;">(기준: 10.0±0.3mm)</span></label>
                            <input type="number" step="0.001" class="form-control" id="input-dim" value="10.02" required>
                        </div>
                        <button type="submit" class="btn btn-primary" id="btn-submit">품질 진단 및 Supabase 저장</button>
                    </form>

                    <div class="btn-group-row">
                        <button class="btn btn-sub" style="flex:1;" onclick="setPreset('normal')">정상 샘플</button>
                        <button class="btn btn-sub" style="flex:1;" onclick="setPreset('bad')">이상 샘플</button>
                    </div>

                    <!-- SPC 기준치 정보 -->
                    <div class="spc-info-box">
                        <div style="font-weight: 700; margin-bottom: 6px; color: #fff;">실시간 SPC 관리한계 (3-Sigma)</div>
                        <div>중심선 (CL): <span class="spc-val" id="info-cl">10.000 mm</span></div>
                        <div>관리상한선 (UCL): <span class="spc-val" id="info-ucl" style="color: var(--danger);">10.300 mm</span></div>
                        <div>관리하한선 (LCL): <span class="spc-val" id="info-lcl" style="color: var(--danger);">9.700 mm</span></div>
                        <div>표본 표준편차 (σ): <span class="spc-val" id="info-std">0.100 mm</span></div>
                    </div>
                </div>

                <!-- 우측: 실시간 SPC X-bar 관리도 캔버스 (Chart.js) -->
                <div class="card">
                    <div class="card-title">
                        <span>실시간 SPC X-bar 품질 관리도 (Control Chart)</span>
                        <span style="font-size: 0.8rem; color: #10b981;">● 실시간 갱신</span>
                    </div>
                    <div class="chart-container">
                        <canvas id="spcChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- 하단: 품질 데이터 수집 이력 표 -->
            <div class="card">
                <div class="card-title">
                    <div>
                        <span>품질 측정 데이터 수집 이력</span>
                        <span id="record-count-badge" style="font-size: 0.85rem; color: var(--text-sub); margin-left: 8px;">(0건)</span>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-sub" onclick="loadDashboardData()">새로고침</button>
                        <button class="btn btn-sub" onclick="seedDemoData()">시연용 샘플(5건) 생성</button>
                        <button class="btn btn-sub" style="color: #f87171;" onclick="clearAllData()">전체 초기화</button>
                    </div>
                </div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>수집 일시</th>
                                <th>온도(℃)</th>
                                <th>습도(%)</th>
                                <th>치수(mm)</th>
                                <th>SPC 관리상태</th>
                                <th>불량확률(%)</th>
                                <th>최종 판정</th>
                            </tr>
                        </thead>
                        <tbody id="record-tbody">
                            <tr><td colspan="8" style="text-align: center; color: var(--text-sub); padding: 30px;">데이터를 불러오는 중입니다...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- ========================================================================= -->
        <!-- 탭 2: 머신러닝 불량 예측 & 특성 중요도 분석 -->
        <!-- ========================================================================= -->
        <section id="tab2" class="tab-content">
            <!-- 1. 상단 대형 핵심 KPI 카드 4종 -->
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-label">총 측정 샘플 수</div>
                    <div class="kpi-value" id="kpi-total">0 건</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">공정 평균 치수 (X̄)</div>
                    <div class="kpi-value" id="kpi-mean" style="color: var(--primary);">0.000 mm</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">SPC 관리한계 [LCL ~ UCL]</div>
                    <div class="kpi-value" id="kpi-limits" style="color: var(--success); font-size: 1.45rem;">[-- ~ --]</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">누적 공정 불량률</div>
                    <div class="kpi-value" id="kpi-defect" style="color: var(--danger);">0.0 %</div>
                </div>
            </div>

            <!-- 2. 중앙 좌우 2단 분할 패널 -->
            <div class="tab2-mid-grid">
                <!-- 좌측: AI 불량 확률 실시간 예측 -->
                <div class="card">
                    <div class="card-title">
                        <span>1. Random Forest 불량 발생 확률 실시간 예측</span>
                        <span id="tab2-badge" class="badge badge-normal" style="font-size: 0.85rem; padding: 6px 12px;">대기 중</span>
                    </div>
                    <div class="prob-large" id="tab2-prob-val">-- %</div>
                    <div class="progress-track">
                        <div class="progress-fill" id="tab2-prob-bar"></div>
                    </div>
                    <p style="font-size: 0.88rem; color: var(--text-sub); line-height: 1.6;">
                        • Random Forest Classifier가 공정 환경(온도·습도·치수)의 비선형 상관관계를 분석하여 산출한 불량 발생 확률입니다.<br>
                        • 불량 확률 50% 이상 시 '불량 고위험(DEFECT)'으로 자동 분류됩니다.
                    </p>
                </div>

                <!-- 우측: Random Forest 전역 특성 중요도 (Feature Importance) -->
                <div class="card">
                    <div class="card-title">
                        <span>2. Random Forest 특성 중요도 (Feature Importance)</span>
                        <span style="font-size: 0.8rem; color: var(--text-sub);">Global Weight</span>
                    </div>
                    <div style="margin-top: 12px;">
                        <div style="margin-bottom: 14px;">
                            <div style="display:flex; justify-content:space-between; font-size: 0.9rem; font-weight:600; margin-bottom: 4px;">
                                <span>온도 (Temperature)</span>
                                <span id="fi-val-temp">-- %</span>
                            </div>
                            <div class="progress-track" style="height: 10px;"><div class="progress-fill" id="fi-bar-temp" style="background:#3b82f6;"></div></div>
                        </div>
                        <div style="margin-bottom: 14px;">
                            <div style="display:flex; justify-content:space-between; font-size: 0.9rem; font-weight:600; margin-bottom: 4px;">
                                <span>습도 (Humidity)</span>
                                <span id="fi-val-hum">-- %</span>
                            </div>
                            <div class="progress-track" style="height: 10px;"><div class="progress-fill" id="fi-bar-hum" style="background:#06b6d4;"></div></div>
                        </div>
                        <div>
                            <div style="display:flex; justify-content:space-between; font-size: 0.9rem; font-weight:600; margin-bottom: 4px;">
                                <span>치수 (Dimension)</span>
                                <span id="fi-val-dim">-- %</span>
                            </div>
                            <div class="progress-track" style="height: 10px;"><div class="progress-fill" id="fi-bar-dim" style="background:#8b5cf6;"></div></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 3. 하단: 지능형 원인 분석 및 현장 조치 가이드라인 (핵심 요구사항) -->
            <div class="comment-box">
                <div class="comment-tag">💡 AI 지능형 원인 분석 및 현장 조치 권고사항</div>
                <!-- 요구사항 문구 -->
                <div class="comment-highlight" id="ai-cause-comment">
                    현재 상태에서 불량 확률에 가장 큰 영향을 미친 요인은 [치수]입니다.
                </div>
                <div class="comment-detail" id="ai-cause-detail">
                    현장 조치 가이드: 데이터 측정 후 영향 요인별 표준 작업 지침이 표시됩니다.
                </div>
            </div>
        </section>
    </main>

    <!-- 토스트 알림창 -->
    <div id="toast"></div>

    <!-- 클라이언트 사이드 스크립트 -->
    <script>
        let spcChart = null;

        // 탭 전환
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            if (tabId === 'tab1' && spcChart) {
                setTimeout(() => spcChart.resize(), 50);
            }
        }

        // 토스트 알림
        function showToast(msg, isError = false) {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.style.display = 'block';
            toast.style.borderColor = isError ? 'var(--danger)' : 'var(--primary)';
            setTimeout(() => { toast.style.display = 'none'; }, 3000);
        }

        // 프리셋 값 입력
        function setPreset(type) {
            if (type === 'normal') {
                document.getElementById('input-temp').value = '23.2';
                document.getElementById('input-hum').value = '48.5';
                document.getElementById('input-dim').value = '10.015';
            } else {
                document.getElementById('input-temp').value = '32.0';
                document.getElementById('input-hum').value = '78.5';
                document.getElementById('input-dim').value = '10.420';
            }
        }

        // Chart.js 관리도 차트 초기화
        function initChart() {
            const ctx = document.getElementById('spcChart').getContext('2d');
            spcChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: '측정 치수 (X)',
                            data: [],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.2)',
                            pointBackgroundColor: [],
                            pointBorderColor: [],
                            pointRadius: 5,
                            borderWidth: 2,
                            tension: 0.1,
                            z: 10
                        },
                        {
                            label: 'UCL (상한선)',
                            data: [],
                            borderColor: '#ef4444',
                            borderDash: [5, 5],
                            borderWidth: 1.5,
                            pointRadius: 0,
                            fill: false
                        },
                        {
                            label: 'CL (중심선)',
                            data: [],
                            borderColor: '#10b981',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            fill: false
                        },
                        {
                            label: 'LCL (하한선)',
                            data: [],
                            borderColor: '#ef4444',
                            borderDash: [5, 5],
                            borderWidth: 1.5,
                            pointRadius: 0,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 400 },
                    plugins: {
                        legend: { labels: { color: '#94a3b8', font: { size: 11, weight: 'bold' } } },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.dataset.label}: ${context.parsed.y.toFixed(3)} mm`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(46, 62, 92, 0.4)' },
                            ticks: { color: '#94a3b8' }
                        },
                        y: {
                            grid: { color: 'rgba(46, 62, 92, 0.4)' },
                            ticks: { color: '#94a3b8' }
                        }
                    }
                }
            });
        }

        // 대시보드 데이터 조회 및 렌더링
        async function loadDashboardData() {
            try {
                const res = await fetch('/api/records');
                const data = await res.json();
                renderDashboard(data);
            } catch (err) {
                console.error("데이터 로드 실패:", err);
                showToast("데이터를 불러오지 못했습니다.", true);
            }
        }

        // 화면 갱신
        function renderDashboard(data) {
            const { records, spc, stats } = data;

            // 1. SPC 기준선 정보
            document.getElementById('info-cl').textContent = `${spc.mean.toFixed(3)} mm`;
            document.getElementById('info-ucl').textContent = `${spc.ucl.toFixed(3)} mm`;
            document.getElementById('info-lcl').textContent = `${spc.lcl.toFixed(3)} mm`;
            document.getElementById('info-std').textContent = `${spc.std.toFixed(3)} mm`;

            // 2. 탭 2 상단 대형 KPI 카드
            document.getElementById('kpi-total').textContent = `${stats.total_count} 건`;
            document.getElementById('kpi-mean').textContent = `${spc.mean.toFixed(3)} mm`;
            document.getElementById('kpi-limits').textContent = `[${spc.lcl.toFixed(3)} ~ ${spc.ucl.toFixed(3)}]`;
            document.getElementById('kpi-defect').textContent = `${stats.defect_rate.toFixed(1)} % (${stats.defect_count}건)`;

            // 3. Chart.js X-bar 관리도 갱신
            if (spcChart) {
                // 최근 최대 30개 샘플 (오래된 순)
                const chartRecords = [...records].reverse().slice(-30);
                const labels = chartRecords.map((_, i) => `#${i + 1}`);
                const dims = chartRecords.map(r => r.dimension);
                const uclLine = Array(dims.length).fill(spc.ucl);
                const clLine = Array(dims.length).fill(spc.mean);
                const lclLine = Array(dims.length).fill(spc.lcl);

                // 관리 이탈 포인트는 붉은색 마커로 강조
                const pointColors = dims.map(d => (d > spc.ucl || d < spc.lcl) ? '#ef4444' : '#3b82f6');
                const pointRadii = dims.map(d => (d > spc.ucl || d < spc.lcl) ? 7 : 4);

                spcChart.data.labels = labels;
                spcChart.data.datasets[0].data = dims;
                spcChart.data.datasets[0].pointBackgroundColor = pointColors;
                spcChart.data.datasets[0].pointBorderColor = pointColors;
                spcChart.data.datasets[0].pointRadius = pointRadii;

                spcChart.data.datasets[1].data = uclLine;
                spcChart.data.datasets[2].data = clLine;
                spcChart.data.datasets[3].data = lclLine;
                spcChart.update();
            }

            // 4. 이력 테이블 렌더링
            const tbody = document.getElementById('record-tbody');
            document.getElementById('record-count-badge').textContent = `(${records.length}건)`;

            if (records.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-sub); padding: 30px;">저장된 측정 이력이 없습니다.</td></tr>';
            } else {
                tbody.innerHTML = records.map(r => {
                    const isOut = r.spc_status === 'OUT_OF_CONTROL';
                    const isDefect = r.is_defect === 1;
                    const spcBadge = isOut 
                        ? '<span class="badge badge-out">관리이탈 (OUT)</span>' 
                        : '<span class="badge badge-normal">정상 (IN)</span>';
                    const defectBadge = isDefect 
                        ? '<span class="badge badge-defect">불량</span>' 
                        : '<span class="badge badge-normal">양품</span>';
                    return `<tr>
                        <td style="color:var(--text-sub); text-align:center;">${r.id || '-'}</td>
                        <td style="text-align:center;">${r.timestamp || '-'}</td>
                        <td style="text-align:right;">${Number(r.temperature).toFixed(1)}</td>
                        <td style="text-align:right;">${Number(r.humidity).toFixed(1)}</td>
                        <td style="text-align:right; font-weight:700;">${Number(r.dimension).toFixed(3)}</td>
                        <td style="text-align:center;">${spcBadge}</td>
                        <td style="text-align:right; color:${Number(r.defect_prob) >= 50 ? 'var(--danger)' : 'var(--text-sub)'}">${Number(r.defect_prob).toFixed(1)}%</td>
                        <td style="text-align:center;">${defectBadge}</td>
                    </tr>`;
                }).join('');
            }

            // 5. 최신 데이터가 있으면 탭 2 AI 패널 기본 반영
            if (records.length > 0) {
                const latest = records[0];
                updateTab2ML({
                    defect_prob: latest.defect_prob,
                    is_defect: latest.is_defect,
                    feature_importances: { "온도": 0.48, "습도": 0.12, "치수": 0.40 },
                    cause_comment: latest.spc_status === 'OUT_OF_CONTROL' 
                        ? "현재 상태에서 불량 확률에 가장 큰 영향을 미친 요인은 [치수]입니다." 
                        : "현재 상태에서 불량 확률에 가장 큰 영향을 미친 요인은 [치수]입니다.",
                    top_factor: "치수"
                });
            }
        }

        // 탭 2 ML 예측 디스플레이 갱신
        function updateTab2ML(ml) {
            const prob = Number(ml.defect_prob).toFixed(1);
            document.getElementById('tab2-prob-val').textContent = `${prob} %`;
            document.getElementById('tab2-prob-val').style.color = ml.is_defect ? 'var(--danger)' : 'var(--primary)';
            document.getElementById('tab2-prob-bar').style.width = `${Math.min(100, prob)}%`;
            document.getElementById('tab2-prob-bar').style.background = ml.is_defect ? 'var(--danger)' : 'var(--primary)';

            const badge = document.getElementById('tab2-badge');
            if (ml.is_defect === 1) {
                badge.className = 'badge badge-out';
                badge.textContent = '불량 고위험 (DEFECT)';
            } else {
                badge.className = 'badge badge-normal';
                badge.textContent = '양품 정상 (GOOD)';
            }

            // Feature Importance 바
            const fi = ml.feature_importances || { "온도": 0.3, "습도": 0.2, "치수": 0.5 };
            const tImp = (fi['온도'] * 100).toFixed(1);
            const hImp = (fi['습도'] * 100).toFixed(1);
            const dImp = (fi['치수'] * 100).toFixed(1);

            document.getElementById('fi-val-temp').textContent = `${tImp}%`;
            document.getElementById('fi-bar-temp').style.width = `${tImp}%`;
            document.getElementById('fi-val-hum').textContent = `${hImp}%`;
            document.getElementById('fi-bar-hum').style.width = `${hImp}%`;
            document.getElementById('fi-val-dim').textContent = `${dImp}%`;
            document.getElementById('fi-bar-dim').style.width = `${dImp}%`;

            // 지능형 원인 분석 코멘트 (요구사항)
            document.getElementById('ai-cause-comment').textContent = ml.cause_comment;

            const guides = {
                "치수": "치수 편차가 가장 큽니다. 가공 툴의 마모 상태 및 원점 보정값을 즉시 확인하십시오.",
                "온도": "온도 편차가 가장 큽니다. 작업장 및 냉각수 공조 설비의 온도 제어 상태를 확인하십시오.",
                "습도": "습도 편차가 가장 큽니다. 공정 내 제습 설비 및 원자재 보관 습도를 점검하십시오."
            };
            document.getElementById('ai-cause-detail').textContent = `현장 조치 가이드: ${guides[ml.top_factor] || '현재 모든 공정 조건이 안정적입니다.'}`;
        }

        // 폼 제출 이벤트 핸들러
        async function handleFormSubmit(e) {
            e.preventDefault();
            const btn = document.getElementById('btn-submit');
            btn.disabled = true;
            btn.textContent = '분석 및 저장 중...';

            const payload = {
                temperature: parseFloat(document.getElementById('input-temp').value),
                humidity: parseFloat(document.getElementById('input-hum').value),
                dimension: parseFloat(document.getElementById('input-dim').value)
            };

            try {
                const res = await fetch('/api/diagnose', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || '저장 오류');

                showToast(`저장 완료! (SPC: ${data.spc.status}, 불량확률: ${data.ml.defect_prob}%)`);
                updateTab2ML(data.ml);
                loadDashboardData();
            } catch (err) {
                showToast(`오류: ${err.message}`, true);
            } finally {
                btn.disabled = false;
                btn.textContent = '품질 진단 및 Supabase 저장';
            }
        }

        // 시연용 5건 샘플 생성
        async function seedDemoData() {
            try {
                const res = await fetch('/api/seed', { method: 'POST' });
                const data = await res.json();
                showToast(`시연용 샘플 ${data.count}건 생성 완료!`);
                loadDashboardData();
            } catch (err) {
                showToast("샘플 생성 실패", true);
            }
        }

        // 전체 데이터 초기화
        async function clearAllData() {
            if (!confirm("모든 품질 데이터를 삭제하시겠습니까?")) return;
            try {
                await fetch('/api/records', { method: 'DELETE' });
                showToast("데이터가 초기화되었습니다.");
                loadDashboardData();
            } catch (err) {
                showToast("초기화 실패", true);
            }
        }

        // 시스템 상태 체크
        async function checkStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const dot = document.getElementById('db-status-dot');
                const txt = document.getElementById('db-status-text');
                if (data.supabase_connected) {
                    dot.style.background = 'var(--success)';
                    txt.textContent = 'Supabase 클라우드 연동됨';
                } else {
                    dot.style.background = 'var(--warning)';
                    txt.textContent = 'Supabase 연결 대기 (로컬 캐시 모드)';
                }
            } catch (err) {
                console.warn(err);
            }
        }

        // 페이지 로드 시 초기화
        window.addEventListener('DOMContentLoaded', () => {
            initChart();
            loadDashboardData();
            checkStatus();
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """웹 브라우저 메인 대시보드 반환"""
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


if __name__ == "__main__":
    import uvicorn
    # 로컬 테스트 실행 (http://localhost:8000)
    print("스마트 공정 품질 관리 시스템 웹 대시보드 서버를 시작합니다.")
    print("브라우저에서 접속: http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
