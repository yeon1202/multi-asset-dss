# 다자산 의사결정 지원 시스템 (Multi-Asset DSS)

> 자동 주문 X · **분석·추천만** 하는 개인 투자 어드바이저

상세 사양은 [PROJECT_SPEC.md](PROJECT_SPEC.md) 참조.

---

## Phase 1 — 다자산 모니터

한국·미국 ETF + 금 + 채권 + 달러 6개 자산의 가격을 수집해
RSI·이동평균·변동성과 함께 Streamlit 대시보드로 보여줍니다.

### 자산 유니버스 (Phase 1)
| 코드 | 종목명 |
|---|---|
| 069500 | KODEX 200 |
| 360750 | TIGER 미국S&P500 |
| 133690 | TIGER 미국나스닥100 |
| 148070 | KOSEF 국고채10년 |
| 132030 | KODEX 골드선물(H) |
| 261240 | KODEX 미국달러선물 |

## Phase 4 — 포트폴리오 최적화 + 일일 리포트

기술적 모멘텀(Phase 1) + 레짐 적합도(Phase 3) → 통합 점수 → 마코위츠 평균-분산 최적화 → Half-Kelly → 자동 Markdown 리포트.

### 추가 의존성
- **PyPortfolioOpt** (cvxpy backend) — 마코위츠 평균-분산
- Phase 3 의 ECOS / FRED 키 필요 (레짐 입력)

### CLI 실행
```powershell
python -m src.scripts.run_daily_report
```
결과: `reports/YYYY-MM-DD.md` — 사람이 바로 읽을 수 있는 일일 리포트.

### 핵심 알고리즘
1. `composite_score = 0.5·모멘텀_점수 + 0.5·레짐_적합도` (자산별 0-100)
2. `mu = (composite - 50) / 50 × max_expected_return` (선형 매핑, 자산별)
3. `EfficientFrontier(mu, cov).max_sharpe()` (제약: 자산당 ≤ 40%, ≥ 0%)
4. `kelly = (port_μ - r_f) / port_σ²`, 적용 = `0.5 · kelly` (Half-Kelly)
5. `위험자산 비중 = optimizer_weights × kelly_scale`, 나머지 = 현금 (최소 5%)

---

## Phase 3 — 시장 국면(레짐) 판단

거시 데이터(한국 + 미국) → Risk-On / Risk-Off / Neutral 룰 기반 분류 → 자산별 적합도 점수.

### 추가 의존성
- **ECOS API 키 필요** (한국은행 경제통계시스템, 무료) — [ecos.bok.or.kr/api](https://ecos.bok.or.kr/api)
- **FRED API 키 필요** (St. Louis Fed, 무료) — [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- `.env` 에 `ECOS_API_KEY=...`, `FRED_API_KEY=...` 추가

### 수집 지표
- **ECOS**: 한국 기준금리, 원/달러 환율, CPI
- **FRED**: VIX (공포지수), 미국 10Y/2Y 국채금리, 장단기 스프레드, 하이일드 OAS

### CLI 실행
```powershell
python -m src.scripts.run_regime_analysis
```
결과: `reports/regime_YYYY-MM-DD.csv` + 일별 히스토리.

---

## Phase 2 — 펀더멘털 스코어링

KOSPI 시총 상위 30종목의 DART 재무제표 + pykrx 시가총액을 결합해
PER · PBR · ROE · 영업이익률 · 부채비율 5지표 → 0~100 종합 점수 산출.

### 추가 의존성
- **DART API 키 필요** — [opendart.fss.or.kr](https://opendart.fss.or.kr) 에서 인증키 무료 발급
- 키 등록: 프로젝트 루트에 `.env` 파일 만들고 `DART_API_KEY=발급받은_40자리_키`

### CLI 실행 (대시보드 없이)
```powershell
python -m src.scripts.run_fundamental_scoring
python -m src.scripts.run_fundamental_scoring --year 2024 --top 20
```
결과: `reports/fundamental_YYYY-MM-DD.csv`

---

## 빠른 시작

### 1. 가상환경 만들기 (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> `Activate.ps1` 실행이 막히면 한 번만:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 2. 의존성 설치
```powershell
pip install -r requirements.txt
```

### 3. 단위 테스트 실행
```powershell
pytest -v
```

### 4. 대시보드 실행
```powershell
streamlit run src/report/dashboard.py
```
실행 후 브라우저가 자동으로 열리며 `http://localhost:8501` 로 접속됩니다.

---

## 디렉토리 구조

```
주식/
├── PROJECT_SPEC.md       ← 전체 사양 (이거 먼저 읽기)
├── README.md             ← 이 파일
├── requirements.txt
├── .env.example
├── config/
│   ├── universe.yaml     ← 추적할 자산 리스트
│   └── thresholds.yaml   ← RSI 등 임계값
├── src/
│   ├── data/             ← 데이터 수집 & 캐싱 (가격·DART·시총·ECOS·FRED)
│   ├── indicators/       ← 기술적 지표 + 펀더멘털 비율
│   ├── regime/           ← Phase 3 레짐 분류기
│   ├── scoring/          ← 0-100 종합 점수 + 자산 적합도
│   ├── scripts/          ← CLI 진입점
│   └── report/           ← 대시보드
├── tests/                ← pytest 단위 테스트
└── data/                 ← (gitignore) raw·processed CSV
```

---

## 면책

이 시스템은 **정보 제공 목적**이며, 투자 결정과 그 결과는 본인 책임입니다.
자동 주문 기능은 포함되지 않으며, 모든 주문은 사용자가 증권사 앱에서 직접 수행합니다.
