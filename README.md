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
│   ├── data/             ← 데이터 수집 & 캐싱
│   ├── indicators/       ← 기술적 지표 계산
│   └── report/           ← 대시보드
├── tests/                ← pytest 단위 테스트
└── data/                 ← (gitignore) raw·processed CSV
```

---

## 면책

이 시스템은 **정보 제공 목적**이며, 투자 결정과 그 결과는 본인 책임입니다.
자동 주문 기능은 포함되지 않으며, 모든 주문은 사용자가 증권사 앱에서 직접 수행합니다.
