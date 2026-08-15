# <h1 style="text-align: center;">⚡2025 전력사용량 예측 AI 경진대회</h1>
<hr>
<p style="text-align: center;">
    <a href="https://github.com/donghyun0518/dacon-power-consumption-xgboost-catboost-lightgbm/blob/main/%EC%A0%84%EB%A0%A5%20(1).pdf" target="_blank">
        <img src="https://github.com/donghyun0518/dacon-power-consumption-xgboost-catboost-lightgbm/blob/main/power_consuption_main.png" alt="Project Cover" style="width: 1000px; border: 1px solid #c9d1d9; border-radius: 8px;">
    </a>
</p>

[프로젝트 발표 자료](https://github.com/donghyun0518/dacon-power-consumption-xgboost-catboost-lightgbm/blob/main/%EC%A0%84%EB%A0%A5%20(1).pdf)

# 풍력발전량 예측 — BARAM 2026 (Public 49위 → **Private 18위**)

> 기상예보만으로 하루 앞 풍력발전량을 맞히는 문제.
> LightGBM + MLP + SeqCNN 3-way 앙상블에 **정산금 지표 구조 분석 → 데이터의 도메인 사실(터빈 정지) 발견 → 조건부 보정**을 얹었다.
> **Public 49위 → Private 18위 (+31계단).** Public 상위권 대부분이 Private에서 무너지는 동안(1위 → 16위) 내 점수 수축은 −0.0014로 상위권 중앙값(−0.0113)의 1/8 — 공개 리더보드에 맞추지 않고 **사전 등록한 검증 규칙**으로만 제출을 고른 결과다.
> 가장 큰 도약은 모델이 아니라 **데이터 안의 사실을 찾아낸 데서** 나왔고, 자체 감사로 **내 코드의 규정 위반을 찾아 점수를 깎아가며 고쳤다.**

<p align="center">
  <img src="docs/img/public_private_shakeup.png" width="820" alt="Public→Private 순위 변동">
</p>
<p align="center">
  <img src="docs/img/hero_score_timeline.png" width="720" alt="점수 타임라인">
</p>

---

## 1. 문제

| 항목 | 내용 |
|---|---|
| 대회 | 제3회 풍력발전량 예측 AI 경진대회 **BARAM 2026** (Dacon · KPX) |
| 대상 | 태백 가덕산 풍력단지 3개 KPX 그룹 (Vestas V126 ×N / Unison U136 ×5 등) |
| 입력 | day-ahead NWP: **LDAPS 16격자** (50 m·10 m 바람 등) + **GFS 9격자** (850 hPa 등), 학습기간 SCADA |
| 출력 | 2025년 **8,760시간 × 3그룹** 발전량(kWh) — 매일 D-1 13:00 KST에 발표된 예보로 D일 24시간 예측 |
| 학습 | 2022-01 ~ 2024-12 라벨 (26,304행; g3는 2023-준공이라 2년치) |
| 제약 | 자기회귀 불가(실측 미제공), 예측기준시점 이후 정보 일절 금지, 재분석자료(ERA5) 금지, 외부데이터는 공개·재현 가능해야 함 |

### 평가 지표 — 이게 이 대회의 절반이었다
```
Score = 0.5·(1 − NMAE) + 0.5·FICR          (실측 ≥ 설비용량 10% 시각만 채점)
FICR  = Σ 실측·price(e) / (4·Σ 실측),      e = |예측−실측| / 설비용량
price(e) = 4 (e ≤ 6%) / 3 (e ≤ 8%) / 0 (그 외)
```
FICR은 **KPX 재생에너지 발전량 예측제도의 정산금 산식** 그대로다. 오차 8% 이내에 들어가면 돈을 받고 아니면 0 — **계단형 지시함수 손실**이라 평균 오차를 줄이는 것과 밴드 안에 들어가는 것이 다른 문제가 된다. 프로젝트 후반의 거의 모든 의사결정이 이 구조에서 나왔다.

---

## 2. 결과

| | Public (40%) | Private (60%) |
|---|---|---|
| 리더보드 | 0.65826 · **49위** (1−NMAE 0.8773 / FICR 0.4393) | **0.65686 · 18위** (1−NMAE 0.8819 / FICR 0.4318) |
| 최종 파이프라인 `sub_g3hw106` (규정 준수 · 비트단위 재현 검증) | 0.65799 (1−NMAE 0.8772 / FICR 0.4387) | — |
| 베이스라인 (LightGBM 단일) | 0.6362 | — |
| 누적 개선 (Public) | **+0.0218** | |

### Public → Private: 왜 31계단을 올랐나
Public 상위 12팀 중 Private 25위 안에 남은 팀은 8팀, 그중 순위가 오른 팀은 3팀뿐이었다(Public 1위 → 16위, 2·4·10·11위 → 25위 밖). 상위권의 Public→Private 점수 수축 중앙값은 **−0.0113**, 나는 **−0.0014**.
이건 운이 아니라 설계다. 프로젝트 중반에 **공개 LB에서 최댓값을 고르면 Private를 잃는다**는 것을 정량화했고(§3-2: 분할 간 Public/Private 상관 −0.9999, 페어 비교 표준편차 0.0006~0.0009), 이후 모든 제출은 **사전 등록한 채택 문턱**(+0.0012)으로만 판정했다. 61회 제출 중 "LB 튜닝"성 제출은 조건부 보정 창을 정하는 3~4회의 **분해 프로브**뿐이며, 그것도 그룹·분기 단위 기여를 분리하는 용도였다.

<details>
<summary><b>점수 여정 (클릭)</b></summary>

| 단계 | 변경 | Public | Δ |
|---|---|---|---|
| SUB-001 | LightGBM(L1) + CF 타깃 + α/floor 후처리 | 0.6362 | — |
| SUB-002 | + CF 샘플가중 (채점 필터 반영) | 0.6399 | +0.0037 |
| SUB-008 | + PyTorch MLP 앙상블 | 0.6436 | +0.0037 |
| SUB-009 | + SeqCNN (24h 블록 dilated 1D-CNN) → **3-way** | 0.6478 | +0.0042 |
| SUB-013 | + **ECMWF IFS** 850/925 hPa (외부데이터, 직접 수집) | 0.6500 | +0.0022 |
| SUB-016 | 확장 ECMWF를 **g2 GBM 멤버에만** (멤버별 용량 제어) | 0.6504 | +0.0004 |
| SUB-021~025 | ★ **g3 터빈 정지 발견** → 계단형 보정 (Q1 ×0.88 / Q2 ×0.95) | 0.6570 | **+0.0067** |
| SUB-032 | 계절 캘리브레이션 (Q1 g1·g2 ×0.96) | 0.6583 | +0.0013 |
| — | ★ **자체 감사: ECMWF 런시각·보간 규정 위반 2건 적발·수정** | 0.6556 | **−0.0027 (감수)** |
| ecmem3 | + ECMWF 전용 4번째 멤버 (입력 정보원 분리) | 0.6568 | +0.0012 |
| **g3hw106** | ★ **g3 고풍속 게이트 ×1.06** (라벨 오염의 증상 보정) | **0.6580** | +0.0012 |
</details>

---

## 3. 접근

### 3-1. 파이프라인
```
LDAPS 16격자 + GFS 9격자 + (ECMWF IFS 직접수집)
        │  src/features.py  — 103개 피처: 풍속·풍향 sin/cos·ρ·ws³·시어·인접시각 램프·공간 산포
        ▼
 ┌─ LightGBM (L1, 3 seed) ─┐
 ├─ MLP (가중 L1, 3 seed) ──┼─ 등가중 평균 ─→ α·pred → floor → clip
 ├─ SeqCNN (24h 블록) ─────┤                    │
 └─ EC-only LightGBM (w=.15/.15/.12, 정보원 분리) ┘   ▼
                                    조건부 보정 (g3 정지 · g3 고풍속 · 계절)
```
- **타깃**: 이용률 CF = kWh / 설비용량. **샘플가중** `clip(0.5+1.5·CF, 0.5, 2)` — 채점이 고출력 시각만 보므로.
- **그룹별 구성**: g1 = 제공 피처(102) / g2 = GBM만 ECMWF 확장(148), NN은 112 / g3 = 112. 같은 데이터도 **어느 멤버에 주느냐**로 부호가 갈렸다(전 멤버 −0.0019 vs GBM만 +0.0004).
- **후처리**: α(1.06/1.04/1.01)·floor(0.18C)는 LOYO OOF에서 총점 최적으로 추정 — 채점 필터가 만드는 조건부 과소예측을 교정하는 스칼라 근사.

### 3-2. 검증 프로토콜 (실패에서 배워 세운 것)
로컬 CV와 리더보드가 **부호까지 뒤집힌 사례가 8건**이었다. 원인은 셋: (a) 2025가 고풍속 OOD 연도, (b) g3 라벨 2년치, (c) 후처리와 모델이 강결합. 그래서 규칙을 만들었다.

| 규칙 | 근거 |
|---|---|
| **matched-pair**만 신뢰: 프로덕션 실물 base에 변경 하나만 얹고 pooled post 고정, LOYO | 정보 추가 레버 3전 3승 (전이율 74~98%) |
| 채택 문턱 **3그룹환산 ≥ +0.0012 AND 고풍속(상위 30%) 부분점수 ≥ 0** | LB 프로브로 역산한 체계적 음성 오프셋 −0.0008 |
| GBM만으로 거르지 않는다 (3멤버 전부 재학습) | GBM-only 스크린에서 부호 반전 8건 |
| 사후 파라미터 선택 금지 — nested, 사전 등록한 판정 규칙 | Public/Private 상관 −0.9999: 공개 LB에서 최댓값 고르면 private를 잃는다 |
| **"모델 동역학 변경"은 로컬 CV로 판정 불가** (LB 0-3) → 정보 추가에 집중 | in-distribution CV가 OOD 외삽 변화를 못 잰다 |

### 3-3. ★ 가장 큰 도약: 데이터 안의 사실
학습기간 SCADA를 상태기계(정지=풍속>4 m/s & 출력≈0, 가동=출력>5%)로 읽으니 **g3 5기 중 wtg01이 2024-12-02부터 학습 종료까지 완전 정지**였고, 학습행의 **33%가 부분 정지 상태**였다.
- 2025 상반기 정지가 이어질 것으로 보고 **Q1 ×0.88 / Q2 ×0.95** 계단 보정 → **+0.0067** (다른 모든 레버의 10배). 3개 제출을 동시에 올려 **분기별 기여를 LB에서 분해**(Q1 +0.0046 / Q2 +0.0005 / H2 −0.0009)해서 창을 정했다.
- 정지 오염은 **고풍속 라벨을 4/5로 눌러** 모델에 과소예측을 학습시킨다. 마지막 주에 세 개의 독립 실험(밴드최적 결정규칙·가법 후처리·멤버스프레드)이 전부 "g3 고풍속대"를 가리켜 → **예보 허브풍속 ≥ 학습 q70 시각만 ×1.06** → +0.0012. 라벨 자체를 복원해 재학습하는 근원 치료는 폴드 간 충돌로 기각했다 — **작고, 양 폴드에서 검증되고, 게이트가 좁은 개입만 전이된다.**

### 3-4. 지표 구조에서 나온 결론들 (전부 실측)
- **"정확도를 팔아 FICR을 산다"는 모든 층에서 교환비 1:1** — 학습손실(pinball-τ), 결정규칙(Gneiting 최빈구간 argmax), 후처리 가족(α,β,floor), 선택 목적함수(κ). α/floor가 이미 총점 최적으로 그 거래를 수행 중이라 최적점의 한계교환비는 정의상 1:1. **점수는 실재하는 조건부 편향의 수리에서만 오른다.**
- **불확실성 정보는 점예측 파이프라인에서 점수화되지 않는다** — 앙상블 스프레드, 예보 개정폭(D-2런 vs D-3런 직접 수집), 저풍속 노출 등 5종 모두 진단에선 신호가 뚜렷(밴드 획득률 1.6배)했으나 점수 0. 분산은 조건부 중앙값을 안 바꾼다.
- **알고리즘 다양성 ≈ 0**: CatBoost/XGBoost/ExtraTrees 잔차상관 0.95~0.97 (기존 멤버 간 0.91). **입력 정보원 분리**만 상관을 낮췄다(EC-only 멤버 0.77~0.84).

---

## 4. 규정 준수 — 스스로 찾은 위반 2건
외부데이터(ECMWF)를 직접 수집하면서 **"예보가 생성·공개된 시각" 기준** 규정을 두 군데서 어긴 것을 자체 감사에서 발견했다.
1. **런시각 선택**: 병렬 다운로드의 완료 순서에 따라 `drop_duplicates(keep='first')`가 경계 시각에 **D-1 18Z 런(기준시점보다 14시간 뒤)** 을 택함 → 2025 시각의 17.0%에 영향. `ec_legal_rows()`로 런시각을 명시 계산해 배제.
2. **잔여 보간**: 1차 수정 후 남은 결측을 pandas `interpolate(limit=6)`이 채웠는데 **`limit`은 채우는 칸 수만 제한하고 앵커 거리는 제한하지 않는다** (`[0, NaN×26, 100]`에서 [1] = 3.70 = 100/27). 129시각 오염. `legal_interpolate()`로 앵커의 합법성까지 검사.

수정으로 Public **0.6583 → 0.6556 (−0.0027)** 을 감수했고, 이후 규정 안에서 다시 0.6580까지 회복했다. 최종 파이프라인은 `make_final_submission.py → make_sub_ecmem.py → make_g3hw.py` 로 **비트단위 재현**을 확인했다(3그룹 최대 |Δ| = 0.0 kWh). 프로덕션 경로 전수 점검(LDAPS/GFS 보간 앵커, ECMWF 런선택·보간, 결측 대체 통계, NN 스케일러, 예보 lead 참조)까지 완료. 상세: [`experiments/COMPLIANCE.md`](experiments/COMPLIANCE.md).

---

## 5. 재현
```bash
pip install -r requirements.txt          # Python 3.13 / CPU 학습 (torch CPU 빌드)
# 제공 데이터를 data/train, data/test 에 배치
python scripts/fetch_ecmwf_dayahead.py   # ECMWF IFS open data (Herbie) — D-2 18Z 런만
python scripts/make_final_submission.py  # 3-way 학습·예측 → outputs/cache/raw_test_pred.pkl
python scripts/make_sub_ecmem.py         # + EC 멤버 → sub_canonical.csv (재현 검산 내장)
python scripts/make_g3hw.py              # + g3 고풍속 게이트 → 최종 sub_g3hw106.csv
```
시드 고정(42/202/777), 결정론적 학습. 로컬 검증: `python -m src.train --stage cv`.

## 6. 저장소 구조
```
src/         features · leakfree(누수차단) · nn_model(MLP/SeqCNN, pinball 옵션) · train · score(공식 산식 미러) · scada
scripts/     make_*(제출 빌더) · fetch_*(외부데이터 수집) · exp_*(실험 100여 편)
experiments/ LOG.md(실험 일지, 190 KB) · submissions.md(제출 원장 40건) · COMPLIANCE.md(규정 소명) · results/(실험 출력 125건)
docs/        발표자료(PDF/PPTX) · 그림
```

## 7. 배운 것
- **지표를 먼저 분해하라.** FICR = 3·1{e≤8%} + 1·1{e≤6%}. 이 한 줄이 후처리·검증·레버 선택을 전부 바꿨다.
- **가장 큰 이득은 모델 밖에 있었다.** 터빈 정지는 로컬 CV로 절대 못 찾는다 — SCADA를 읽고, 대조군으로 검증하고, LB 프로브로 창을 잡았다.
- **검증 프로토콜은 실패를 먹고 자란다.** 8건의 부호 반전 → matched-pair·문턱·nested·사전등록. 이후 정보 추가 레버는 3전 3승, 그리고 **Public 49위 → Private 18위**.
- **공개 리더보드는 최적화 대상이 아니라 노이즈가 있는 계측기다.** 페어 비교 표준편차 0.0006~0.0009를 먼저 재고, 그 위에서만 판정한다.
- **점수를 깎더라도 규정은 지킨다.** 위반은 수집 단계가 아니라 *최종 피처에 들어간 값* 기준으로 검사해야 한다.

## License / 데이터
코드는 MIT. 대회 데이터는 재배포하지 않는다(Dacon 규정). ECMWF open data는 CC-BY-4.0.
