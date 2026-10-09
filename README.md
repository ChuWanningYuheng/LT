# Эмпирическая проверка трудовой теории стоимости

Воспроизводимый расчёт: объясняет ли вертикально интегрированный труд отраслевые цены **лучше альтернатив** (цен производства, других «базисов стоимости», случайных базисов) по метрикам, устойчивым к известной критике (эффект масштаба, круговость редукции через зарплаты, асимметрия трудового и товарных базисов, неразличимость теорий издержек).

* Итоговый отчёт: [`report/REPORT_FINAL.md`](report/REPORT_FINAL.md) (кратко — [`report/SUMMARY.md`](report/SUMMARY.md)). Отчёты итераций: [`report/REPORT.md`](report/REPORT.md), [`report/REPORT_v2.md`](report/REPORT_v2.md), [`report/REPORT_v3.md`](report/REPORT_v3.md)
* Обзор литературы: [`report/literature.md`](report/literature.md)
* Журнал допущений: [`ASSUMPTIONS.md`](ASSUMPTIONS.md)
* Все сгенерированные таблицы: [`results/report_tables.md`](results/report_tables.md), графики: `results/figures/`
* Метрики по годам без плацебо (импорт по цене, подвыборки all/core): `results/tables/metrics_no_placebo.csv.gz`; тесты T1–T4: `results/tables/dyn_T*_{main,bea}.csv`; валидация: `results/tables/validation_*.csv`. Полные метрики с плацебо (~130 МБ) не хранятся, их создаёт `lts.levels`.

**Главный результат.** Стандартная близость стоимостей и цен воспроизводится (методика Zachariah 2006: MAWD 0,07 для Германии, 0,11 для США), но создаётся в основном прокси «оплата труда», отбором отраслей и асимметричным сравнением с другими базисами. В симметричной постановке (замкнутая система, рабочая сила воспроизводится корзиной) труд — типичный, а не особенный базис. В динамике и вне выборки он лучше физических базисов и «чужих» часов, но не лучше цен производства с фактическими зарплатами и произвольных номинальных базисов. Подробности — в отчёте.

## Данные

| Источник | Что берём | Годы | URL (скрипт `src/lts/download.py`) | Лицензия |
|---|---|---|---|---|
| Eurostat FIGARO, издание 2026 (`naio_10_fcp_ii1..4`) | межстрановые таблицы IO «отрасль × отрасль», 50 стран + остальной мир, 64 отрасли, текущие цены, млн евро | 2010–2024 | `https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/naio_10_fcp_iiN?format=TSV&compressed=true` | политика повторного использования Eurostat (CC BY 4.0) |
| OECD National Accounts, Table 6 (`DSD_NAMAIN10@DF_TABLE6`) | выпуск, D1, P51C (CFC), B2A3G/N, дефляторы выпуска, ISIC4 | 1995– | `https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE6,/A.<ISO3>..........` | OECD T&C (CC BY 4.0) |
| OECD National Accounts, Table 7 (`DF_TABLE7`) | занятые, рабочие места, часы | 1995– | аналогично | OECD T&C |
| OECD STAN 2025 | резерв для занятых и наёмных | 1995– | `…DSD_STAN@DF_STAN_2025…` | OECD T&C |
| OECD TiMBC 2025 | занятость по образованию (ISCED) и 17 агрегатам отраслей; занятость для MRIO | 2008–2022 | `…DSD_TIMBC_2025@DF_TIMBC_2025,/EMPN...W.PS.A.` | OECD T&C |
| BEA Input-Output (AllTablesIO.zip, выпуск 2024) | Make/Use до перераспределения, цены производителей, сводный уровень (71 отрасль) | 1997–2023 | `https://apps.bea.gov/industry/iTables%20Static%20Files/AllTablesIO.zip` | общественное достояние (US Gov) |
| BEA Import Matrices | импортные затраты | 1997–2023 | `https://apps.bea.gov/industry/xls/io-annual/ImportMatrices_Before_Redefinitions_SUM_1997-2023.xlsx` | общественное достояние |
| BEA GDP by Industry, TGO104 | цепные индексы цен валового выпуска | 1997–2025 | `https://apps.bea.gov/industry/Release/XLS/GDPxInd/GrossOutput.xlsx` | общественное достояние |
| BEA NIPA, таблица 6.5D | FTE наёмных по отраслям | 1998– | `https://apps.bea.gov/national/Release/TXT/NipaDataA.txt` | общественное достояние |

**WIOD 2016 недоступен:** dataverse.nl (GGDC) на автоматические запросы отвечает страницей «BotStopper: Access Denied». Обходить защиту мы не стали. WIOD заменён связкой FIGARO + OECD (для 2010–2023) и BEA (для длинного ряда США). Из-за этого вне-выборочный тест «2000–2008 → 2009–2014» выполнен **только на BEA по США**; для FIGARO разбиения 2010–2016 → 2017–2023 и 2010–2018 → 2019–2023.

**Другие ограничения доступа:** OECD ICIO и сайт oecd.org закрыты проверкой Cloudflare; BLS (почасовые данные KLEMS по США) отвечает 403; OECD Table 7 для Турции отсутствует (404). Точные файлы и их sha256 фиксируются в `data/raw/MANIFEST.csv` (создаётся при загрузке).

Страны: **США** (развитая, рыночная), **Германия** (развитая, экспортно-промышленная), **Мексика** (развивающаяся). Для проверки устойчивости: FRA, ITA, ESP, NLD, AUT, POL, CZE, KOR, JPN, GBR.

## Запуск

```bash
pip install -r requirements.txt
export PYTHONPATH=src OMP_NUM_THREADS=2      # иначе потоки BLAS в параллельных процессах мешают друг другу
python -m lts.download                       # ~450 МБ, в data/raw/ + MANIFEST.csv
python -m lts.figaro                         # FIGARO TSV -> data/processed/figaro/figaro_<год>.npz (~10 мин)
python -m pytest                             # 18 юнит-тестов ядра
python -m lts.levels main                    # США, Германия, Мексика 2010-2023 (FIGARO)
python -m lts.levels bea                     # США 1998-2023 (BEA)
python -m lts.levels robust                  # 10 дополнительных стран
python -m lts.dynamics main && python -m lts.dynamics bea   # динамика, горс-рейс, вне выборки
python -m lts.mrio                           # импорт по трудоёмкости экспортёров
python -m lts.validation                     # репликация Zachariah (2006) + технические проверки
python -m lts.report_tables                  # results/report_tables.md и results/figures/*.png
```

## Структура кода

| Модуль | Назначение |
|---|---|
| `core.py` | `Economy`: матрицы A, Aᵐ, D; **одна функция `content` для всех базисов**; открытая и замкнутая системы (`closed=False/True/"uniform"`); цены производства (линейные при заданном r, собственный вектор при авансированной зарплате); обобщённая «эксплуатация» товаров |
| `figaro.py`, `oecd.py`, `economy.py`, `bea.py`, `codes.py` | загрузка, соответствие классификаций (через множества 2-значных разделов NACE), сборка экономик страна-год |
| `levels.py` | отношения z для всех базисов и спецификаций; плацебо |
| `metrics.py` | CV (взвешенный и нет), MAD, MAWD, d-метрика Steedman–Tomkins, log-SD, корреляции совокупных величин (помечены как смещённые масштабом) и корреляция с поправкой на масштаб |
| `dynamics.py` | T1 (R² при β=1, некруговой), T2 (панельная FE-регрессия, лаги 0–2), T3 (охватывающие веса / горс-рейс), T4 (вне выборки), T5 (стоимости против цен производства) |
| `mrio.py` | мировая матрица Леонтьева, импорт по содержанию труда или ресурса в стране-экспортёре |
| `validation.py` | воспроизведение опубликованного результата; суммы, собственные числа, неотрицательность |


## Итерация 2 (проверка устойчивости)

Отчёт — [`report/REPORT_v2.md`](report/REPORT_v2.md), предрегистрация — [`pre_registration_v2.md`](pre_registration_v2.md), вывод разложения — [`report/methods_decomposition.md`](report/methods_decomposition.md), независимая реализация симметричного теста — [`independent/`](independent/).

Дополнительные данные: EU KLEMS 2023 labour accounts (`https://www.dropbox.com/s/vgtzptui1m1tj9l/labour%20accounts.csv?dl=1`, ссылка со страницы euklems-intanprod-llee.luiss.it/download), BLS «Labor productivity, detailed industries» (`https://www.bls.gov/productivity/tables/labor-productivity-detailed-industries.xlsx`; BLS требует описательный User-Agent), матрица потоков капитала BEA 1997 (`https://apps.bea.gov/industry/xls/flow1997.xls`). WIOD и INEGI недоступны (см. `ASSUMPTIONS.md`, A-DATA-V2-*).

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2
python - <<'PY'                                   # примитивы для независимой реализации
import numpy as np
from lts.economy import build
for c in ['USA','DEU','MEX']:
    for y in range(2010,2024):
        e,_=build(c,y)
        np.savez_compressed(f'data/primitives/{c}_{y}.npz', labels=np.array(e.labels), Zdom=e.A*e.x, Zimp=e.Am*e.x,
            x=e.x, hours=e.hours, D1=e.wages, labour_income=e.labour_income, cfc=e.cfc, gfcf_dom=e.gfcf_dom,
            gfcf_imp=e.gfcf_imp, hh_dom=e.hh_dom, hh_imp=e.hh_imp, va=e.va)
PY
python independent/symtest.py                     # этап 1.1 (независимая реализация)
python -m lts.v2.baskets figaro && python -m lts.v2.baskets bea          # этап 1.3
python -m lts.v2.decomposition && python -m lts.v2.mixture               # этап 1б (+ тест охвата 1в)
python -m lts.v2.placebo_cost && python -m lts.v2.gradient               # этап 1в
python -m lts.v2.reduction_pp figaro && python -m lts.v2.reduction_pp bea && python -m lts.v2.analyze_pp   # этапы 2-3
python -m lts.v2.long_run                                                 # этап 4
python -m lts.v2.data_improve                                             # этап 5 (BEA: BLS, потоки капитала)
python -m lts.v2.figures
```

Для этапов 1в–4 нужны `results/tables/ratios_long_{main,bea,robust}.parquet` из итерации 1 (`python -m lts.levels main|bea|robust`).


## Итерация 3 (плацебо той же формы, редукция без отраслевых зарплат, межстрановые различия)

Отчёт — [`report/REPORT_v3.md`](report/REPORT_v3.md), предрегистрация — [`pre_registration_v3.md`](pre_registration_v3.md), таблицы — [`results/v3/tables_v3.md`](results/v3/tables_v3.md), таблица соответствия O*NET/SOC/ISCO — `results/v3/crosswalk_soc2018_soc2010_isco08_jobzone.csv`, источники и sha256 — `results/v3/MANIFEST_v3.csv`.

Новые данные (все публичные): ILOSTAT SDMX (занятость ISIC4 × ISCO-08, заработки по ISCO-08), OECD Education at a Glance (относительные заработки по образованию; число учащихся UOE), OECD охват коллективными договорами, Eurostat SES 2018 и LFS, O*NET 29.0, BLS OEWS 2022 и таблицы соответствия SOC, население World Bank. Недоступны: EWCS (Eurofound — 429, микроданные по регистрации), ICTWSS (oecd.org за Cloudflare), WIOD.

**Исправление данных:** для Японии коды D, M, R в OECD Tables 6/7 означают D+E, M+N, R+S+T (`oecd.ACT_REMAP`). Прежние результаты по Японии в итерациях 1–2 устарели (см. REPORT_v3, §5.1).

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2
python -m lts.v3.download              # data/raw/v3/ + MANIFEST_v3.csv
python -m lts.v3.placebo_disp          # этап П: плацебо (а) перестановки, (б) лог-нормальные, (в) смеси v2
python -m lts.v3.bea_check             # этап П для США на данных BEA
python -m lts.v3.reduction             # этап 1: варианты редукции (≈ 40 мин)
python -m lts.v3.section_bound         # диагностика: потолок весов на уровне секций
python -m lts.v3.premium               # этап 2: надбавки и прибыльность
python -m lts.v3.crosscountry harmonised && python -m lts.v3.crosscountry   # этап 3
python -m lts.v3.summarize             # results/v3/tables_v3.md, results/figures/v3_*.png
```

## Итерация 4 (часть А: устойчивость v3)

Отчёт — [`report/REPORT_v4.md`](report/REPORT_v4.md), предрегистрация — [`pre_registration_v4.md`](pre_registration_v4.md), рецензии — `results/v3/review_independent.md` (v3), `results/v4/review_a1.md` (А1). Часть Б (источник прибыли, EU KLEMS 2023) предрегистрирована, но не начата.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2
python -m lts.v4.splits     # А1: списки отраслей с разбитой занятостью (results/v4/a1_split_lists.csv)
python -m lts.v4.a1         # А1: этап П без разбитых отраслей и с донорской разбивкой (JPN, KOR, USA)
python -m lts.v4.a2         # А2: мировая система FIGARO, импорт по трудоёмкости поставщиков (~40 мин)
```
Для А2 нужен `data/raw/v4/oecd_avg_hours.csv` (`https://sdmx.oecd.org/public/rest/data/OECD.ELS.SAE,DSD_HW@DF_AVG_ANN_HRS_WKD,/all?startPeriod=2008&format=csvfilewithlabels`).

## Итерация 4 (часть Б: откуда прибыль)

Данные: EU KLEMS 2023 (`national accounts.csv`, `capital accounts.csv`, `intangibles analytical.csv`, `Variable-List-2023.xlsx` со страницы euklems-intanprod-llee.luiss.it/download) в `data/raw/euklems/`; Eurostat `nama_10_a64` (P51C, D29X39, B1G, D1; `.../data/nama_10_a64/A.CP_MNAC..<ITEM>.?format=TSV&compressed=true`) в `data/raw/v4/nama64_<ITEM>.tsv.gz`.

```bash
python -m lts.v4.b0              # Б0: T1 с плоскими эталонами и перестановками
python -m lts.v4.b_data          # панель KLEMS (results/v4/b_panel.csv.gz)
python -m lts.v4.b0b             # Б0б: равный разброс (нужна панель)
python -m lts.v4.b_tests b31 && python -m lts.v4.b_tests b33 && python -m lts.v4.b_tests b32   # Б3.1, Б3.3, Б3.2 (~1,5 ч)
python -m lts.v4.b_summary       # results/v4/tables_b.md
```

## Итоговый отчёт

[`report/REPORT_FINAL.md`](report/REPORT_FINAL.md) — сводка итераций 1–4 по темам, с метками статуса, хронологией изменений выводов, методологическими находками и рецензией. Краткая версия — [`report/SUMMARY.md`](report/SUMMARY.md). Каждое число с меткой [N…] сверено с `results/`:

```bash
export PYTHONPATH=src
python -m lts.final.verify_benchmarks   # плоский вектор и степени часов (results/final/benchmarks_flat_power.csv)
python -m lts.final.check_numbers       # реестр чисел → report/FINAL_numbers_check.md (файлы и текст)
```

## Итерация 5 (автоматизация, совокупная норма прибыли, механизм цены)

Отчёт: [`report/REPORT_v5.md`](report/REPORT_v5.md); предрегистрация и журнал: [`pre_registration_v5.md`](pre_registration_v5.md).

Данные (в `data/raw/v5/`, кроме KLEMS и Eurostat из итерации 4):
* BEA NIPA `NipaDataA.txt`, `SeriesRegister.txt` (apps.bea.gov/national/Release/TXT);
* FRED: `BOGZ1FL105013865A`, `BOGZ1FL105013765A` (Z.1, капитал нефинансовых корпораций), `TCU` — `fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`;
* OECD Economic Outlook, разрыв выпуска: `oecd_eo_gap.csv` (DSD_EO@DF_EO, `.GAP.A`);
* Eurostat `ei_bsin_q_r2` (загрузка мощностей);
* EPWT 7.0: `epwt70.xlsx` (Harvard Dataverse);
* Wright (2008): `wright2008.pdf`.

Нужен `numba` (агентные модели).

```bash
export PYTHONPATH=src
python -m lts.v5.agg        # агрегаты страна-год (results/v5/agg_*.csv)
python -m lts.v5.stage1     # этап 1: разложения, T1–T3, США 1951–2025, EPWT (~3 мин)
python -m lts.v5.stage2     # этап 2: меры автоматизации и тесты (~5 мин)
python -m lts.v5.wright 10  # этап 3.1: Wright (2008), 80 прогонов (~40 мин на 3 ядрах)
python -m lts.v5.wright 3 diag   # диагностические варианты
python -m lts.v5.abm 10     # этап 3.2: 400 прогонов расширенной модели (~2 мин)
python -m lts.v5.summary    # results/v5/tables_v5.md
```

## Итерация 6 (длинные ряды нормы прибыли; выравнивание нормы прибыли)

Отчёт: [`report/REPORT_v6.md`](report/REPORT_v6.md); предрегистрация и журнал: [`pre_registration_v6.md`](pre_registration_v6.md).

Данные (перечислены в `data/raw/MANIFEST.csv` с sha256):
* AMECO — `data/raw/v6/ameco0.zip` (https://ec.europa.eu/economy_finance/db_indicators/ameco/documents/ameco0.zip), распаковать в `data/raw/v6/ameco/`;
* BEA Fixed Assets, разделы 1–9 — `data/raw/v6/bea_fa/` (в репозитории; прямое скачивание отсюда даёт 403);
* WIOD 2016 — `data/raw/v6/wiod/` (Dataverse doi:10.34894/PJ2M1C: WIOTS_in_R.zip → `wiot/`, Socio_Economic_Accounts.xlsx, Exchange_Rates.xlsx);
* NIPA, FIGARO, EU KLEMS, OECD TiMBC — из итераций 4–5.

Нужен `pyreadr` (чтение WIOD).

```bash
export PYTHONPATH=src
python -m lts.v6.series    # длинные ряды (results/v6/series.csv)
python -m lts.v6.tests     # этапы 2–3: разложения, тренды, разрывы, панель (~15 мин)
python -m lts.v6.stage4    # этап 4 (разведочно): ВНД, иностранный труд в импорте
python -m lts.v6.stage4_icio   # этап 4 на OECD ICIO 1995–2022 + капитал StatCan (журнал п. 9; данные — Google Drive пользователя, data/raw/v6/gdrive; PWT 10.01 — data/raw/v6/pwt1001.dta)
python -m lts.v6.national   # официальные ряды сектора НФК вне США (Eurostat, ONS, ABS, ESRI, StatCan) и Z.1 для 5.3 (журнал п. 10; data/raw/v6/national, data/raw/v6/z1)
python -m lts.v6.stage5    # этап 5: авансированный капитал, инкрементальные нормы, финансовые доходы
python -m lts.v6.charts    # рисунки report/fig_v6/
python -m lts.v6.summary   # results/v6/tables_v6.md
python -m lts.v6.review_checks  # проверки по рецензии (журнал п. 8): ист. стоимость на США НФК, fixed-b, симуляция 5.2
```

## Пересмотр итераций 1–5 на новых данных

Отчёт: [`report/REPORT_revisit.md`](report/REPORT_revisit.md); предрегистрация и журнал: [`pre_registration_revisit.md`](pre_registration_revisit.md). Данные — WIOD 2016 (как в итерации 6).

```bash
export PYTHONPATH=src
python -m lts.revisit.wiod r1 main          # R1: стадия П итерации 3 на WIOD (42 страны, 2000–2014; ~20 мин)
python -m lts.revisit.wiod r1 no_capital    # чувствительность: без капитала
python -m lts.revisit.wiod r1 delta_lo      # δ × 0,5
python -m lts.revisit.wiod r1 delta_hi      # δ × 1,5
python -m lts.revisit.wiod r2               # R2: Б3.1 итерации 4 на WIOD
python -m lts.revisit.wiod r2iv             # R2: инструментальные оценки
```

## Итерация 7: неучтённый капитал и редукция сложного труда

Отчёт: [`report/REPORT_v7.md`](report/REPORT_v7.md); предрегистрация и журнал: [`pre_registration_v7.md`](pre_registration_v7.md); источники: [`report/sources_v7.md`](report/sources_v7.md); таблицы соответствия: [`report/crosswalk_v7.md`](report/crosswalk_v7.md).

Данные (`data/raw/v7/`, записаны в MANIFEST): EUKLEMS intangibles analytical (тот же файл, что в итерации 4), Ewens–Peters–Wang (параметры и запасы), SEC Financial Statement Data Sets (качаются скриптом), ONS intangibles by industry, CBS marketing assets, BLS EP табл. 5.4, O*NET 31.0; из итерации 3 — OEWS 2022.

```bash
export PYTHONPATH=src
python -m lts.v7.part1      # часть 1: порог m*, кривые, прочие проверки, Oster / Cinelli–Hazlett
python -m lts.v7.sec        # SEC: SG&A и основные средства по SIC (16 квартальных архивов, ~20 мин)
python -m lts.v7.sec threshold   # порог на профиле SEC (results/v7/p1_sec_*.csv)
python -m lts.v7.profiles   # национальные проверки профиля (ONS, CBS)
python -m lts.v7.part2      # часть 2: ψ по профессиям, тест надбавок, ценовые тесты с эталонами сжатия
```

После независимой рецензии итерации 7 (журнал п. 6):

```bash
python -m lts.v7.extra           # проверки по внешней рецензии (b, MDE, (б′), К5, мощность теста 3)
python -m lts.v7.review_checks   # проверки по независимой рецензии (порог в логарифмах, SEC на всей панели, RV, R² строгой формы)
```

## Итерация 8: единое правило вывода, чистка тестов цен, закон тенденции

Отчёт: [`report/REPORT_v8.md`](report/REPORT_v8.md); сводная таблица исходов: [`report/outcomes_table.md`](report/outcomes_table.md); предрегистрация и журнал: [`pre_registration_v8.md`](pre_registration_v8.md); доступность данных: [`data/raw/v8_availability.md`](data/raw/v8_availability.md).

Данные (`data/raw/v8/`, записаны в MANIFEST):
* Eurostat `nama_10_nfa_st`, `nama_10_a64` (по переменным), `ei_bsin_q_r2`, `gov_10a_*`;
* квартальные NIPA (`NipaDataQ.txt`);
* таблицы BEA SUP (`AllTablesSUP.zip`); детальные Make/Use 2017 — из `data/raw/bea/AllTablesIO.zip`;
* QCEW 2017;
* FRED (CUMFNS, TCU, USREC, JHDUSRGDPBR, HOANBS);
* Census Economic Census (концентрация 2012, 2017, 2022).

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2
python -m lts.v8.stage2            # степень трансформации b, отношения, время оборота, концентрация
python -m lts.v8.stage3 market     # рыночный сектор, факторные издержки, ФИСИМ (~15 мин)
python -m lts.v8.stage3 gross      # валовой запас Eurostat в тесте Б0б
python -m lts.v8.stage3 gradient   # градиент по дезагрегации (BEA 2017 + QCEW); "gradient fix" — вариант охвата (журнал п. 7)
python -m lts.v8.stage3 hedonic    # T1 без гедонически дефлируемых отраслей
python -m lts.v8.stage4            # закон тенденции: r_n, фазы, обесценение, сжатие прибыли, внешний критерий, налоги
python -m lts.v8.stage5            # Маркс против Калецки (квартальные NIPA)
python -m lts.v8.stage1            # сводная таблица исходов 1–19 по единому правилу -> report/outcomes_table.md
```

## Итерация 9: итоги о прибыли, рабочая сила как товар, мировой рынок, непроизводительный труд

Отчёт: [`report/REPORT_v9.md`](report/REPORT_v9.md); сводная таблица исходов 1–38: [`report/outcomes_table.md`](report/outcomes_table.md); предрегистрация и журнал (п. 1–13): [`pre_registration_v9.md`](pre_registration_v9.md); доступность данных: [`data/raw/v9_availability.md`](data/raw/v9_availability.md).

Данные (`data/raw/v9/`, записаны в MANIFEST): OECD SDMX (почасовой заработок, ИПЦ, охват договорами, профсоюзы), Eurostat (HICP, LCI, ИЦП, безработица, slack), FRED, BLS (ECI, ip), GGDC PLD 2005 и 2023, PWT 11, EXIOBASE 3.8.2 (2011, 2017 ixi; 2017, 2021 pxp), приложения Hickel et al. 2024, BEA GDP by Industry, MPRA 81542 и 84035. Повторно — AMECO, WIOD 2016 + SEA, OECD STAN, BEA FA, EU KLEMS, BEA 2017 + QCEW.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2
python -m lts.v9.download             # загрузка (пропускает уже скачанное; имя файла — скачать заново)
python -m lts.v9.stage1               # панель Б3 без перекрытий, фазы, реальная переоценка, парный бутстреп градиента
python -m lts.v9.b_est                # семь оценщиков b, выбор основного (исход 21)
python -m lts.v9.stage2               # перенос цен, 2021–2023, якорь и возврат, срез (исходы 22–27); части: pt, infl, anchor
python -m lts.v9.stage34              # мировые стоимости и ячейки «страна × торгуемая отрасль», 2000–2014 (~часы)
python -m lts.v9.stage34 outcomes     # исходы 28–33 с бутстрепом
python -m lts.v9.stage5               # Hickel et al., разложение перетока, b между странами, корреляции T_c; части: replicate, between, corr
python -m lts.v9.stage6               # реальный курс (исходы 35–36)
python -m lts.v9.stage7               # марксова норма прибыли с производительным трудом (исходы 37–38)
python -m lts.v9.stage8               # градиент с часами BLS
python -m lts.v9.outcomes             # сводная таблица исходов 1–38 -> report/outcomes_table.md
```


## Итерация 9б: доработка итерации 9 и расширение марксовой нормы прибыли

Отчёт: [`report/REPORT_v9b.md`](report/REPORT_v9b.md); сводная таблица исходов 1–46: [`report/outcomes_table.md`](report/outcomes_table.md); предрегистрация и журнал: [`pre_registration_v9b.md`](pre_registration_v9b.md); доступность данных: [`data/raw/v9b_availability.md`](data/raw/v9b_availability.md).

Данные (`data/raw/v9b/`, записаны в MANIFEST): BEA GDP by Industry по SIC (1947–1997), NIPA (все годовые ряды), EUKLEMS & INTANProd 2025, OEWS 2012–2024, Z.1, Eurostat `nrg_ind_id` и B2A3N.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2
python -m lts.v9b.download            # загрузка
python -m lts.v9b.stage1 long         # концы ряда, длинный ряд США 1947–2024, разложение, Мозли (исходы 39, 40, 43)
python -m lts.v9b.stage1 countries    # страны Eurostat и EU KLEMS (исходы 41, 42)
python -m lts.v9b.stage1 oews         # внутриотраслевые непроизводительные функции
python -m lts.v9b.stage1 external     # внешний критерий с r_m
python -m lts.v9b.stage23 world       # мировые отрасли, эталон Р18 (исходы 44–46; ~1 ч)
python -m lts.v9b.stage23 unequal     # аддитивное разложение неравноценного обмена
python -m lts.v9b.stage45 labour      # slack, исход 27 с диким бутстрепом, LCI D11, условия торговли, исход 26 без прогнозов AMECO
python -m lts.v9b.stage45 minor       # рыночное обесценение (Z.1), исход 35 с лагом 1
python -m lts.v9b.stage45 o34         # парный бутстреп исхода 34
python -m lts.v9b.stage45 b           # НМНК b на сетке [−1; 4], кластерная F первого шага ИП
python -m lts.v9b.outcomes            # сводная таблица исходов 1–46
```


## Итерация 10: особая роль труда как контроль над трудом; закрытие открытых вопросов о ценах и прибыли

Отчёт: [`report/REPORT_v10.md`](report/REPORT_v10.md); сводная таблица исходов 1–63: [`report/outcomes_table.md`](report/outcomes_table.md); предрегистрация и журнал (16 пунктов): [`pre_registration_v10.md`](pre_registration_v10.md); доступность данных и список для ручной загрузки: [`data/raw/v10_availability.md`](data/raw/v10_availability.md).

Данные (`data/raw/v10/`, записаны в MANIFEST):
* BEA Regional (штаты);
* шоки нефти BH, Känzig, Kilian; FRED (WTI, газ, безработица, PPI машин);
* Eurostat `namq_10_a10`, HETUS;
* NLRB (SQLite, 1,1 ГБ, не коммитится);
* SEC FSDS 2009q2–2026q2 (выписка; архивы не хранятся);
* OECD BSDB и ISDB; EU KLEMS 2009;
* Pink Sheet, BIS, WDI;
* ATUS; Comtrade HS 847950.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2
python -P -m lts.v10.download && python -P -m lts.v10.download2   # загрузка; Känzig — из клона репозитория автора
python -P -m lts.v10.sec_fsds rebuild   # выписка SEC (без размерных строк; журнал 7; ~1,5 ч)
python -P -m lts.v10.stage1             # шоки силы труда против поставщиков: США, RTW, синтетический контроль, газ, сталь (47–51)
python -P -m lts.v10.stage1 rtw_trend   # RTW с предтрендом (журнал 4)
python -P -m lts.v10.stage2             # выборы NLRB × SEC, разрыв на 50% (53); `robust` — проверки (журнал 13)
python -P -m lts.v10.stage3             # плацебо исхода 44 (54; ~1 ч), фазы обесценения, Мозли-опережение
python -P -m lts.v10.stage4             # проверка данных R1–R4 и пересчёт 1, 3, 15, 28, 44 (55; ~1 ч)
python -P -m lts.v10.stage5             # распределение норм прибыли фирм (56, 57)
python -P -m lts.v10.stage6             # домашний труд в симметричной постановке (~1 ч; `resume` — продолжение)
python -P -m lts.v10.stage7             # абсолютная и относительная прибавочная стоимость
python -P -m lts.v10.stage8             # роботы
python -P -m lts.v10.stage9             # длинные ряды нормы прибыли бизнес-сектора и r_m (58, 59)
python -P -m lts.v10.stage10            # рента (60, 61)
python -P -m lts.v10.stage11            # Окисио против Маркса (62, 63)
python -P -m lts.v10.review             # проверки по независимой рецензии (журнал 16; этап 6 ~20 мин)
python -P -m lts.v10.outcomes           # сводная таблица исходов 1–63
```


## Итерация 11: закон стоимости как регулятор распределения труда; процент и фиктивный капитал

Отчёт: [`report/REPORT_v11.md`](report/REPORT_v11.md); сводная таблица исходов 1–74: [`report/outcomes_table.md`](report/outcomes_table.md); предрегистрация и журнал: [`pre_registration_v11.md`](pre_registration_v11.md); доступность данных и список для ручной загрузки: [`data/raw/v11_availability.md`](data/raw/v11_availability.md).

Данные (`data/raw/v11/`, записаны в MANIFEST): JST Macrohistory R6; FRED (GS10, TB3MS, FEDFUNDS, Aaa, Baa, CPI-U); OECD (долгосрочные и 3-месячные ставки). Повторно — WIOD 2016, EU KLEMS 2025, таблицы FIGARO, Z.1, BEA FA, PWT 11, ряды итерации 6.

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2
python -P -m lts.v11.download             # загрузка
python -P -m lts.v11.stage1 signals       # сигналы S_v, S_pp, плацебо по WIOD (~3 мин; parquet не коммитится)
python -P -m lts.v11.stage1 figaro_hours  # часы FIGARO (13 стран, 2010–2022)
python -P -m lts.v11.stage1 estimate      # исходы 64–67, тест 1, плацебо, варианты, репликация FIGARO (~15 мин)
python -P -m lts.v11.stage2               # процент и фиктивный капитал (68–74), разложение 2.3
python -P -m lts.v11.checks               # описательные проверки после результатов (журнал 3)
python -P -m lts.v11.outcomes             # сводная таблица исходов 1–74
```

