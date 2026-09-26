# Эмпирическая проверка трудовой теории стоимости

Воспроизводимый расчёт: объясняет ли вертикально интегрированный труд отраслевые цены **лучше альтернатив** (цен производства, других «базисов стоимости», случайных базисов) по метрикам, устойчивым к известной критике (эффект масштаба, круговость редукции через зарплаты, асимметрия трудового и товарных базисов, неразличимость теорий издержек).

* Отчёт: [`report/REPORT.md`](report/REPORT.md)
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
