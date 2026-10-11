# Итерация 12б: доступность данных (проверено 2026-10-11, до расчётов)

Новые файлы — в `data/raw/v12b/`, записи — в `data/raw/MANIFEST.csv` (группа `v12b`). Скрипт — `src/lts/v12b/download.py`.

## Скачано в этой итерации

| Этап | Источник | Что | Файл |
|---|---|---|---|
| 1, 3 | BEA, Input-Output Accounts (исторические годовые, NAICS) | use до перераспределений, цены производителей, 1947–1962 (46 отраслей) и 1963–1996 (65); make за те же годы | `bea_IOUse_*`, `bea_IOMake_*` |
| 1, 3 | BEA, GDP by Industry 1947–1997 (NAICS) | ДС, компоненты (оплата, налоги, валовая прибыль), выпуск | `bea_AllTablesHist.zip` |
| 1 | EIA, Monthly Energy Review | производство (T01.02) и потребление (T01.03) первичной энергии по источникам, 1949– | `eia_T01.02.csv`, `eia_T01.03.csv` |
| 1 | Eurostat `env_ac_pefa04` | физические счета энергии по NACE (добыча природной энергии, чистое внутреннее использование), DE, FR, IT, ES, NL, AT, PL, CZ, 2000–2024 | `es_env_ac_pefa04.json` |
| 2 | Maito (2014), MPRA 55894 | текст статьи (методика, страны) | `maito2014_mpra55894.pdf` |

## Используется из прошлых итераций

| Этап | Источник | Где |
|---|---|---|
| 1 | BEA use / make / импорт 1997–2023 (summary), NIPA (6.5D FTE, A065RC, DPCERX, A191RD), OECD Table 7 (самозанятые) | `data/raw/bea/`, `data/raw/oecd/` |
| 1 | BEA занятость по отраслям SIC 1948–1997 (лица, занятые в производстве, FTE) | `data/raw/v9b/bea_GDPbyInd_VA_SIC.xls` |
| 1, 3 | BEA Fixed Assets: 2.1–2.2 (виды активов), 3.1–3.4 E/S/I (отрасли) | `data/raw/v6/bea_fa/` |
| 1 | NIPA квартальные (запасы, табл. 5.8.5) | `data/raw/v8/NipaDataQ.txt` |
| 0, 1, 2 | WIOD 2016 (WIOT, SEA), PWT 11, PLD | `data/raw/v6/wiod/`, `data/raw/v9/` |
| 1, 3 | FIGARO (13 стран), EU KLEMS 2025 (капитал по видам активов), EU KLEMS 2009 | `data/raw/v9b/`, `data/raw/v10/` |
| 2 | EPWT 7.0 | `data/raw/v5/epwt70.xlsx` |
| 4 | ряды НФК США (итерация 6), AMECO, JST R6, пики NBER (итерация 8) | `results/v6/`, `data/raw/v6/ameco*`, `data/raw/v11/`, `data/raw/v8/` |

## Не скачано — список для ручной загрузки

| Приоритет | Источник | Зачем (этап) | Где взять | Что сделано без него |
|---|---|---|---|---|
| средний | **Энергетические счета WIOD 2016** (Kulionis, 2019; энергия по отраслям WIOD, 2000–2014, ТДж) и/или **JRC WIOD Environmental Accounts Update 2000–2016** (Corsatea et al., 2019, EUR 29727 EN) | 1, 2: единица «энергия» на WIOD | запись набора: https://service.tib.eu/ldm_nfdi4energy/ldmservice/vdataset/openaire_396c8408-0c28-4a29-948c-dc3b3b298fc5 (в архиве «download_all» только метаданные, без файлов); отчёт JRC: https://publications.jrc.ec.europa.eu/repository/handle/JRC116234 | энергия — только США (EIA) и 8 стран ЕС (PEFA, FIGARO) |
| средний | **Basu, Huato, Lara Jauregui, Wasner, World Profit Rates 1960–2019**: текст рабочей версии и данные «World Profitability Dashboard» | 2: сверка | ScholarWorks UMass (Economics Department Working Paper 318; скачивание PDF отвечает 403) | расчёт повторён на их источнике (EPWT 7.0); сверка — по аннотации статьи |
| низкий | **BLS Total Factor Productivity for Major Industries** (часы всех занятых по ~60 отраслям, 1987–) | 1: вариант с фактическими часами вместо лиц | bls.gov/productivity (прежний адрес `total-factor-productivity-major-industries.xlsx` отвечает 404; в репозитории — сохранённая страница ошибки) | лица, занятые в производстве (одинаковые часы на человека внутри года) |
| низкий | NIPA, исторические таблицы 6.9B / 6.9C (часы наёмных по отраслям SIC, 1948–2000) | 1: часы до 1998 г. | apps.bea.gov, интерактивные таблицы NIPA, раздел 6 (в плоских файлах только версии D) | лица, занятые в производстве |

**Для этапов 0 и 1 приоритета 1 вручную качать ничего не обязательно.**
