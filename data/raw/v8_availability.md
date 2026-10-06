# Итерация 8: доступность данных (проверено 2026-10-06, до расчётов)

Все файлы — в `data/raw/v8/` (кроме указанных); записи — в `data/raw/MANIFEST.csv`.

| Источник | Переменные | Статус | URL |
|---|---|---|---|
| Eurostat `nama_10_nfa_st` | валовой (N11G) и чистый (N11N) запас основных фондов по отраслям A*64, восстановительные цены (CRC_MEUR), 34 страны, 1975–2025 | скачан | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nama_10_nfa_st?format=JSON&unit=CRC_MEUR&asset10=N11G&asset10=N11N |
| BEA NIPA T50805B | запасы (inventories) по отраслям, квартальные/годовые | есть в `NipaDataA.txt` / `NipaDataQ.txt` (ряды A371RC, A373RC, …; см. `SeriesRegister.txt`) | https://apps.bea.gov/national/Release/TXT/NipaDataQ.txt |
| BEA NIPA, квартальные | прибыль, инвестиции, сальдо госбюджета, сбережения | скачан (`NipaDataQ.txt`, 35 МБ) | https://apps.bea.gov/national/Release/TXT/NipaDataQ.txt |
| BEA «затраты — выпуск» | 2017: детальный уровень (402), summary (71), sector (15); 1997–2023 summary | есть (`data/raw/bea/AllTablesIO.zip`) + таблицы SUP (`AllTablesSUP.zip`) | https://apps.bea.gov/industry/iTables%20Static%20Files/AllTablesIO.zip, …/AllTablesSUP.zip |
| BEA «138 отраслей» | — | **отдельной таблицы нет** (в архивах BEA только 15 / 71 / 402). Средний уровень строится агрегированием той же детальной таблицы 2017 г. до NAICS 3 знаков | — |
| BLS QCEW 2017 | занятость и зарплаты по NAICS 6 знаков (для детального уровня) | скачан (143 МБ) | https://data.bls.gov/cew/data/files/2017/csv/2017_annual_by_industry.zip |
| FRED | загрузка мощностей TCU (1967+), CUMFNS (обработка, 1948+); USREC (NBER); JHDUSRGDPBR | скачаны | https://fred.stlouisfed.org/graph/fredgraph.csv?id=… |
| Eurostat `ei_bsin_q_r2` | загрузка мощностей в обработке (BS-ICU-PC), квартальные, 1980Q1–2026Q3, ЕС (без UK) | скачан | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsin_q_r2?indic=BS-ICU-PC&s_adj=SA |
| Census Economic Census | концентрация CR4/CR8/CR20/CR50 по NAICS, все секторы: 2017, 2022; обработка: 2012 | скачаны (`EC1700SIZECONCEN.zip`, `EC2200SIZECONCEN.zip`, `EC1231SR2.zip`) | https://www2.census.gov/programs-surveys/economic-census/data/2017/sector00/EC1700SIZECONCEN.zip и т. п. |
| Census API | — | требует ключ; не нужен (есть файлы) | — |
| Eurostat `gov_10a_taxag`, `gov_10a_main` | налоги, взносы, социальные выплаты | скачаны | API Eurostat |
| Z.1, BEA Fixed Assets, AMECO, KLEMS, WIOD, FIGARO, OECD ICIO | — | есть с прошлых итераций | — |

**Не нужно ничего качать вручную.** Ограничения: загрузки мощностей для UK в Eurostat нет (после 2020 г.); загрузка мощностей по всей экономике нигде не публикуется — используется обработка.
