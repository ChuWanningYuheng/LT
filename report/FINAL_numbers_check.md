# Сверка чисел итогового отчёта

Генерируется `PYTHONPATH=src python -m lts.final.check_numbers`. Для каждого идентификатора [N…] из REPORT_FINAL.md и SUMMARY.md: значение в тексте, пересчитанное значение, файл и правило.

Итого: 134 чисел, расхождений с файлами: 0. Идентификаторы в тексте без записи в реестре: нет. Записи реестра, не использованные в тексте: ['N49'].

Сверка «текст → реестр» (число, напечатанное перед меткой, совпадает со значением реестра как есть или в процентах): несовпадений 0.


| id | в тексте | пересчёт | файл | правило | статус |
|---|---|---|---|---|---|
| N1 | 0.066 | 0.066 | `results/report_tables.md` | Zachariah-like MAWD LTV, DEU | OK |
| N2 | 0.109 | 0.109 | `results/report_tables.md` | Zachariah-like MAWD LTV, USA | OK |
| N3 | 0.158 | 0.158 | `results/report_tables.md` | MAWD hours DEU, open, all, capital | OK |
| N4 | 0.315 | 0.315 | `results/report_tables.md` | MAWD hours USA (FIGARO) | OK |
| N5 | 0.002 | 0.002 | `results/report_tables.md` | share of commodity bases better than labour, DEU open | OK |
| N6 | 0.18 | 0.183 | `results/report_tables.md` | same, USA open | OK |
| N7 | 0.45 | 0.446 | `results/report_tables.md` | share better, DEU closed uniform | OK |
| N8 | 0.65 | 0.653 | `results/report_tables.md` | share better, USA closed uniform | OK |
| N9 | 0.094 | 0.094 | `results/report_tables.md` | MAWD wage-weighted labour DEU | OK |
| N10 | 0.2 | 0.2 | `results/report_tables.md` | MAWD wage-weighted labour USA | OK |
| N11 | 0.082 | 0.082 | `results/report_tables.md` | MAWD PP actual wage DEU | OK |
| N12 | 13 | 13 | `results/report_tables.md` | countries where PP actual < hours (open; JPN from iteration 1 table, before the data fix: JPN PP 0.156 < hours 0.204 after the fix too) | OK |
| N13 | 0.271 | 0.271 | `results/report_tables.md` | random placebo 5th percentile DEU | OK |
| N14 | 0.34 | 0.336 | `results/report_tables.md` | T1 labour DEU core h=1 | OK |
| N15 | 0.54 | 0.542 | `results/report_tables.md` | T1 labour DEU core h=5 | OK |
| N16 | 0.24 | 0.242 | `results/report_tables.md` | T1 labour USA core h=1 | OK |
| N17 | 0.73 | 0.729 | `results/report_tables.md` | T1 labour USA_BEA core h=5 | OK |
| N18 | 0.066 | 0.066 | `results/report_tables.md` | T4 BEA 2009-2014 core labour RMSE | OK |
| N19 | 0.064 | 0.064 | `results/report_tables.md` | T4 BEA 2009-2014 naive | OK |
| N20 | 0.048 | 0.048 | `results/report_tables.md` | T4 BEA 2009-2014 PP actual | OK |
| N21 | 414 | 414 | `results/v2/decomp_basis.csv` | country x basis pairs with symmetric X better than labour (MAWD) | OK |
| N22 | 796 | 796 | `results/v2/decomp_basis.csv` | country x basis pairs | OK |
| N23 | 0.61 | 0.6131 | `results/v2/decomp_basis.csv` | median labour share L_X among bases better than labour | OK |
| N24 | 0 | 0 | `results/v2/decomp_mixture.csv` | asym pairs where labour+O_X significantly better than labour (BH q<0.05) | OK |
| N25 | 0.04 | 0.0426 | `results/v2/baskets_figaro.csv` | DEU necessities-only actual basket: share of bases better | OK |
| N26 | 0.7 | 0.7 | `results/v2/decomp_theta.csv` | s* DEU | OK |
| N27 | 0.1 | 0.1 | `results/v2/decomp_theta.csv` | s* USA | OK |
| N28 | 0.4 | 0.4 | `results/v3/jpn_fixed_theta.csv` | s* JPN after data fix | OK |
| N29 | 0.126 | 0.1258 | `results/v2/pp_levels.csv` | MAWD PP(b) uniform wage DEU | OK |
| N30 | 0.226 | 0.226 | `results/v2/pp_levels.csv` | MAWD PP(b) USA | OK |
| N31 | 0.244 | 0.2439 | `results/v2/pp_levels.csv` | MAWD PP(b) USA BEA | OK |
| N32 | 5.1 | 5.0579 | `results/v2/long_run.csv` | half-life labour DEU (model B, core) | OK |
| N33 | 16 | 16.4426 | `results/v2/long_run.csv` | half-life labour USA BEA | OK |
| N34 | 10 | 10 | `results/v3/placebo_disp.csv` | countries in top 5% vs (a) and (b), both metrics | OK |
| N35 | 9 | 9 | `results/v3/placebo_disp_prefix_jpn.csv` | same before JPN fix | OK |
| N36 | 8 | 8 | `results/v3/placebo_disp.csv` | countries in top 5% vs (c) mixtures | OK |
| N37 | 0.04 | 0.0444 | `results/v3/reduction_levels.csv` | median G 1.1a (MAWD) | OK |
| N38 | 0.13 | 0.127 | `results/v3/reduction_levels.csv` | median G 1.1b | OK |
| N39 | 0.01 | 0.0125 | `results/v3/reduction_levels.csv` | median G 1.2a | OK |
| N40 | 0.92 | 0.9158 | `results/v3/reduction_levels.csv` | median G 1.5a frozen 2010 own wages | OK |
| N41 | 0.72 | 0.7244 | `results/v3/reduction_levels.csv` | median G 1.5b foreign median wages | OK |
| N42 | 0.57 | 0.5732 | `results/v3/section_bound.csv` | median G own wages averaged to sections | OK |
| N43 | -0.279 | -0.2793 | `results/v3/premium_2a_pooled.csv` | beta premium on PCM | OK |
| N44 | 0.029 | 0.0292 | `results/v3/premium_2a_pooled.csv` | p premium on PCM | OK |
| N45 | 0.336 | 0.3363 | `results/v3/cc_panel_tests.csv` | wild bootstrap p K1 coverage (DV mix) | OK |
| N46 | -0.6 | -0.599 | `results/v3/cc_cross_country.csv` | Spearman K1 cross-country (mix) | OK |
| N47 | 0.031 | 0.0312 | `results/v3/cc_cross_country.csv` | permutation p K1 | OK |
| N48 | 1.0 | 1.0 | `results/v3/placebo_disp.csv` | USA share of mixtures better than labour (MAWD) | OK |
| N49 | 0.204 | 0.204 | `results/v3/reduction_levels.csv` | MAWD hours JPN after fix | OK |
| N50 | 0.073 | 0.0734 | `results/v4/a1_placebo.csv` | JPN A1(a) share of permutations better (MAWD) | OK |
| N51 | 0.14 | 0.1398 | `results/v4/a1_placebo.csv` | JPN A1(a) permutations (d) | OK |
| N52 | 0.066 | 0.0662 | `results/v4/a1_placebo.csv` | KOR A1(b) permutations (d) | OK |
| N53 | 54 | 54 | `results/v4/a1_split_lists.csv` | KOR split industries from 2013 | OK |
| N54 | 45 | 45 | `results/v4/a1_split_lists.csv` | JPN split industries | OK |
| N55 | 10 | 10 | `results/v4/a2_mrio_placebo.csv` | MRIO: countries top 5% vs (a),(b), rest of world x1 | OK |
| N56 | 9 | 9 | `results/v4/a2_mrio_placebo.csv` | same, x0.5 | OK |
| N57 | 9 | 9 | `results/v4/a2_mrio_placebo.csv` | same, x2 | OK |
| N58 | 12 | 12 | `results/final/benchmarks_flat_power.csv` | countries where flat vector beats hours (MAWD, all industries) | OK |
| N59 | 10 | 10 | `results/final/benchmarks_flat_power.csv` | same, d | OK |
| N60 | 4 | 4 | `results/final/benchmarks_flat_power.csv` | flat beats hours without B,K,L,J61-63 (MAWD) | OK |
| N61 | 0.91 | 0.9147 | `results/final/benchmarks_flat_power.csv` | median G of hours^0.5 (MAWD) | OK |
| N62 | 0.6 | 0.6006 | `results/final/benchmarks_flat_power.csv` | median G of hours^0.75 (MAWD) | OK |
| N63 | 8 | 8 | `results/v4/b0_summary.csv` | B0: data sets where money-flat beats labour, core h=1 | OK |
| N64 | 6 | 6 | `results/v4/b0_summary.csv` | B0: labour beats physical flat and 95% of permutations, h=5 | OK |
| N65 | 2 | 2 | `results/v4/b0_summary.csv` | same, h=1 | OK |
| N66 | 9 | 9.0 | `results/v4/b0b_counts.csv` | B0b: non-labour costs not worse than hours, (a) all | OK |
| N67 | 10 | 10.0 | `results/v4/b0b_counts.csv` | B0b: hours point-better than non-labour on both metrics, (a) all | OK |
| N68 | 6 | 6.0 | `results/v4/b0b_counts.csv` | B0b: hours significantly better than CFC, (a) all | OK |
| N69 | 11 | 11.0 | `results/v4/b0b_counts.csv` | same, without rent industries | OK |
| N70 | 7 | 7.0 | `results/v4/b0b_counts.csv` | B0b: hours significantly better than capital (of 10) | OK |
| N71 | 4 | 4.0 | `results/v4/b0b_counts.csv` | B0b: wage bill significantly better than hours, (a) | OK |
| N72 | 8 | 8.0 | `results/v4/b0b_counts.csv` | same, (c) common dispersion | OK |
| N73 | 0.77 | 0.7649 | `results/v4/b0b_boot.csv` | median share of own permutations better than capital (MAWD) | OK |
| N74 | 0.94 | 0.9352 | `results/v4/b0b_boot.csv` | same, d | OK |
| N75 | 0.003 | 0.0033 | `results/v4/b0b_boot.csv` | same for hours, MAWD | OK |
| N76 | -0.028 | -0.0284 | `results/v4/b31.csv` | B3.1 beta ln(K/W), primary | OK |
| N77 | 0.043 | 0.0429 | `results/v4/b31.csv` | Holm p, primary | OK |
| N78 | -0.003 | -0.0033 | `results/v4/b31.csv` | beta with controls | OK |
| N79 | 0.79 | 0.794 | `results/v4/b31.csv` | p with controls | OK |
| N80 | -0.111 | -0.1111 | `results/v4/b31.csv` | beta without mixed-income correction | OK |
| N81 | 0.23 | 0.2287 | `results/v4/b31.csv` | p, IV median of other countries | OK |
| N82 | 0.044 | 0.0438 | `results/v4/b31_no_pcm_exploratory.csv` | exploratory beta, controls without PCM | OK |
| N83 | -1.15 | -1.1484 | `results/v4/b32.csv` | B3.2 lambda labour cost | OK |
| N84 | -3.93 | -3.9337 | `results/v4/b32.csv` | lower bootstrap bound | OK |
| N85 | 1.38 | 1.3767 | `results/v4/b32.csv` | upper bootstrap bound | OK |
| N86 | 0.25 | 0.252 | `results/v4/b32.csv` | median R2 labour | OK |
| N87 | 0.45 | 0.4466 | `results/v4/b32.csv` | median R2 gross output | OK |
| N88 | 0.01 | 0.0111 | `results/v4/b32.csv` | lambda gross output | OK |
| N89 | 0.0 | 0.0 | `results/v4/b32.csv` | share of perm placebos with lambda <= labour | OK |
| N90 | 0.074 | 0.0741 | `results/v4/b33.csv` | B3.3 beta h=1 | OK |
| N91 | 12595 | 12595 | `results/v4/b32.csv` | observations B3.2 primary | OK |
| N92 | 28 | 28 | `results/v4/b_panel.csv.gz` | countries in the KLEMS panel | OK |
| N93 | -0.033 | -0.0329 | `results/v4/b31.csv` | beta ln(K/W) with rent industries, no controls | OK |
| N94 | 0.61 | 0.6093 | `results/v4/b32.csv` | R2 labour with rent industries | OK |
| N95 | 0.67 | 0.6686 | `results/v4/b32.csv` | R2 GO with rent industries | OK |
| N96 | 0.007 | 0.007 | `results/report_tables.md` | share better, MEX open | OK |
| N97 | 0.26 | 0.259 | `results/report_tables.md` | share better, USA BEA open | OK |
| N98 | 0.45 | 0.452 | `results/report_tables.md` | share better, MEX closed uniform | OK |
| N99 | 0.68 | 0.681 | `results/report_tables.md` | share better, USA BEA closed uniform | OK |
| N100 | 0.127 | 0.1267 | `results/v2/pp_levels.csv` | MAWD PP(a) USA | OK |
| N101 | 0.157 | 0.1567 | `results/v2/pp_levels.csv` | MAWD PP(a) USA BEA | OK |
| N102 | 0.339 | 0.3389 | `results/v2/pp_levels.csv` | MAWD hours USA BEA | OK |
| N103 | 0.13 | 0.1324 | `results/v2/reduction_gap.csv` | max G of education / Cockshott reductions (DEU, USA, BEA; core, MAWD) | OK |
| N104 | 6 | 6.0 | `results/v4/b0b_counts.csv` | non-labour not worse than hours, (a) without rent | OK |
| N105 | 7 | 7 | `results/final/benchmarks_flat_power.csv` | flat beats hours without rent industries (d) | OK |
| N106 | 0.51 | 0.5098 | `results/v4/b32.csv` | median R2 value added (reference) | OK |
| N107 | 8 | 8 | `results/v4/a1_placebo.csv + results/v3/placebo_disp.csv` | A1(a): countries top 5% (of defined) | OK |
| N108 | 12 | 12 | `results/v4/a1_placebo.csv + results/v3/placebo_disp.csv` | A1(a): defined countries | OK |
| N109 | 9 | 9 | `results/v4/a1_placebo.csv + results/v3/placebo_disp.csv` | A1(b): countries top 5% | OK |
| N110 | 0.75 | 0.7535 | `results/v2/decomp_mixture.csv` | asym pairs where labour dominates (X adds <5%, labour adds >20%) | OK |
| N112 | 0.02 | 0.0177 | `results/v3/reduction_levels.csv` | median G 1.3a Hilferding | OK |
| N111 | 0.0 | 0.0 | `results/v4/a1_placebo.csv` | JPN A1(b) share of permutations better (MAWD) | OK |
| N113 | 13 | 13 | `results/final/benchmarks_flat_power.csv + results/report_tables.md` | countries where hours^0.5 beats PP uniform wage (MAWD) | OK |
| N114 | 7 | 7 | `results/final/benchmarks_flat_power.csv + results/report_tables.md` | countries where flat beats PP uniform wage | OK |
| N115 | 3 | 3 | `results/final/benchmarks_flat_power.csv + results/report_tables.md` | countries where flat beats PP actual wage | OK |
| N116 | 11 | 11 | `results/final/benchmarks_flat_power.csv` | flat beats hours by more than 0.005 MAWD | OK |
| N117 | 0.93 | 0.9275 | `results/v2/decomp_basis.csv` | share of winners whose own part O_X is worse than labour | OK |
| N118 | 0.56 | 0.5629 | `results/v2/decomp_basis.csv` | median labour share among bases worse than labour | OK |
| N119 | 368 | 368 | `results/v2/decomp_basis.csv` | winners excluding JPN (JPN not recomputed after fix) | OK |
| N120 | 734 | 734 | `results/v2/decomp_basis.csv` | pairs excluding JPN | OK |
| N121 | 0.6 | 0.6017 | `results/v2/decomp_basis.csv` | median labour share among winners excluding JPN | OK |
| N122 | 795 | 795 | `results/v2/decomp_mixture.csv` | country x basis pairs in the encompassing test | OK |
| N123 | 0.026 | 0.0261 | `results/v4/b31.csv` | p, IV lag 2 | OK |
| N124 | -0.046 | -0.0459 | `results/v4/b31.csv` | beta with rent industries and controls | OK |
| N125 | 0.005 | 0.0053 | `results/v4/b31.csv` | p with rent industries and controls | OK |
| N126 | 10848 | 10848 | `results/v4/b31.csv` | observations with controls | OK |
| N127 | 3 | 3.0 | `results/v4/b0b_counts.csv` | profit corrected for mixed income not worse than hours, (a) all | OK |
| N128 | 0.55 | 0.5483 | `results/v4/b0b_boot.csv` | capital: own permutations better, MAWD, without rent | OK |
| N129 | 0.4 | 0.3999 | `results/v4/b0b_boot.csv` | CFC: own permutations better, MAWD, without rent | OK |
| N130 | 3 | 3 | `results/v4/b0_summary.csv` | B0 all industries: informative vs physical flat, h=5 | OK |
| N131 | 1 | 1 | `results/v4/b0_summary.csv` | same, h=1 | OK |
| N132 | 0.48 | 0.4797 | `results/v4/b0_summary.csv` | B0 labour R2 DEU core h=5 (v3 spec) | OK |
| N133 | 6 | 6 | `results/v4/b0_summary.csv` | B0 core: labour beats 95% of permutations, h=1 | OK |
| N134 | 0.92 | 0.9228 | `results/v4/b0b_boot.csv` | capital: own permutations better, d, without rent | OK |
