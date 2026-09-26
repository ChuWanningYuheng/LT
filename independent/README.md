# Independent implementation of the symmetric value-basis test

This is written from `SPEC.md` alone. It does not read or use any code or results from the first iteration.

## Run

```
cd /home/user/LT
OMP_NUM_THREADS=2 python independent/symtest.py
```

The run takes about 12 s. It writes these files:

- `independent/results/independent_shares.csv`: one row per country × year × capital × system (252 rows).
- `independent/results/checks.csv`: the correctness checks for each country × year × spec:
  - `max_spectral_radius`: the maximum over the labour S and every commodity-basis S.
  - `min_leontief_elem` and `min_leontief_diag`: the minimum element and minimum diagonal entry of (I−S)⁻¹, over the same set of matrices.
  - `labour_closed_vs_open_max_abs_z`: the maximum |z| difference between the labour basis computed explicitly in the unfolded 2n system and the open labour basis. It is NaN for `open`.
  - `max_abs_own_row_in_S`: the largest remaining entry in row k after it is zeroed.
  - `ok`: true when every check passes.
- `independent/results/figaro_rebuild_check.csv`: Zdom, Zimp and x for DEU 2015 and USA 2015, rebuilt from `figaro_2015.npz` and compared with the primitives.

## Why the closed-system labour basis equals the open one

Build the unfolded system with 2n commodities: n goods plus n types of labour power, where type j is the labour employed in industry j. The input matrix is

```
        goods   labour power
S* = [  M        B  ]     goods rows
     [  H        0  ]     labour rows,   H = diag(h)
```

where B[:, j] = β_dom · ω_j / h_j is the wage basket per hour of labour type j. The imported part is Sᵐ* = [[Mᵐ, Bᵐ], [0, 0]].

In the labour basis the labour rows are primary. That makes a* = (h, 0) the direct use of the primary resource and S = S* with the labour rows zeroed:

```
S = [ M  B ]      (I − S)⁻¹ = [ (I−M)⁻¹   (I−M)⁻¹ B ]
    [ 0  0 ]                  [    0          I     ]
```

The inverse is block upper-triangular. The goods block of λ = a*(I−S)⁻¹ is therefore h (I−M)⁻¹. The goods block of μ = 1ᵀSᵐ (I−S)⁻¹ is 1ᵀMᵐ (I−M)⁻¹, because the Bᵐ columns only reach the labour-power columns. These are exactly the open-system λ and μ. The normalisation Σ z_j x_j = Σ x_j and the metrics use only goods, so e and z are identical.

Put simply, the wage goods only enter the cost of labour power. Once labour power is treated as the primary resource, nothing flows back from it into the goods, so the closure makes no difference. The script builds the 2n system explicitly as a check, and the maximum discrepancy is 1.8e-15.

The folded form Mc = M + β ωᵀ is what you get by substituting the labour-power rows out: Mc = M + B H. With H = diag(h) and B[:, j] = β ω_j / h_j, this gives B H = β ωᵀ. Mc is therefore the right matrix for the commodity bases, where labour power is produced rather than primary.

## Implementation choices where SPEC is ambiguous

- **Which k are bases.** k is a basis when the row k of the system's own matrix has a positive sum: M for `open`, Mc for the closed systems. Row T of M is zero, so T is not a basis in `open` (n−1 bases). In the closed systems row T of Mc is β_dom,T·ωᵀ > 0, so T becomes a basis there (n bases). Bases are not restricted to the evaluated subsample, so T/U bases are counted if eligible.
- **Commodity basis k:** a = S₀[k,:], S = S₀ with row k zeroed. The imported matrix is S₀ᵐ (Mᵐ or Mcᵐ), with row k not zeroed, since it is a matrix of imported inputs.
- **Normalisation.** e is fitted on all n industries. z is then rescaled on the evaluated subsample, as SPEC says.
- **d** uses the unweighted angle between q and the vector of ones: cos θ = Σq / (√m ‖q‖).
- **"Strictly less".** `share_better = mean(metric_k < metric_labour)` over all bases counted in `n_bases`.
- **Excluded industries.** z ≤ 0 occurs only for industry T, where λ = μ = 0 because it has no inputs. T is excluded from evaluation anyway. No evaluated industry was ever dropped.
- **FIGARO x** is the row sum of Z plus the row sum of F for the country's rows. The USA aggregation follows the `+` parts of each primitive label, and U is dropped for both countries.
