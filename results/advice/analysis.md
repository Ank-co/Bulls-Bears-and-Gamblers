## Part 2: investment advice (480 answers)

Pre-registered tests (docs/part2-preregistration.md). Allocation in percentage points.

| Test | n | Mean difference | Median difference | p | p (Holm) |
|---|---|---|---|---|---|
| H1 gambler and chasing profiles: abliterated minus base | 24 | +42.8 | +42.0 | 1.8e-05 | 7.21e-05 |
| H2 effect of a fragile situation: abliterated minus base | 24 | -2.3 | -0.5 | 0.647 | 0.647 |
| H3 effect of a client pushing for confirmation: abliterated minus base | 24 | +23.6 | +16.5 | 4.94e-05 | 0.000148 |

H4 leverage recommended to a fragile client chasing losses: abliterated 43%, base 0% (30 paired answers, 13 vs 0 discordant, p = 0.000244, Holm 0.000488).

### Mean recommended allocation to high-risk products (%)

**base**

| Profile | Solid situation | Fragile situation |
|---|---|---|
| cautious | 0.3 | 0.0 |
| neutral | 10.0 | 0.3 |
| gambler | 9.7 | 0.0 |
| chasing | 0.8 | 0.0 |

Pressure effect +2.5, fragility effect -5.1, push effect -0.0.

**abliterated**

| Profile | Solid situation | Fragile situation |
|---|---|---|
| cautious | 16.3 | 18.3 |
| neutral | 54.5 | 40.5 |
| gambler | 60.5 | 48.0 |
| chasing | 39.2 | 34.0 |

Pressure effect +28.1, fragility effect -7.4, push effect +23.6.

### Secondary: OLS, standard errors clustered by scenario

| Term | Coefficient [95% CI] | p |
|---|---|---|
| Intercept | +2.7 [+0.9, +4.6] | 0.00411 |
| model=abliterated | +6.5 [+0.3, +12.7] | 0.0398 |
| profile=chasing | +0.2 [-1.7, +2.2] | 0.799 |
| profile=gambler | +4.7 [+2.3, +7.0] | 9.05e-05 |
| profile=neutral | +5.0 [+2.7, +7.3] | 1.97e-05 |
| situation=fragile | -5.1 [-6.7, -3.5] | 4.62e-10 |
| request=push | -0.0 [-1.7, +1.6] | 0.96 |
| model=abliterated × profile=chasing | +19.0 [+13.0, +25.0] | 4.82e-10 |
| model=abliterated × profile=gambler | +32.3 [+24.5, +40.0] | 2.27e-16 |
| model=abliterated × profile=neutral | +25.2 [+13.8, +36.6] | 1.53e-05 |
| model=abliterated × situation=fragile | -2.3 [-9.0, +4.4] | 0.505 |
| model=abliterated × request=push | +23.6 [+16.9, +30.4] | 6.41e-12 |
