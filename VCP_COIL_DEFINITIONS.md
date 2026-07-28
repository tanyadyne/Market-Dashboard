# VCP coil definitions

Both signals are calculated from adjusted daily OHLC data and published on
every stock row in `leaders.json`. Their aligned daily histories live under
the same keys in each ticker entry in `leaders_score_history.json`.

## `vcp_coil_1`

Existing RS Ranking filter definition:

1. Calculate 21-day average daily range percentage.
2. Calculate 10-day close spread and full high-low spread.
3. Divide both spreads by ADR and average them.
4. Min-max normalize the current value against the last 100 observations.
5. Signal when the score is less than or equal to 10.

This remains setup flag bit `128`. Existing filters and overview panels keep
using that bit, so their behavior does not change.

## `vcp_coil_2`

Direct translation of the supplied Pine Script's `coil_detected` value under
`Coil Sensitivity = "Balanced"`:

```text
balanced_coil OR strict_coil
```

It includes Pine-compatible ATR/Wilder RMA, EMA, HMA, candle contraction,
RMV, alternate RMV, trend, MA-spread, and price-proximity calculations.

It does not include the source indicator's stateful coil boxes, coil breakout,
`Coil Active`, or `Inside Box` outputs. Those are separate Pine signals.

Insufficient history is stored as `null`. When these fields first enter an
existing history file, prior dates are padded with `null`.
