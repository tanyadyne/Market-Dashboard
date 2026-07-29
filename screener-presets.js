(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ScreenerPresets = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const FLAGS = Object.freeze({
    SMA10_ABOVE_SMA50: 1,
    SMA10_BELOW_SMA50: 2,
    PRIOR_TWO_HAS_RED: 4,
    PRIOR_TWO_HAS_GREEN: 8,
    NEW_52W_HIGH_LAST_3: 16,
    NEW_52W_LOW_LAST_3: 32,
    SMA10_DECLINING: 64,
    SMA50_DECLINING: 128,
    EMA9_ABOVE_EMA21: 256,
    EMA9_BELOW_EMA21: 512,
  });

  const PRESETS = Object.freeze([
    {
      id: "vcp-liquid",
      group: "leaders",
      name: "VCP Coil (Liquid)",
      info: "This preset scans for liquid stocks that are coiled in a volatility contraction pattern.",
      criteria: [
        "Either VCP Coil definition 1 or definition 2 is active.",
        "RS position is Mid-Range or better.",
        "Market cap is above $2B.",
        "Price is above the 21 EMA by less than 1× ATR.",
        "10 SMA is above the 50 SMA.",
      ],
    },
    {
      id: "fresh-leader-tight",
      group: "leaders",
      name: "Above 9/21ema (tight)",
      info: "This preset scans for fresh leaders that are near a tight entry point.",
      criteria: [
        "Price is above the 21 EMA by less than 0.5× ATR, or above the 9 EMA by less than 0.5× ATR.",
        "9 EMA is above the 21 EMA.",
        "10 SMA is above the 50 SMA.",
        "At least one of the prior two daily candles was red.",
        "RS position is Mid-Range or better.",
      ],
    },
    {
      id: "fresh-leader-loose",
      group: "leaders",
      name: "Above 21/50sma (loose)",
      info: "This preset scans for fresh leaders that are near a loose entry point.",
      criteria: [
        "Price is above the 21 EMA by less than 1× ATR, or above the 50 SMA by less than 0.5× ATR.",
        "9 EMA is above the 21 EMA.",
        "10 SMA is above the 50 SMA.",
        "RS position is Mid-Range or better.",
      ],
    },
    {
      id: "new-52w-highs",
      group: "leaders",
      name: "New 52wk Highs",
      info: "This preset scans for stocks making a new 52-week high.",
      criteria: [
        "Price made a new 52-week high within the past three trading days, including today.",
        "Average 20-day dollar volume is above $100M.",
      ],
    },
    {
      id: "fresh-laggard-tight",
      group: "laggards",
      name: "Below 50sma (tight)",
      info: "This preset scans for lagging stocks that are trading below a tight pivot.",
      criteria: [
        "Price is below the 50 SMA by less than 0.5× ATR.",
        "9 EMA is below the 21 EMA.",
        "10 SMA is below the 50 SMA.",
        "At least one of the prior two daily candles was green.",
        "RS position is Mid-Range or worse.",
      ],
    },
    {
      id: "fresh-laggard-loose",
      group: "laggards",
      name: "Below 50sma (loose)",
      info: "This preset scans for lagging stocks that are trading below a loose pivot.",
      criteria: [
        "Price is below the 50 SMA by less than 1× ATR.",
        "9 EMA is below the 21 EMA.",
        "10 SMA is below the 50 SMA.",
        "10 SMA or 50 SMA is declining.",
        "RS position is Mid-Range or worse.",
      ],
    },
    {
      id: "new-52w-lows",
      group: "laggards",
      name: "New 52wk Lows",
      info: "This preset scans for stocks making a new 52-week low.",
      criteria: [
        "Price made a new 52-week low within the past three trading days, including today.",
        "Average 20-day dollar volume is above $100M.",
      ],
    },
  ]);

  const PRESET_BY_ID = Object.freeze(
    Object.fromEntries(PRESETS.map((preset) => [preset.id, preset])),
  );

  function hasFlag(stock, flag) {
    return ((Number(stock && stock.pf) || 0) & flag) === flag;
  }

  function atrDistance(stock, index) {
    if (!stock || !Array.isArray(stock.md)) return null;
    const value = Number(stock.md[index]);
    return Number.isFinite(value) ? value : null;
  }

  function priceAboveWithin(stock, index, limit) {
    const distance = atrDistance(stock, index);
    return distance != null && distance > 0 && distance < limit;
  }

  function priceBelowWithin(stock, index, limit) {
    const distance = atrDistance(stock, index);
    return distance != null && distance < 0 && Math.abs(distance) < limit;
  }

  function positionTier(rank, total) {
    if (rank == null || rank === "") return null;
    rank = Number(rank);
    total = Number(total);
    if (
      !Number.isFinite(rank)
      || rank <= 0
      || !Number.isFinite(total)
      || total <= 0
    ) {
      return null;
    }
    if (rank <= 100) return "Strong Leader";
    if (rank <= 200) return "Moderate Leader";
    if (rank / total <= 0.5) return "Mid-Range";
    if (rank / total <= 0.75) return "Moderate Laggard";
    return "Deep Laggard";
  }

  function isMidRangeOrBetter(stock, total) {
    return ["Strong Leader", "Moderate Leader", "Mid-Range"].includes(
      positionTier(stock && stock.w_rk, total),
    );
  }

  function isMidRangeOrWorse(stock, total) {
    return ["Mid-Range", "Moderate Laggard", "Deep Laggard"].includes(
      positionTier(stock && stock.w_rk, total),
    );
  }

  function definition1Active(stock) {
    return stock && (
      stock.vcp_coil_1 === true
      || (((Number(stock.sf) || 0) & 128) === 128)
    );
  }

  function definition2Active(stock) {
    return Boolean(stock && stock.vcp_coil_2 === true);
  }

  function matchesPreset(presetId, stock, rankedTotal) {
    switch (presetId) {
      case "vcp-liquid":
        return (
          (definition1Active(stock) || definition2Active(stock))
          && isMidRangeOrBetter(stock, rankedTotal)
          && Number(stock.mc) > 2e9
          && priceAboveWithin(stock, 1, 1)
          && hasFlag(stock, FLAGS.SMA10_ABOVE_SMA50)
        );
      case "fresh-leader-tight":
        return (
          (priceAboveWithin(stock, 1, 0.5) || priceAboveWithin(stock, 0, 0.5))
          && hasFlag(stock, FLAGS.EMA9_ABOVE_EMA21)
          && hasFlag(stock, FLAGS.SMA10_ABOVE_SMA50)
          && hasFlag(stock, FLAGS.PRIOR_TWO_HAS_RED)
          && isMidRangeOrBetter(stock, rankedTotal)
        );
      case "fresh-leader-loose":
        return (
          (priceAboveWithin(stock, 1, 1) || priceAboveWithin(stock, 2, 0.5))
          && hasFlag(stock, FLAGS.EMA9_ABOVE_EMA21)
          && hasFlag(stock, FLAGS.SMA10_ABOVE_SMA50)
          && isMidRangeOrBetter(stock, rankedTotal)
        );
      case "new-52w-highs":
        return (
          hasFlag(stock, FLAGS.NEW_52W_HIGH_LAST_3)
          && Number(stock.dv) > 100e6
        );
      case "fresh-laggard-tight":
        return (
          priceBelowWithin(stock, 2, 0.5)
          && hasFlag(stock, FLAGS.EMA9_BELOW_EMA21)
          && hasFlag(stock, FLAGS.SMA10_BELOW_SMA50)
          && hasFlag(stock, FLAGS.PRIOR_TWO_HAS_GREEN)
          && isMidRangeOrWorse(stock, rankedTotal)
        );
      case "fresh-laggard-loose":
        return (
          priceBelowWithin(stock, 2, 1)
          && hasFlag(stock, FLAGS.EMA9_BELOW_EMA21)
          && hasFlag(stock, FLAGS.SMA10_BELOW_SMA50)
          && (
            hasFlag(stock, FLAGS.SMA10_DECLINING)
            || hasFlag(stock, FLAGS.SMA50_DECLINING)
          )
          && isMidRangeOrWorse(stock, rankedTotal)
        );
      case "new-52w-lows":
        return (
          hasFlag(stock, FLAGS.NEW_52W_LOW_LAST_3)
          && Number(stock.dv) > 100e6
        );
      default:
        return true;
    }
  }

  return Object.freeze({
    FLAGS,
    PRESETS,
    PRESET_BY_ID,
    matchesPreset,
    positionTier,
  });
});
