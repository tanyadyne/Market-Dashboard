"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const {
  FLAGS,
  PRESETS,
  matchesPreset,
} = require("../screener-presets.js");
const {
  validatePresetData,
} = require("./validate_screener_preset_data.js");

const rankedTotal = 1000;

function stock(overrides = {}) {
  return {
    w_rk: 250,
    mc: 3e9,
    dv: 150e6,
    md: [0.2, 0.3, 0.25, 0, 0],
    pf: 0,
    sf: 0,
    vcp_coil_1: false,
    vcp_coil_2: false,
    ...overrides,
  };
}

function hasAll(...flags) {
  return flags.reduce((value, flag) => value | flag, 0);
}

function withoutFlag(value, flag) {
  return { ...value, pf: value.pf & ~flag };
}

const fixtures = {
  "vcp-liquid": stock({
    vcp_coil_2: true,
    pf: FLAGS.SMA10_ABOVE_SMA50,
  }),
  "fresh-leader-tight": stock({
    pf: hasAll(
      FLAGS.EMA9_ABOVE_EMA21,
      FLAGS.SMA10_ABOVE_SMA50,
      FLAGS.PRIOR_TWO_HAS_RED,
    ),
  }),
  "fresh-leader-loose": stock({
    pf: hasAll(
      FLAGS.EMA9_ABOVE_EMA21,
      FLAGS.SMA10_ABOVE_SMA50,
    ),
  }),
  "new-52w-highs": stock({
    pf: FLAGS.NEW_52W_HIGH_LAST_3,
  }),
  "fresh-laggard-tight": stock({
    md: [-0.2, -0.3, -0.25, 0, 0],
    pf: hasAll(
      FLAGS.EMA9_BELOW_EMA21,
      FLAGS.SMA10_BELOW_SMA50,
      FLAGS.PRIOR_TWO_HAS_GREEN,
    ),
  }),
  "fresh-laggard-loose": stock({
    md: [-0.2, -0.3, -0.75, 0, 0],
    pf: hasAll(
      FLAGS.EMA9_BELOW_EMA21,
      FLAGS.SMA10_BELOW_SMA50,
      FLAGS.SMA50_DECLINING,
    ),
  }),
  "new-52w-lows": stock({
    pf: FLAGS.NEW_52W_LOW_LAST_3,
  }),
};

assert.strictEqual(PRESETS.length, 7);
assert.deepStrictEqual(
  PRESETS.filter((preset) => preset.group === "leaders").map((preset) => preset.name),
  [
    "VCP Coil (Liquid)",
    "Above 9/21ema (tight)",
    "Above 21/50sma (loose)",
    "New 52wk Highs",
  ],
);
assert.deepStrictEqual(
  PRESETS.filter((preset) => preset.group === "laggards").map((preset) => preset.name),
  [
    "Below 50sma (tight)",
    "Below 50sma (loose)",
    "New 52wk Lows",
  ],
);
for (const preset of PRESETS) {
  assert.strictEqual(
    matchesPreset(preset.id, fixtures[preset.id], rankedTotal),
    true,
    `${preset.name} should match its complete fixture`,
  );
}

const requiredFailures = {
  "vcp-liquid": [
    { ...fixtures["vcp-liquid"], vcp_coil_2: false },
    { ...fixtures["vcp-liquid"], w_rk: 501 },
    { ...fixtures["vcp-liquid"], mc: 2e9 },
    { ...fixtures["vcp-liquid"], md: [0.2, -0.1, 0.25, 0, 0] },
    withoutFlag(fixtures["vcp-liquid"], FLAGS.SMA10_ABOVE_SMA50),
  ],
  "fresh-leader-tight": [
    { ...fixtures["fresh-leader-tight"], md: [0.5, 0.5, 0.25, 0, 0] },
    withoutFlag(fixtures["fresh-leader-tight"], FLAGS.EMA9_ABOVE_EMA21),
    withoutFlag(fixtures["fresh-leader-tight"], FLAGS.SMA10_ABOVE_SMA50),
    withoutFlag(fixtures["fresh-leader-tight"], FLAGS.PRIOR_TWO_HAS_RED),
    { ...fixtures["fresh-leader-tight"], w_rk: 501 },
  ],
  "fresh-leader-loose": [
    { ...fixtures["fresh-leader-loose"], md: [0.2, 1, 0.5, 0, 0] },
    withoutFlag(fixtures["fresh-leader-loose"], FLAGS.EMA9_ABOVE_EMA21),
    withoutFlag(fixtures["fresh-leader-loose"], FLAGS.SMA10_ABOVE_SMA50),
    { ...fixtures["fresh-leader-loose"], w_rk: 501 },
  ],
  "new-52w-highs": [
    withoutFlag(fixtures["new-52w-highs"], FLAGS.NEW_52W_HIGH_LAST_3),
    { ...fixtures["new-52w-highs"], dv: 100e6 },
  ],
  "fresh-laggard-tight": [
    { ...fixtures["fresh-laggard-tight"], md: [-0.2, -0.3, -0.5, 0, 0] },
    withoutFlag(fixtures["fresh-laggard-tight"], FLAGS.EMA9_BELOW_EMA21),
    withoutFlag(fixtures["fresh-laggard-tight"], FLAGS.SMA10_BELOW_SMA50),
    withoutFlag(fixtures["fresh-laggard-tight"], FLAGS.PRIOR_TWO_HAS_GREEN),
    { ...fixtures["fresh-laggard-tight"], w_rk: 200 },
  ],
  "fresh-laggard-loose": [
    { ...fixtures["fresh-laggard-loose"], md: [-0.2, -0.3, -1, 0, 0] },
    withoutFlag(fixtures["fresh-laggard-loose"], FLAGS.EMA9_BELOW_EMA21),
    withoutFlag(fixtures["fresh-laggard-loose"], FLAGS.SMA10_BELOW_SMA50),
    {
      ...fixtures["fresh-laggard-loose"],
      pf: hasAll(FLAGS.EMA9_BELOW_EMA21, FLAGS.SMA10_BELOW_SMA50),
    },
    { ...fixtures["fresh-laggard-loose"], w_rk: 200 },
  ],
  "new-52w-lows": [
    withoutFlag(fixtures["new-52w-lows"], FLAGS.NEW_52W_LOW_LAST_3),
    { ...fixtures["new-52w-lows"], dv: 100e6 },
  ],
};
for (const [presetId, failures] of Object.entries(requiredFailures)) {
  failures.forEach((candidate, index) => {
    assert.strictEqual(
      matchesPreset(presetId, candidate, rankedTotal),
      false,
      `${presetId} required-condition counterexample ${index + 1} must fail`,
    );
  });
}

assert.strictEqual(
  matchesPreset(
    "vcp-liquid",
    { ...fixtures["vcp-liquid"], vcp_coil_2: false, sf: 128 },
    rankedTotal,
  ),
  true,
  "VCP Liquid should retain definition 1 bit-128 compatibility",
);

for (const md of [
  [0.6, 0.4, 0.25, 0, 0],
  [0.4, 0.6, 0.25, 0, 0],
]) {
  assert.strictEqual(
    matchesPreset(
      "fresh-leader-tight",
      { ...fixtures["fresh-leader-tight"], md },
      rankedTotal,
    ),
    true,
    "Tight leader accepts either the 21 EMA or 9 EMA proximity branch",
  );
}

for (const md of [
  [0.2, 0.9, 0.5, 0, 0],
  [0.2, 1, 0.4, 0, 0],
]) {
  assert.strictEqual(
    matchesPreset(
      "fresh-leader-loose",
      { ...fixtures["fresh-leader-loose"], md },
      rankedTotal,
    ),
    true,
    "Loose leader accepts either the 21 EMA or 50 SMA proximity branch",
  );
}

assert.strictEqual(
  matchesPreset(
    "fresh-laggard-loose",
    {
      ...fixtures["fresh-laggard-loose"],
      pf: hasAll(
        FLAGS.EMA9_BELOW_EMA21,
        FLAGS.SMA10_BELOW_SMA50,
        FLAGS.SMA10_DECLINING,
      ),
    },
    rankedTotal,
  ),
  true,
  "Loose laggard accepts a declining 10 SMA without a declining 50 SMA",
);

for (const presetId of ["fresh-leader-tight", "fresh-laggard-tight"]) {
  assert.strictEqual(
    matchesPreset(
      presetId,
      { ...fixtures[presetId], w_rk: 500 },
      rankedTotal,
    ),
    true,
    "The 50% rank boundary is Mid-Range in both directional preset sets",
  );
}

assert.strictEqual(
  matchesPreset(
    "fresh-leader-tight",
    { ...fixtures["fresh-leader-tight"], md: [0.5, 0.5, 0, 0, 0] },
    rankedTotal,
  ),
  false,
  "ATR limits are strict less-than boundaries",
);

assert.strictEqual(
  matchesPreset(
    "fresh-leader-tight",
    { ...fixtures["fresh-leader-tight"], md: [0, 0, 0.25, 0, 0] },
    rankedTotal,
  ),
  false,
  "A zero distance is not above a moving average",
);

assert.strictEqual(
  matchesPreset(
    "fresh-laggard-tight",
    { ...fixtures["fresh-laggard-tight"], md: [-0.2, -0.3, 0, 0, 0] },
    rankedTotal,
  ),
  false,
  "A zero distance is not below a moving average",
);

assert.strictEqual(
  matchesPreset(
    "new-52w-highs",
    { ...fixtures["new-52w-highs"], dv: 100e6 },
    rankedTotal,
  ),
  false,
  "Average dollar volume must be strictly above $100M",
);

assert.strictEqual(
  matchesPreset(
    "vcp-liquid",
    { ...fixtures["vcp-liquid"], mc: 2e9 },
    rankedTotal,
  ),
  false,
  "Market cap must be strictly above $2B",
);

assert.strictEqual(
  matchesPreset(
    "fresh-laggard-loose",
    {
      ...fixtures["fresh-laggard-loose"],
      pf: hasAll(FLAGS.EMA9_BELOW_EMA21, FLAGS.SMA10_BELOW_SMA50),
    },
    rankedTotal,
  ),
  false,
  "Loose laggard requires at least one declining SMA",
);

assert.strictEqual(
  matchesPreset(
    "fresh-laggard-tight",
    { ...fixtures["fresh-laggard-tight"], w_rk: 200 },
    rankedTotal,
  ),
  false,
  "Moderate Leaders are not Mid-Range or worse",
);

assert.strictEqual(
  matchesPreset(
    "fresh-leader-tight",
    { ...fixtures["fresh-leader-tight"], w_rk: 751 },
    rankedTotal,
  ),
  false,
  "Deep Laggards are not Mid-Range or better",
);

assert.strictEqual(
  matchesPreset(
    "fresh-leader-tight",
    { ...fixtures["fresh-leader-tight"], w_rk: null },
    rankedTotal,
  ),
  false,
  "Unranked stocks must not be treated as Strong Leaders",
);

const html = fs.readFileSync(path.join(__dirname, "..", "screener.html"), "utf8");
for (const id of ["filterBtn", "filterMenu", "presetsBtn", "presetMenu"]) {
  const count = html.split(`id="${id}"`).length - 1;
  assert.strictEqual(count, 1, `${id} must be unique`);
}
assert.ok(
  html.indexOf('id="filterWrap"') < html.indexOf('id="viewToggle"'),
  "Filter cog belongs immediately before the view toggle in the header controls",
);
assert.ok(
  html.indexOf('id="presetWrap"') < html.indexOf('id="si"'),
  "Presets belongs before the RS Ranking search input",
);
assert.ok(
  html.includes('class="preset-icon"') &&
    html.includes("height:12px") &&
    html.includes("flex:0 0 12px"),
  "The Presets funnel icon must match the label height without shrinking",
);
assert.ok(
  html.includes('stroke="currentColor"') &&
    html.includes('<path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3Z"></path>'),
  "The Presets button must render a gray-by-default funnel icon",
);
assert.ok(
  html.includes(".preset-btn.has-preset .preset-icon{filter:drop-shadow"),
  "The Presets funnel icon must glow with the active purple state",
);
assert.ok(
  html.includes('<script src="screener-presets.js"></script>'),
  "The preset predicate bundle must load on the screener page",
);
assert.ok(
  html.includes('onclick="resetAllFilters()"'),
  "The dropdown reset action must clear preset and manual filters",
);
assert.ok(
  !html.includes("preset-section-title") && !html.includes("preset-criteria-"),
  "The preset list must not render leader/laggard sections or criteria tooltips",
);
assert.ok(
  html.includes('class="preset-help"') && html.includes('>i</button>'),
  "Each preset must expose its info description through an i tooltip",
);

const validationRows = Object.entries(fixtures).map(([ticker, value]) => ({
  ...value,
  t: ticker,
}));
const validationBaselines = Object.fromEntries(
  validationRows.map((row) => [
    row.t,
    { _preset_last_date: "2026-07-28" },
  ]),
);
const validation = validatePresetData(
  { e: validationRows },
  { d: validationBaselines },
);
assert.strictEqual(validation.baselineCoverage, 1);
assert.strictEqual(validation.rowCount, PRESETS.length);
assert.throws(
  () => validatePresetData(
    { e: [{ ...validationRows[0], pf: undefined }] },
    { d: { [validationRows[0].t]: validationBaselines[validationRows[0].t] } },
  ),
  /invalid pf value/,
);

console.log("screener preset tests: ok");
