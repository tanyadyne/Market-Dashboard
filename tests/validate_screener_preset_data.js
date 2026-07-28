"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const { PRESETS, matchesPreset } = require("../screener-presets.js");

function validatePresetData(leaders, baselinePayload) {
  const rows = leaders.e;
  const baselines = baselinePayload.d;

  assert.ok(Array.isArray(rows) && rows.length > 0, "leaders.json has no stocks");
  assert.ok(
    baselines && typeof baselines === "object" && Object.keys(baselines).length > 0,
    "leaders_intraday_baselines.json has no baselines",
  );

  const requiredFields = ["pf", "vcp_coil_1", "vcp_coil_2"];
  for (const row of rows) {
    for (const field of requiredFields) {
      assert.ok(
        Object.prototype.hasOwnProperty.call(row, field),
        `${row.t || "<unknown>"} is missing ${field}`,
      );
    }
    assert.ok(
      row.pf === null || Number.isInteger(row.pf),
      `${row.t || "<unknown>"} has an invalid pf value`,
    );
    for (const field of ["vcp_coil_1", "vcp_coil_2"]) {
      assert.ok(
        row[field] === null || typeof row[field] === "boolean",
        `${row.t || "<unknown>"} has an invalid ${field} value`,
      );
    }
  }

  const eligibleRows = rows.filter((row) => !row.po && row.t);
  const presetReadyBaselines = eligibleRows.filter(
    (row) => baselines[row.t] && baselines[row.t]._preset_last_date,
  );
  const baselineCoverage = presetReadyBaselines.length / eligibleRows.length;
  assert.ok(
    baselineCoverage >= 0.9,
    `preset baseline coverage ${(baselineCoverage * 100).toFixed(1)}% is below 90%`,
  );

  const rankedTotal = rows.filter((row) => !row.po && row.w_rk != null).length;
  assert.ok(rankedTotal > 0, "leaders.json has no ranked stocks");

  const counts = Object.fromEntries(
    PRESETS.map((preset) => [
      preset.name,
      rows.filter((row) => matchesPreset(preset.id, row, rankedTotal)).length,
    ]),
  );
  return {
    baselineCoverage,
    eligibleCount: eligibleRows.length,
    presetReadyCount: presetReadyBaselines.length,
    rankedTotal,
    rowCount: rows.length,
    counts,
  };
}

if (require.main === module) {
  const root = path.join(__dirname, "..");
  const leaders = JSON.parse(
    fs.readFileSync(path.join(root, "leaders.json"), "utf8"),
  );
  const baselinePayload = JSON.parse(
    fs.readFileSync(path.join(root, "leaders_intraday_baselines.json"), "utf8"),
  );
  const validation = validatePresetData(leaders, baselinePayload);

  console.log(
    `preset data valid: ${validation.rowCount} stocks, `
    + `${validation.presetReadyCount}/${validation.eligibleCount} baselines `
    + `(${(validation.baselineCoverage * 100).toFixed(1)}%)`,
  );
  for (const preset of PRESETS) {
    console.log(`  ${preset.name}: ${validation.counts[preset.name]}`);
  }
}

module.exports = { validatePresetData };
