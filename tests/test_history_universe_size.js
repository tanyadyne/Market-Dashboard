const assert = require('assert');
const fs = require('fs');

const source = fs.readFileSync('screener.html', 'utf8');

assert.match(
  source,
  /function historicalRankedTotalAt\(index,fallbackTotal\)\{[\s\S]*?HISTORY\.wr_total/,
  'history rendering should read the stored per-session universe size',
);
assert.match(
  source,
  /const snapshotTotal=historicalRankedTotalAt\(i,total\);[\s\S]*?getPosition\(rank,snapshotTotal\)/,
  'history charts should classify each point against its own session universe',
);
assert.match(
  source,
  /Math\.min\(snapshotTotal,Number\(rank\)\)/,
  'history scores should use the per-session denominator',
);
assert.match(
  source,
  /getPosition\(wr,historicalRankedTotalAt\(i,total\)\)/,
  'history table labels should use the per-session denominator',
);
assert.match(
  source,
  /#\$\{nearest\.rank\} of \$\{nearest\.total\}/,
  'history tooltip should disclose the session universe size',
);

console.log('RS history universe size: OK');
