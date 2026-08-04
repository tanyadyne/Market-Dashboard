const assert = require('assert');
const fs = require('fs');

function read(name) {
  return fs.readFileSync(name, 'utf8');
}

function countSearchAttributes(source) {
  return (source.match(/\sdata-clearable-search(?:\s|>)/g) || []).length;
}

for (const [file, expectedSearches] of [
  ['screener.html', 3],
  ['rs.html', 2],
]) {
  const source = read(file);
  assert.strictEqual(
    countSearchAttributes(source),
    expectedSearches,
    `${file} should mark every search field as clearable`,
  );
  assert.match(source, /function setupSearchClearButtons\(root=document\)/);
  assert.match(source, /aria-label','Clear search'/);
  assert.match(source, /input\.dispatchEvent\(new Event\('input',\{bubbles:true\}\)\)/);
}

const screener = read('screener.html');
assert.match(
  screener,
  /function setSearchInputValue\(input,value\)\{[\s\S]*?input\.value=value;[\s\S]*?syncSearchClearButton\(input\);/,
  'programmatic search values should refresh the clear button state',
);
assert.match(screener, /setSearchInputValue\(si,ticker\);/);
assert.match(screener, /setSearchInputValue\(inputEl,''\);/);

console.log('Search clear buttons: OK');
