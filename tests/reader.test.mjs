// JS reader test against the committed dist/ (schema v2). Mirrors how the web
// viewer parses the binary files. Skips cleanly when dist/ is absent (before the
// first generate). Run: node --test tests/reader.test.mjs
import { test } from "node:test";
import assert from "node:assert";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const indexPath = join(ROOT, "dist", "index.json");

test("dist/ schema v2 is readable", (t) => {
  if (!existsSync(indexPath)) {
    t.skip("no dist/index.json yet — nothing generated");
    return;
  }
  const index = JSON.parse(readFileSync(indexPath, "utf-8"));
  assert.ok(Array.isArray(index) && index.length > 0, "index has datasets");

  for (const entry of index) {
    const d = join(ROOT, "dist", entry.id);
    const meta = JSON.parse(readFileSync(join(d, "meta.json"), "utf-8"));
    const count = meta.count;
    assert.equal(meta.v, 2, `${entry.id}: schema v2`);

    const labels = readFileSync(join(d, "labels.tsv"), "utf-8")
      .split("\n").filter((l) => l.length > 0);
    assert.equal(labels.length, count, `${entry.id}: labels == count`);

    const pts = readFileSync(join(d, "points.f32"));
    assert.equal(pts.byteLength, 8 * count, `${entry.id}: points.f32 == 8*count`);

    const nb = readFileSync(join(d, "neighbors.bin"));
    assert.equal(nb.byteLength % (3 * count), 0, `${entry.id}: neighbors multiple of 3*count`);
    const k = nb.byteLength / (3 * count);
    const dv = new DataView(nb.buffer, nb.byteOffset, nb.byteLength);
    // every neighbor index must be < count
    for (let i = 0; i < count; i++) {
      for (let j = 0; j < k; j++) {
        const idx = dv.getUint16((i * k + j) * 3, true);
        assert.ok(idx < count, `${entry.id}: neighbor idx < count`);
      }
    }

    const vi8 = readFileSync(join(d, "vectors.i8"));
    assert.equal(vi8.byteLength, meta.dims_i8 * count, `${entry.id}: vectors.i8`);
    const vsc = readFileSync(join(d, "vscales.f32"));
    assert.equal(vsc.byteLength, 4 * count, `${entry.id}: vscales.f32`);
  }
});
