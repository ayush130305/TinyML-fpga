// crosscheck.js - node side of tools/crosscheck.py: reads {spec, configs} as JSON on stdin and prints, per config, the
// estimate, verdict, warnings and a SHA-256 of every generated file (path -> hash), plus the suggest() result.
const crypto = require("crypto");
const C = require("../configurator/configurator.js");
let data = "";
process.stdin.on("data", (d) => (data += d));
process.stdin.on("end", () => {
  const { spec, configs, suggests } = JSON.parse(data);
  const out = { configs: [], suggests: [] };
  for (const cfg of configs) {
    const files = {};
    for (const [p, t] of C.buildFiles(spec, cfg)) files[p] = crypto.createHash("sha256").update(t, "utf8").digest("hex");
    const e = C.estimate(spec, cfg);
    out.configs.push({ est: e, verdict: C.verdict(e, C.normalize(spec, cfg).budget), warnings: C.warnings(spec, cfg), files });
  }
  for (const s of suggests) out.suggests.push(C.suggest(spec, s.budget, s.bus, s.dsp, s.must, {}));
  process.stdout.write(JSON.stringify(out));
});
