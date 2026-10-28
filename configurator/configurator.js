// configurator.js - the same algorithm as firmware/tinyml_fpga/configure.py, for the web page and for node.
// Both read configurator/spec.json; tools/crosscheck.py compares their output file by file.
(function (root) {
  "use strict";
  var METRICS = ["lut4", "lut4_nodsp", "ff", "dsp", "ram4k"];
  var FIT_LIMIT = 0.85;
  var RAM = [["W_AW", 12, 8, 15], ["X_AW", 10, 6, 12], ["Y_AW", 8, 4, 12], ["P_AW", 10, 6, 12], ["D_AW", 9, 7, 12]];

  // Round half up.
  function rnd(x) { return Math.floor(x + 0.5); }

  // Fill in defaults and check a configuration; throws an Error on a bad one.
  function normalize(spec, cfg) {
    var c = {
      lanes: parseInt(cfg.lanes === undefined ? 1 : cfg.lanes, 10),
      rq: parseInt(cfg.rq === undefined ? 0 : cfg.rq, 10),
      features: Array.from(new Set(cfg.features || [])).sort(),
      bus: cfg.bus || "uart",
      dsp: cfg.dsp === undefined ? true : !!cfg.dsp,
      budget: parseInt(cfg.budget === undefined ? 8000 : cfg.budget, 10),
      options: {}
    };
    var src = cfg.options || {}, k;
    for (k in src) c.options[k] = src[k];
    if (spec.lanes.indexOf(c.lanes) < 0) throw new Error("lanes must be one of " + spec.lanes);
    if (c.rq !== 0 && c.rq !== 1) throw new Error("rq must be 0 (serial) or 1 (pipelined)");
    var ids = spec.features.map(function (f) { return f.id; });
    c.features.forEach(function (f) { if (ids.indexOf(f) < 0) throw new Error("unknown feature " + f); });
    if (!spec.buses[c.bus]) throw new Error("unknown bus " + c.bus);
    var o = c.options;
    RAM.forEach(function (r) {
      o[r[0]] = parseInt(o[r[0]] === undefined ? r[1] : o[r[0]], 10);
      if (!(o[r[0]] >= r[2] && o[r[0]] <= r[3])) throw new Error(r[0] + " must be " + r[2] + ".." + r[3]);
    });
    o.clk_hz = parseInt(o.clk_hz === undefined ? 100000000 : o.clk_hz, 10);
    o.baud = parseInt(o.baud === undefined ? 921600 : o.baud, 10);
    o.i2c_addr = parseInt(o.i2c_addr === undefined ? 0x2A : o.i2c_addr, 10);
    if (!(o.i2c_addr >= 8 && o.i2c_addr <= 119)) throw new Error("i2c_addr must be 8..119");
    return c;
  }

  // Cost of a configuration.
  function estimate(spec, cfg) {
    var c = normalize(spec, cfg), tot = {}, m, i, j;
    var l = String(c.lanes), r = String(c.rq);
    METRICS.forEach(function (m) { tot[m] = spec.base[l + "," + r][m]; });
    c.features.forEach(function (f) { METRICS.forEach(function (m) { tot[m] += spec.delta[l][f][m]; }); });
    var pl = c.lanes <= 2 ? "1" : "4", fs = c.features;
    for (i = 0; i < fs.length; i++) for (j = i + 1; j < fs.length; j++) {
      var corr = spec.pair[pl][fs[i] + "+" + fs[j]];
      if (corr) METRICS.forEach(function (m) { tot[m] += corr[m]; });
    }
    METRICS.forEach(function (m) { tot[m] += spec.buses[c.bus].cost[m]; });
    var lut = c.dsp ? tot.lut4 : tot.lut4_nodsp, out = {};
    METRICS.forEach(function (m) { out[m] = Math.max(0, tot[m]); });
    out.lut = Math.max(0, lut);
    out.le = rnd(Math.max(0, lut) * spec.le_per_lut4);
    out.ff_est = rnd(Math.max(0, tot.ff) * spec.ff_per_ff);
    return out;
  }

  function verdict(est, budget) {
    if (est.le <= FIT_LIMIT * budget) return "fits";
    return est.le <= budget ? "tight" : "over";
  }

  function warnings(spec, cfg) {
    var c = normalize(spec, cfg), o = c.options, w = [];
    if (Math.floor((1 << o.P_AW) / 16) < c.lanes) w.push("P_AW=" + o.P_AW + " leaves less than one parameter record per lane");
    if (c.features.indexOf("HAS_QUEUE") >= 0 && (1 << o.D_AW) < 64) w.push("D_AW too small for one descriptor");
    if (c.features.indexOf("HAS_CB") >= 0 && c.features.indexOf("HAS_W4") < 0) w.push("HAS_CB also stores 4-bit indices; HAS_W4 is not needed with it");
    if (c.bus === "spi") w.push("SPI clock must be at most clk/6");
    if (c.bus === "i2c") w.push("I2C clock must be at most clk/16; no clock stretching");
    return w;
  }

  function setParam(text, name, value) {
    var re = new RegExp("(parameter\\s+" + name + "\\s*=\\s*)([^,\\s/]+)");
    return text.replace(re, function (m, a) { return a + String(value); });
  }

  function packageFiles(spec, cfg) {
    var c = normalize(spec, cfg), b = spec.buses[c.bus];
    return spec.core.concat(spec.rq_files[String(c.rq)], b.files, [b.top]);
  }

  function hex2(n) { var s = n.toString(16).toUpperCase(); return s.length < 2 ? "0" + s : s; }

  function topSource(spec, cfg) {
    var c = normalize(spec, cfg), b = spec.buses[c.bus], o = c.options, t = spec.sources[b.top];
    var vals = { LANES: c.lanes, RQ_MODE: c.rq, W_AW: o.W_AW, X_AW: o.X_AW, Y_AW: o.Y_AW, P_AW: o.P_AW, D_AW: o.D_AW };
    spec.features.forEach(function (f) { vals[f.id] = c.features.indexOf(f.id) >= 0 ? 1 : 0; });
    if (c.bus === "uart") { vals.CLK_HZ = o.clk_hz; vals.BAUD = o.baud; }
    if (c.bus === "i2c") vals.I2C_ADDR = "7'h" + hex2(o.i2c_addr);
    Object.keys(vals).sort().forEach(function (k) { t = setParam(t, k, vals[k]); });
    return t;
  }

  function pythonLine(c) {
    var o = c.options;
    return {
      uart: 'tinyml_fpga.load("model.tmf", backend="uart:COM5", baud=' + o.baud + ")",
      spi: 'tinyml_fpga.load("model.tmf", backend="spi:0.0")',
      i2c: 'tinyml_fpga.load("model.tmf", backend="i2c:1:0x' + hex2(o.i2c_addr) + '")',
      axil: 'tinyml_fpga.load("model.tmf", backend="axi:0x40000000")'
    }[c.bus];
  }

  function readme(spec, cfg) {
    var c = normalize(spec, cfg), e = estimate(spec, c), b = spec.buses[c.bus], o = c.options;
    var feats = c.features.join(", ") || "none";
    var L = [
      "# tinyml-fpga configured engine", "",
      "Generated by the tinyml-fpga configurator (version " + spec.version + ").", "",
      "## Configuration", "",
      "- lanes: " + c.lanes,
      "- requantiser: " + (c.rq ? "pipelined" : "serial"),
      "- features: " + feats,
      "- bus: " + c.bus + " (top module `" + b.top.slice(0, -2) + "`)",
      "- RAM: W " + (1 << o.W_AW) + " B, X " + (1 << o.X_AW) + " B, Y " + (1 << o.Y_AW) + " B, P " + (1 << o.P_AW) + " B", "",
      "## Estimated size", "",
      "- about " + e.le + " logic elements (budget " + c.budget + ", " + verdict(e, c.budget) + ")",
      "- about " + e.ff_est + " flip-flops, " + e.dsp + " multiplier blocks, " + e.ram4k + " 4-kbit RAM blocks",
      "- " + (c.dsp ? "multipliers on DSP blocks" : "multipliers built from LUTs"), "",
      "Estimate from Yosys iCE40 counts scaled to Efinity Trion (calibrated on one design). Run your own synthesis for the real number.", "",
      "## Use", "",
      "1. Add every file in `rtl/` to your FPGA project and make `" + b.top.slice(0, -2) + "` the top module.",
      "2. Connect its ports in your pin constraints (clock, reset, the bus pins, optional status LEDs).",
      "3. On the host: `pip install -e .` from the tinyml-fpga repository, convert a model with `python -m tinyml_fpga convert`, then", "",
      "```python", "import tinyml_fpga", "m = " + pythonLine(c), "```", ""
    ];
    warnings(spec, c).forEach(function (w) { L.push("- note: " + w); });
    return L.join("\n") + "\n";
  }

  // Sorted-key JSON with 2-space indent, like Python's json.dumps(indent=2, sort_keys=True).
  function pyJson(v, ind) {
    ind = ind || "";
    if (v === null || typeof v !== "object") return JSON.stringify(v);
    var n = ind + "  ", items;
    if (Array.isArray(v)) {
      if (!v.length) return "[]";
      items = v.map(function (x) { return n + pyJson(x, n); });
      return "[\n" + items.join(",\n") + "\n" + ind + "]";
    }
    var keys = Object.keys(v).sort();
    if (!keys.length) return "{}";
    items = keys.map(function (k) { return n + JSON.stringify(k) + ": " + pyJson(v[k], n); });
    return "{\n" + items.join(",\n") + "\n" + ind + "}";
  }

  function buildFiles(spec, cfg) {
    var c = normalize(spec, cfg), b = spec.buses[c.bus], out = [];
    packageFiles(spec, c).forEach(function (n) { out.push(["rtl/" + n, n === b.top ? topSource(spec, c) : spec.sources[n]]); });
    out.push(["config.json", pyJson(c) + "\n"]);
    out.push(["README.md", readme(spec, c)]);
    return out;
  }

  // Fastest configuration that fits (same ordering as the Python suggest()).
  function suggest(spec, budget, bus, dsp, must, options) {
    var ids = spec.features.map(function (f) { return f.id; }), best = null, bestKey = null;
    must = must || [];
    spec.lanes.forEach(function (lanes) {
      [0, 1].forEach(function (rq) {
        for (var mask = 0; mask < (1 << ids.length); mask++) {
          var fs = ids.filter(function (_, i) { return (mask >> i) & 1; });
          if (must.some(function (m) { return fs.indexOf(m) < 0; })) continue;
          var cfg = { lanes: lanes, rq: rq, features: fs, bus: bus || "uart", dsp: dsp === undefined ? true : dsp, budget: budget, options: options || {} };
          var e = estimate(spec, cfg);
          if (verdict(e, budget) !== "fits") continue;
          var key = [lanes * (rq ? 1.5 : 1.0), fs.length, -e.le];
          if (bestKey === null || key[0] > bestKey[0] || (key[0] === bestKey[0] && (key[1] > bestKey[1] || (key[1] === bestKey[1] && key[2] > bestKey[2])))) { best = cfg; bestKey = key; }
        }
      });
    });
    return best;
  }

  var api = { normalize: normalize, estimate: estimate, verdict: verdict, warnings: warnings, buildFiles: buildFiles, suggest: suggest, FIT_LIMIT: FIT_LIMIT, RAM: RAM };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.TmfConfigurator = api;
})(typeof window !== "undefined" ? window : this);
