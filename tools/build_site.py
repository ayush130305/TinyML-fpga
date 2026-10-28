# build_site.py - build the static configurator page docs/index.html (GitHub Pages: branch main, folder /docs).
# The page embeds configurator/spec.json and configurator/configurator.js, so it needs no server; only JSZip
# (cdnjs) is loaded at run time, for the download.
#   python3 tools/build_site.py

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tinyml-fpga configurator</title>
<style>
:root { --bg:#fff; --fg:#111; --mute:#666; --line:#ddd; --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; }
@media (prefers-color-scheme: dark) { :root { --bg:#111; --fg:#eee; --mute:#999; --line:#333; --ok:#3fb950; --warn:#d29922; --bad:#f85149; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:640px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:20px; font-weight:600; margin:0 0 4px; }
p.sub { color:var(--mute); margin:0 0 28px; font-size:14px; }
h2 { font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:var(--mute); margin:28px 0 8px; }
.row { display:flex; flex-wrap:wrap; gap:8px; }
button, select, input[type=number] { font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:6px 12px; }
button { cursor:pointer; }
button.on { border-color:var(--fg); }
button.primary { border-color:var(--fg); background:var(--fg); color:var(--bg); }
label.f { display:flex; gap:10px; align-items:baseline; padding:6px 0; border-bottom:1px solid var(--line); }
label.f span.n { flex:1; }
label.f small { color:var(--mute); display:block; font-size:13px; }
label.f b { font-weight:400; font-family:ui-monospace, Menlo, Consolas, monospace; font-size:13px; color:var(--mute); white-space:nowrap; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin:10px 0 6px; }
.bar i { display:block; height:100%; background:var(--ok); width:0; }
.num { font:600 28px ui-monospace, Menlo, Consolas, monospace; }
.v { font-size:14px; }
.fits { color:var(--ok); } .tight { color:var(--warn); } .over { color:var(--bad); }
ul.w { margin:8px 0 0; padding-left:18px; color:var(--mute); font-size:14px; }
details { margin-top:12px; font-size:14px; } summary { cursor:pointer; color:var(--mute); }
.opt { display:grid; grid-template-columns:1fr 1fr; gap:8px 16px; margin-top:10px; }
.opt label { display:flex; flex-direction:column; font-size:13px; color:var(--mute); gap:2px; }
.foot { color:var(--mute); font-size:13px; margin-top:32px; }
</style>
</head>
<body>
<main>
<h1>tinyml-fpga configurator</h1>
<p class="sub">Pick a LUT budget and the features you want. The package contains only the RTL you need.</p>

<h2>Budget (logic elements)</h2>
<div class="row" id="budgets"></div>

<h2>Engine</h2>
<div class="row">
  <label>Lanes <select id="lanes"></select></label>
  <label>Requantiser <select id="rq"><option value="0">serial (small)</option><option value="1">pipelined (fast)</option></select></label>
  <label>Bus <select id="bus"></select></label>
  <label><input type="checkbox" id="dsp" checked> multiplier blocks available</label>
</div>

<h2>Features</h2>
<div id="features"></div>

<h2>Estimate</h2>
<div class="num" id="le">-</div>
<div class="bar"><i id="barfill"></i></div>
<div class="v" id="verdict"></div>
<div class="v" style="color:var(--mute)" id="detail"></div>
<ul class="w" id="warn"></ul>

<details>
  <summary>Memory sizes and bus settings</summary>
  <div class="opt" id="opts"></div>
</details>

<div class="row" style="margin-top:24px">
  <button class="primary" id="dl">Download package (.zip)</button>
  <button id="suggest">Fastest that fits</button>
</div>
<div class="v" id="msg" style="color:var(--mute);margin-top:8px"></div>

<p class="foot">Estimates are Yosys iCE40 counts scaled to Efinity Trion logic elements (one calibration design), model error on held-out configurations: mean __MEAN__%, worst __MAX__%. Run your own synthesis for the real number.</p>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<script>
var SPEC = __SPEC__;
</script>
<script>
__JS__
</script>
<script>
(function () {
  var C = window.TmfConfigurator, S = SPEC;
  var $ = function (id) { return document.getElementById(id); };
  var state = { budget: 8000, features: [] };
  var OPT = [["W_AW", "Weight RAM address bits"], ["X_AW", "Input RAM address bits"], ["Y_AW", "Output RAM address bits"], ["P_AW", "Parameter RAM address bits"], ["D_AW", "Descriptor RAM address bits"], ["clk_hz", "Clock (Hz, UART)"], ["baud", "Baud (UART)"], ["i2c_addr", "I2C address (decimal)"]];

  S.budgets.forEach(function (b) {
    var el = document.createElement("button");
    el.textContent = b >= 1000 ? (b / 1000) + "k" : b; el.dataset.v = b;
    el.onclick = function () { state.budget = b; render(); };
    $("budgets").appendChild(el);
  });
  var custom = document.createElement("input");
  custom.type = "number"; custom.placeholder = "custom"; custom.min = 500; custom.style.width = "110px";
  custom.onchange = function () { var v = parseInt(custom.value, 10); if (v > 0) { state.budget = v; render(); } };
  $("budgets").appendChild(custom);

  S.lanes.forEach(function (l) { var o = document.createElement("option"); o.value = l; o.textContent = l; $("lanes").appendChild(o); });
  Object.keys(S.buses).forEach(function (k) { var o = document.createElement("option"); o.value = k; o.textContent = S.buses[k].label; $("bus").appendChild(o); });
  S.features.forEach(function (f) {
    var l = document.createElement("label"); l.className = "f";
    l.innerHTML = '<input type="checkbox" data-id="' + f.id + '"><span class="n">' + f.label + '<small>' + f.help + '</small></span><b></b>';
    $("features").appendChild(l);
  });
  OPT.forEach(function (o) {
    var l = document.createElement("label"); l.textContent = o[1];
    var i = document.createElement("input"); i.type = "number"; i.id = "o_" + o[0]; l.appendChild(i);
    $("opts").appendChild(l);
  });
  var def = { W_AW: 12, X_AW: 10, Y_AW: 8, P_AW: 10, D_AW: 9, clk_hz: 100000000, baud: 921600, i2c_addr: 42 };
  OPT.forEach(function (o) { $("o_" + o[0]).value = def[o[0]]; });

  function cfg() {
    var o = {};
    OPT.forEach(function (x) { o[x[0]] = $("o_" + x[0]).value; });
    return { lanes: $("lanes").value, rq: $("rq").value, bus: $("bus").value, dsp: $("dsp").checked, budget: state.budget,
             features: Array.prototype.filter.call(document.querySelectorAll("#features input"), function (i) { return i.checked; }).map(function (i) { return i.dataset.id; }), options: o };
  }

  function render() {
    Array.prototype.forEach.call($("budgets").querySelectorAll("button"), function (b) { b.className = (+b.dataset.v === state.budget) ? "on" : ""; });
    var c;
    try { c = cfg(); C.normalize(S, c); } catch (e) { $("msg").textContent = e.message; return; }
    $("msg").textContent = "";
    var e = C.estimate(S, c), v = C.verdict(e, state.budget);
    $("le").textContent = "~" + e.le.toLocaleString("en-US") + " LE";
    $("barfill").style.width = Math.min(100, 100 * e.le / state.budget) + "%";
    $("barfill").style.background = "var(--" + (v === "fits" ? "ok" : v === "tight" ? "warn" : "bad") + ")";
    $("verdict").className = "v " + v;
    $("verdict").textContent = Math.round(100 * e.le / state.budget) + "% of " + state.budget.toLocaleString("en-US") + " - " + (v === "fits" ? "fits" : v === "tight" ? "tight (routing and the rest of your design need room)" : "over budget");
    $("detail").textContent = e.ff_est + " flip-flops, " + e.dsp + " multipliers, " + e.ram4k + " RAM blocks";
    $("warn").innerHTML = C.warnings(S, c).map(function (w) { return "<li>" + w + "</li>"; }).join("");
    // Per-feature cost at the current lanes.
    Array.prototype.forEach.call(document.querySelectorAll("#features label"), function (l) {
      var id = l.querySelector("input").dataset.id, d = S.delta[c.lanes][id];
      l.querySelector("b").textContent = "+" + Math.round((c.dsp ? d.lut4 : d.lut4_nodsp) * S.le_per_lut4) + " LE";
    });
  }

  document.addEventListener("change", render);
  document.addEventListener("input", function (ev) { if (ev.target.type === "number") render(); });

  $("suggest").onclick = function () {
    var c = cfg(), must = c.features;
    var s = C.suggest(S, state.budget, c.bus, c.dsp, must, c.options);
    if (!s) { $("msg").textContent = "Nothing fits with those features checked."; return; }
    $("lanes").value = s.lanes; $("rq").value = s.rq;
    Array.prototype.forEach.call(document.querySelectorAll("#features input"), function (i) { i.checked = s.features.indexOf(i.dataset.id) >= 0; });
    render();
  };

  $("dl").onclick = function () {
    var c = cfg();
    if (typeof JSZip === "undefined") { $("msg").textContent = "JSZip did not load (offline?). Use python -m tinyml_fpga configure instead."; return; }
    var z = new JSZip(), files = C.buildFiles(S, c);
    files.forEach(function (f) { z.file(f[0], f[1], { date: new Date(Date.UTC(2026, 0, 1)) }); });
    z.generateAsync({ type: "blob" }).then(function (blob) {
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "tinyml-fpga-" + c.lanes + "L-" + c.bus + ".zip";
      document.body.appendChild(a); a.click(); a.remove();
    });
  };
  render();
})();
</script>
</body>
</html>
'''


def main():
    spec = json.load(open(os.path.join(ROOT, "configurator", "spec.json")))
    js = open(os.path.join(ROOT, "configurator", "configurator.js")).read()
    page = (PAGE.replace("__SPEC__", json.dumps(spec, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__JS__", js).replace("__MEAN__", str(spec["validation"]["mean_abs_error_pct"]))
            .replace("__MAX__", str(spec["validation"]["max_abs_error_pct"])))
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "index.html"), "w") as f:
        f.write(page)
    open(os.path.join(ROOT, "docs", ".nojekyll"), "w").close()
    print("wrote docs/index.html (%d KB)" % (len(page) // 1024))


if __name__ == "__main__":
    main()
