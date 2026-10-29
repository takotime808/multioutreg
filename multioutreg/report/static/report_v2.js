/* Copyright (c) 2026 takotime808
   Charts and controls for the v2 report. Plain ES2017, no dependencies; it is
   inlined into the report, reads the #report-data JSON and draws inline SVG
   into every [data-chart] element. Everything else on the page works without it. */
(function () {
  "use strict";
  var el = document.getElementById("report-data");
  if (!el) return;
  var D;
  try { D = JSON.parse(el.textContent); } catch (e) { return; }
  var OUT = D.outputs || [], TH = D.thresholds || {}, N = (D.run || {}).n_test_points || 1;
  var NS = "http://www.w3.org/2000/svg";

  /* ---------- helpers ---------- */
  function E(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function lin(d0, d1, r0, r1) { var s = (d1 - d0) || 1; return function (v) { return r0 + (v - d0) / s * (r1 - r0); }; }
  function ticks(a, b, n) {
    n = n || 5; if (!(b > a)) return [a];
    var st0 = (b - a) / n, p = Math.pow(10, Math.floor(Math.log10(st0))), m = st0 / p;
    var st = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p, t = [];
    for (var v = Math.ceil(a / st - 1e-9) * st; v <= b + st * 1e-6; v += st) t.push(Math.abs(v) < st * 1e-6 ? 0 : +v.toPrecision(10));
    return t;
  }
  function fmt(v) {
    if (v == null || !isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e5 || a < 1e-3) return v.toExponential(1);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 0.1) return v.toFixed(2);
    if (a >= 0.01) return v.toFixed(3);
    return v.toFixed(4);
  }
  function pct(v, d) { return (100 * v).toFixed(d || 0) + "%"; }
  function sgnPct(v) { return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(0) + "%"; }
  function pad(lo, hi, f) { f = f == null ? 0.06 : f; var r = (hi - lo) || Math.abs(hi) || 1; return [lo - r * f, hi + r * f]; }
  function mn(a) { var m = Infinity; for (var i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }
  function mx(a) { var m = -Infinity; for (var i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
  function color(o) { return "var(--s" + ((o.idx % 8) + 1) + ")"; }
  function band(p) { return (TH.z || 1.96) * Math.sqrt(p * (1 - p) / N); }
  function trunc(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
  function barPath(x, y, w, h, r) {
    if (h <= 0 || w <= 0) return "";
    r = Math.min(r || 3, w / 2, h);
    return "M" + x + "," + (y + h) + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y + "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) + "V" + (y + h) + "Z";
  }

  function frame(host, o) {
    var W = o.w || 440, H = o.h || 300, m = Object.assign({ t: 14, r: 16, b: 42, l: 56 }, o.m || {});
    var svg = E("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": o.label || "" }, null);
    host.appendChild(svg);
    var X = lin(o.x[0], o.x[1], m.l, W - m.r), Y = lin(o.y[0], o.y[1], H - m.b, m.t);
    var g = E("g", {}, svg);
    var yt = o.yticks || ticks(o.y[0], o.y[1], o.ny || 5), xt = o.xticks || ticks(o.x[0], o.x[1], o.nx || 5);
    var yf = o.yfmt || fmt, xf = o.xfmt || fmt;
    if (!o.noYgrid) yt.forEach(function (v) {
      E("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 }, g);
      E("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 11, fill: "var(--muted)" }, g, yf(v));
    });
    if (!o.noX) {
      xt.forEach(function (v) { E("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle", "font-size": 11, fill: "var(--muted)" }, g, xf(v)); });
      E("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1 }, g);
    }
    if (o.xlab) E("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", "font-size": 11.5, fill: "var(--fg2)" }, g, o.xlab);
    if (o.ylab) E("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "font-size": 11.5, fill: "var(--fg2)", transform: "rotate(-90 12 " + (m.t + H - m.b) / 2 + ")" }, g, o.ylab);
    return { svg: svg, g: E("g", {}, svg), X: X, Y: Y, W: W, H: H, m: m };
  }
  function dot(g, x, y, r, fill, tip) {
    return E("circle", { cx: x, cy: y, r: r, fill: fill, stroke: "var(--surface)", "stroke-width": 1.2, "data-tip": tip || null }, g);
  }
  function note(host, text) { var p = document.createElement("div"); p.className = "look"; p.textContent = text; host.appendChild(p); }

  /* intervals for one output: Gaussian 95% if there is a std, else conformal */
  function intervals(o) {
    var P = o.points, z = 1.959964;
    if (P.std) return { lo: P.pred.map(function (p, i) { return p - z * P.std[i]; }), hi: P.pred.map(function (p, i) { return p + z * P.std[i]; }), label: "95% Gaussian interval" };
    if (P.lower && P.upper) return { lo: P.lower, hi: P.upper, label: "Conformal interval" };
    return null;
  }

  /* ---------- per-output charts ---------- */
  function parity(host, o, mini) {
    var P = o.points, iv = intervals(o);
    var all = P.y.concat(P.pred);
    var d = pad(mn(all), mx(all));
    var f = frame(host, { x: d, y: d, w: mini ? 460 : 440, h: mini ? 260 : 320, xlab: "Observed " + o.name, ylab: "Predicted " + o.name, label: "Predicted vs observed " + o.name });
    E("line", { x1: f.X(d[0]), y1: f.Y(d[0]), x2: f.X(d[1]), y2: f.Y(d[1]), stroke: "var(--axis)", "stroke-width": 1.5, "stroke-dasharray": "4 4" }, f.g);
    if (iv) P.y.forEach(function (v, i) {
      E("line", { x1: f.X(v), x2: f.X(v), y1: f.Y(Math.max(d[0], iv.lo[i])), y2: f.Y(Math.min(d[1], iv.hi[i])), stroke: color(o), "stroke-width": 2, "stroke-opacity": 0.25, "stroke-linecap": "round" }, f.g);
    });
    var r = P.y.length > 400 ? 2.5 : 4;
    P.y.forEach(function (v, i) {
      dot(f.g, f.X(v), f.Y(P.pred[i]), r, color(o), "Observed " + fmt(v) + "\nPredicted " + fmt(P.pred[i]) + (iv ? "\nInterval " + fmt(iv.lo[i]) + " to " + fmt(iv.hi[i]) : ""));
    });
    E("text", { x: f.m.l + 8, y: f.m.t + 14, "font-size": 12, fill: "var(--fg)", "font-weight": 600 }, f.g, "R² " + (o.r2 == null ? "—" : o.r2.toFixed(3)));
  }
  function residuals(host, o, mini) {
    var P = o.points, xs = P.pred, ys = P.y.map(function (v, i) { return v - P.pred[i]; });
    var ym = (mx(ys.map(Math.abs)) || 1) * 1.15;
    var f = frame(host, { x: pad(mn(xs), mx(xs)), y: [-ym, ym], w: mini ? 460 : 440, h: mini ? 240 : 300, xlab: "Predicted " + o.name, ylab: "Observed − predicted", label: "Residuals " + o.name });
    E("line", { x1: f.m.l, x2: f.W - f.m.r, y1: f.Y(0), y2: f.Y(0), stroke: "var(--axis)", "stroke-width": 1.5 }, f.g);
    var r = xs.length > 400 ? 2.5 : 4;
    xs.forEach(function (x, i) { dot(f.g, f.X(x), f.Y(ys[i]), r, color(o), "Predicted " + fmt(x) + "\nResidual " + fmt(ys[i])); });
    var t = o.trend;
    if (t) {
      var x0 = mn(xs), x1 = mx(xs), sig = t.significant;
      E("line", { x1: f.X(x0), x2: f.X(x1), y1: f.Y(t.intercept + t.slope * x0), y2: f.Y(t.intercept + t.slope * x1), stroke: sig ? "var(--crit)" : "var(--fg2)", "stroke-width": 2, "stroke-dasharray": sig ? null : "5 4" }, f.g);
      E("text", { x: f.W - f.m.r - 4, y: f.m.t + 14, "text-anchor": "end", "font-size": 11.5, fill: "var(--fg)", "font-weight": 600 }, f.g,
        sig ? "Trend: slope " + t.slope.toFixed(2) + " (p = " + t.p.toPrecision(2) + ")" : "No significant trend (p = " + t.p.toPrecision(2) + ")");
    }
  }
  function histogram(host, o) {
    var P = o.points, res = P.y.map(function (v, i) { return v - P.pred[i]; });
    var ym = (mx(res.map(Math.abs)) || 1) * 1.05, nb = Math.max(8, Math.min(30, Math.round(Math.sqrt(res.length) * 1.5))), w = 2 * ym / nb;
    var counts = []; for (var i = 0; i < nb; i++) counts.push(0);
    res.forEach(function (r) { counts[Math.min(nb - 1, Math.max(0, Math.floor((r + ym) / w)))]++; });
    var f = frame(host, { x: [-ym, ym], y: [0, mx(counts) + 1], xlab: "Residual " + o.name, ylab: "Test points", ny: 4, label: "Residual histogram " + o.name });
    counts.forEach(function (c, i) {
      var x0 = f.X(-ym + i * w) + 1, x1 = f.X(-ym + (i + 1) * w) - 1;
      E("path", { d: barPath(x0, f.Y(c), x1 - x0, f.Y(0) - f.Y(c)), fill: color(o), "data-tip": c + " points between " + fmt(-ym + i * w) + " and " + fmt(-ym + (i + 1) * w) }, f.g);
    });
    E("line", { x1: f.X(0), x2: f.X(0), y1: f.m.t, y2: f.H - f.m.b, stroke: "var(--fg2)", "stroke-width": 1, "stroke-dasharray": "3 3" }, f.g);
  }
  function ordered(host, o) {
    var P = o.points, iv = intervals(o);
    if (!iv) { note(host, "No intervals to show: the model gives no std and conformal intervals were not requested."); return; }
    var idx = P.y.map(function (_, i) { return i; }).sort(function (a, b) { return P.y[a] - P.y[b]; });
    var lo = mn(idx.map(function (i) { return Math.min(P.y[i], iv.lo[i]); })), hi = mx(idx.map(function (i) { return Math.max(P.y[i], iv.hi[i]); }));
    if (!isFinite(lo) || !isFinite(hi)) { note(host, "Intervals are unbounded (too few calibration points)."); return; }
    var f = frame(host, { x: [-1, idx.length], y: pad(lo, hi), xlab: "Test points, sorted by observed value", ylab: o.name, xticks: [], label: "Ordered intervals " + o.name });
    var out = 0, sw = Math.max(1, Math.min(3, 300 / idx.length));
    idx.forEach(function (i, k) {
      var inside = P.y[i] >= iv.lo[i] && P.y[i] <= iv.hi[i]; if (!inside) out++;
      E("line", { x1: f.X(k), x2: f.X(k), y1: f.Y(iv.lo[i]), y2: f.Y(iv.hi[i]), stroke: color(o), "stroke-width": sw, "stroke-opacity": 0.35, "stroke-linecap": "round" }, f.g);
      dot(f.g, f.X(k), f.Y(P.y[i]), inside ? Math.min(3.5, sw + 1) : Math.min(4.5, sw + 2), inside ? "var(--fg2)" : "var(--crit)",
        "Observed " + fmt(P.y[i]) + "\nInterval " + fmt(iv.lo[i]) + " to " + fmt(iv.hi[i]) + (inside ? "" : "\nOutside interval"));
    });
    E("text", { x: f.m.l + 8, y: f.m.t + 14, "font-size": 11.5, fill: "var(--fg)", "font-weight": 600 }, f.g, out + " of " + idx.length + " outside" + (P.std ? " (expected ≈ " + Math.round(0.05 * idx.length) + ")" : ""));
    var lg = document.createElement("div"); lg.className = "legend";
    lg.innerHTML = '<span><i class="lg-line" style="border-color:' + color(o) + ';opacity:.5;border-top-width:3px"></i>' + iv.label + '</span><span><i class="lg-dot" style="background:var(--fg2)"></i>Observed, inside</span><span><i class="lg-dot" style="background:var(--crit)"></i>Observed, outside</span>';
    host.appendChild(lg);
  }
  function calib(host, o, mini) {
    var f = frame(host, { x: [0, 1], y: [0, 1], w: mini ? 460 : 320, h: mini ? 240 : 300, m: { l: 46, b: 42 }, xlab: "Nominal coverage", ylab: "Observed coverage", xfmt: function (v) { return pct(v); }, yfmt: function (v) { return pct(v); }, nx: 4, ny: 4, label: "Calibration " + o.name });
    var ps = []; for (var i = 0; i <= 40; i++) ps.push(i / 40);
    var up = ps.map(function (p) { return f.X(p) + "," + f.Y(Math.min(1, p + band(p))); });
    var dn = ps.map(function (p) { return f.X(p) + "," + f.Y(Math.max(0, p - band(p))); }).reverse();
    E("path", { d: "M" + up.concat(dn).join("L") + "Z", fill: "var(--band)" }, f.g);
    E("line", { x1: f.X(0), y1: f.Y(0), x2: f.X(1), y2: f.Y(1), stroke: "var(--axis)", "stroke-width": 1.5 }, f.g);
    function path(arr) { return "M" + o.levels.map(function (p, i) { return f.X(p) + "," + f.Y(arr[i]); }).join("L"); }
    var i95 = o.levels.indexOf(0.95);
    if (o.gauss_curve) {
      E("path", { d: path(o.gauss_curve), fill: "none", stroke: "var(--fg2)", "stroke-width": 2, "stroke-dasharray": "5 4" }, f.g);
      dot(f.g, f.X(0.95), f.Y(o.gauss_curve[i95]), 4.5, "var(--fg2)", "Gaussian 95%: " + pct(o.gauss_curve[i95]));
    }
    if (o.conf_curve) {
      E("path", { d: path(o.conf_curve), fill: "none", stroke: color(o), "stroke-width": 2 }, f.g);
      dot(f.g, f.X(0.95), f.Y(o.conf_curve[i95]), 4.5, color(o), "Conformal 95%: " + pct(o.conf_curve[i95]));
    } else if (o.conformal) {
      dot(f.g, f.X(o.conformal.nominal), f.Y(o.conformal.coverage), 5, color(o), "Conformal " + pct(o.conformal.nominal) + ": " + pct(o.conformal.coverage));
    }
    o.levels.forEach(function (p, i) {
      var t = o.name + " at " + pct(p) + " nominal" + (o.gauss_curve ? "\nGaussian " + pct(o.gauss_curve[i]) : "") + (o.conf_curve ? "\nConformal " + pct(o.conf_curve[i]) : "");
      E("circle", { cx: f.X(p), cy: f.Y(o.gauss_curve ? o.gauss_curve[i] : o.conf_curve ? o.conf_curve[i] : p), r: 9, fill: "transparent", "data-tip": t }, f.g);
    });
    E("text", { x: f.m.l + 8, y: f.m.t + 14, "font-size": 13, "font-weight": 600, fill: "var(--fg)" }, f.g, o.name);
    if (!o.gauss_curve && !o.conformal) note(host, "No intervals to check for " + o.name + ".");
  }

  /* ---------- cross-output charts ---------- */
  function cov95(host, mini) {
    var rows = OUT.filter(function (o) { return o.has_std || (o.conformal && o.conformal.nominal === 0.95); });
    if (!rows.length) { note(host, "No 95% intervals to compare."); return; }
    var vals = []; rows.forEach(function (o) { if (o.has_std) vals.push(o.gauss95); if (o.conf_curve) vals.push(o.conformal.coverage); });
    var b = band(0.95), x0 = Math.max(0, Math.min(0.6, mn(vals) - 0.05, 0.95 - 2.5 * b));
    var rh = 40, l = 12 + 7 * Math.min(18, mx(rows.map(function (o) { return o.name.length; })));
    var f = frame(host, { x: [x0, 1], y: [0, rows.length], w: mini ? 460 : 440, h: 40 + rh * rows.length, m: { l: l, t: 6, b: 34, r: 16 }, noYgrid: true, xfmt: function (v) { return pct(v); }, xlab: "Observed coverage of the 95% interval", nx: 4, label: "Coverage at 95%" });
    E("rect", { x: f.X(Math.max(x0, 0.95 - b)), y: f.m.t, width: f.X(1) - f.X(Math.max(x0, 0.95 - b)), height: f.H - f.m.b - f.m.t, fill: "var(--band)" }, f.g);
    E("line", { x1: f.X(0.95), x2: f.X(0.95), y1: f.m.t, y2: f.H - f.m.b, stroke: "var(--fg2)", "stroke-dasharray": "3 3" }, f.g);
    rows.forEach(function (o, i) {
      var cy = f.m.t + rh * (i + 0.5);
      E("text", { x: f.m.l - 10, y: cy + 4, "text-anchor": "end", "font-size": 12, "font-weight": 600, fill: "var(--fg)" }, f.g, trunc(o.name, 18));
      var g = o.has_std ? Math.max(x0, o.gauss95) : null, c = o.conf_curve ? o.conformal.coverage : null;
      if (g != null && c != null) E("line", { x1: f.X(g), x2: f.X(c), y1: cy, y2: cy, stroke: "var(--axis)", "stroke-width": 2 }, f.g);
      if (g != null) E("rect", { x: f.X(g) - 5, y: cy - 5, width: 10, height: 10, rx: 2, fill: "var(--surface)", stroke: "var(--fg2)", "stroke-width": 2, "data-tip": o.name + " Gaussian: " + pct(o.gauss95) }, f.g);
      if (c != null) dot(f.g, f.X(c), cy, 5.5, color(o), o.name + " conformal: " + pct(c));
    });
    if (!mini) {
      var lg = document.createElement("div"); lg.className = "legend";
      lg.innerHTML = '<span><i class="lg-dot" style="background:transparent;border:2px solid var(--fg2);border-radius:2px"></i>Gaussian</span><span><i class="lg-dot" style="background:var(--fg2)"></i>Conformal (output color)</span><span><i class="lg-band"></i>expected from noise</span>';
      host.appendChild(lg);
    }
  }
  function width(host) {
    var rows = OUT.filter(function (o) { return o.width_gauss_pct != null || o.width_conf_pct != null; });
    if (!rows.length) { note(host, "No interval widths to compare."); return; }
    var wmax = mx(rows.map(function (o) { return Math.max(o.width_gauss_pct || 0, o.width_conf_pct || 0); })) * 1.18 || 1;
    var rh = 40, l = 12 + 7 * Math.min(18, mx(rows.map(function (o) { return o.name.length; })));
    var g = frame(host, { x: [0, wmax], y: [0, rows.length], w: 440, h: 40 + rh * rows.length, m: { l: l, t: 6, b: 34, r: 30 }, noYgrid: true, xfmt: function (v) { return v.toFixed(0) + "%"; }, xlab: "Mean interval width, % of test range", nx: 4, label: "Interval width" });
    rows.forEach(function (o, i) {
      var y0 = g.m.t + rh * i + 7;
      E("text", { x: g.m.l - 10, y: y0 + 17, "text-anchor": "end", "font-size": 12, "font-weight": 600, fill: "var(--fg)" }, g.g, trunc(o.name, 18));
      if (o.width_gauss_pct != null) {
        E("rect", { x: g.X(0), y: y0, width: Math.max(0, g.X(o.width_gauss_pct) - g.X(0)), height: 11, rx: 2, fill: "var(--axis)", "data-tip": o.name + " Gaussian 95% width " + o.width_gauss_pct.toFixed(1) + "% of range" }, g.g);
        E("text", { x: g.X(o.width_gauss_pct) + 5, y: y0 + 9, "font-size": 10.5, fill: "var(--fg2)" }, g.g, o.width_gauss_pct.toFixed(0) + "%");
      }
      if (o.width_conf_pct != null) {
        E("rect", { x: g.X(0), y: y0 + 13, width: Math.max(0, g.X(o.width_conf_pct) - g.X(0)), height: 11, rx: 2, fill: color(o), "data-tip": o.name + " conformal width " + o.width_conf_pct.toFixed(1) + "% of range" }, g.g);
        E("text", { x: g.X(o.width_conf_pct) + 5, y: y0 + 22, "font-size": 10.5, fill: "var(--fg2)" }, g.g, o.width_conf_pct.toFixed(0) + "%");
      }
    });
    var lg = document.createElement("div"); lg.className = "legend";
    lg.innerHTML = '<span><i class="lg-dot" style="background:var(--axis);border-radius:2px"></i>Gaussian 95%</span><span><i class="lg-dot" style="background:var(--fg2);border-radius:2px"></i>Conformal (output color)</span>';
    host.appendChild(lg);
  }
  function dumbbell(host, mini) {
    var rows = OUT.filter(function (o) { return o.mf_change_pct != null; });
    if (!rows.length) return;
    var vals = rows.map(function (o) { return 1 + o.mf_change_pct / 100; });
    var x = [Math.min(0.5, mn(vals) - 0.05), Math.max(1.15, mx(vals) + 0.08)];
    var rh = mini ? 32 : 44, l = 14 + 7 * Math.min(16, mx(rows.map(function (o) { return o.name.length; })));
    var gain = 1 + (TH.mf_gain || -5) / 100, hurt = 1 + (TH.mf_hurt || 5) / 100;
    var f = frame(host, { x: x, y: [0, rows.length], w: mini ? 460 : 860, h: 40 + rh * rows.length, m: { l: l, t: 8, b: 34, r: 34 }, noYgrid: true, xfmt: function (v) { return pct(v); }, xlab: "Test RMSE, % of HF-only model", nx: 6, label: "Multi-fidelity vs HF-only RMSE" });
    E("rect", { x: f.X(gain), y: f.m.t, width: f.X(hurt) - f.X(gain), height: f.H - f.m.b - f.m.t, fill: "var(--band)" }, f.g);
    E("line", { x1: f.X(1), x2: f.X(1), y1: f.m.t, y2: f.H - f.m.b, stroke: "var(--axis)", "stroke-width": 1 }, f.g);
    rows.forEach(function (o, i) {
      var cy = f.m.t + rh * (i + 0.5), v = 1 + o.mf_change_pct / 100;
      E("text", { x: f.m.l - 12, y: cy + 4, "text-anchor": "end", "font-size": 12.5, "font-weight": 600, fill: "var(--fg)" }, f.g, trunc(o.name, 16));
      E("line", { x1: f.X(1), x2: f.X(v), y1: cy, y2: cy, stroke: color(o), "stroke-width": 3, "stroke-linecap": "round", "stroke-opacity": 0.6 }, f.g);
      dot(f.g, f.X(1), cy, 5, "var(--muted)", o.name + " HF-only RMSE " + fmt(o.hf_rmse));
      dot(f.g, f.X(v), cy, 6, color(o), o.name + " MF RMSE " + fmt(o.rmse) + " (" + sgnPct(o.mf_change_pct) + " vs HF-only)");
      var right = v > 1;
      E("text", { x: f.X(v) + (right ? 12 : -12), y: cy + 4, "text-anchor": right ? "start" : "end", "font-size": 12, "font-weight": 600, fill: "var(--fg)" }, f.g, sgnPct(o.mf_change_pct));
    });
  }
  var SHAPES = ["circle", "square", "diamond", "triangle"];
  function marker(g, shape, cx, cy, r, fill, tip) {
    var a = { fill: fill, stroke: "var(--surface)", "stroke-width": 1.2, "data-tip": tip };
    if (shape === "square") return E("rect", Object.assign(a, { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, rx: 2 }), g);
    if (shape === "diamond") return E("path", Object.assign(a, { d: "M" + cx + "," + (cy - r * 1.3) + "L" + (cx + r * 1.3) + "," + cy + "L" + cx + "," + (cy + r * 1.3) + "L" + (cx - r * 1.3) + "," + cy + "Z" }), g);
    if (shape === "triangle") return E("path", Object.assign(a, { d: "M" + cx + "," + (cy - r * 1.3) + "L" + (cx + r * 1.2) + "," + (cy + r) + "L" + (cx - r * 1.2) + "," + (cy + r) + "Z" }), g);
    return dot(g, cx, cy, r, fill, tip);
  }
  function quadrant(host, mini) {
    var lower = (D.levels || []).filter(function (l) { return !l.is_highest; });
    if (!lower.length) { note(host, "Only one fidelity level: nothing to compare."); return; }
    var pts = [];
    lower.forEach(function (lv, li) {
      OUT.forEach(function (o) {
        var v = lv.outputs[o.name];
        if (v && v.r != null && v.r2 != null) pts.push({ lv: lv.level, li: li, o: o, r: v.r, r2: v.r2 });
      });
    });
    var ylo = Math.max(-1.5, Math.min(-0.5, mn(pts.map(function (p) { return p.r2; })) - 0.1));
    var xlo = Math.min(0, mn(pts.map(function (p) { return p.r; })) - 0.05);
    var wr = TH.weak_level_r || 0.5;
    var f = frame(host, { x: [xlo, 1], y: [ylo, 1], w: mini ? 460 : 440, h: mini ? 260 : 320, xlab: "Correlation r with HF truth", ylab: "R² against HF truth", nx: 5, ny: 5, label: "Fidelity level informativeness" });
    E("rect", { x: f.X(xlo), y: f.Y(1), width: f.X(wr) - f.X(xlo), height: f.Y(ylo) - f.Y(1), fill: "var(--band)" }, f.g);
    E("line", { x1: f.X(wr), x2: f.X(wr), y1: f.Y(1), y2: f.Y(ylo), stroke: "var(--axis)", "stroke-dasharray": "4 4" }, f.g);
    E("line", { x1: f.X(wr), x2: f.X(1), y1: f.Y(TH.r2_warn || 0.8), y2: f.Y(TH.r2_warn || 0.8), stroke: "var(--axis)", "stroke-dasharray": "4 4" }, f.g);
    if (!mini) [["Little usable information", (xlo + wr) / 2, ylo + 0.12 * (1 - ylo)], ["Informative, unbiased", (wr + 1) / 2, 0.93], ["Informative but biased:", (wr + 1) / 2, ylo + 0.42 * (1 - ylo)], ["MF can correct", (wr + 1) / 2, ylo + 0.42 * (1 - ylo) - 0.09]].forEach(function (t) {
      E("text", { x: f.X(t[1]), y: f.Y(t[2]), "text-anchor": "middle", "font-size": 10.5, fill: "var(--muted)", "font-weight": 600 }, f.g, t[0]);
    });
    pts.forEach(function (p) {
      var cy = f.Y(Math.max(ylo, p.r2));
      marker(f.g, SHAPES[p.li % SHAPES.length], f.X(p.r), cy, 5, color(p.o), "Level " + p.lv + " for " + p.o.name + "\nr = " + p.r.toFixed(2) + ", R² = " + p.r2.toFixed(2) + (p.r2 < ylo ? " (below axis)" : ""));
    });
    var lg = host.parentNode && host.parentNode.querySelector('[data-legend="quadrant"]');
    if (lg && !mini) {
      lg.innerHTML = lower.map(function (lv, li) { var s = SHAPES[li % SHAPES.length]; return '<span><i class="lg-dot" style="background:var(--fg2);border-radius:' + (s === "circle" ? "50%" : "2px") + (s === "diamond" ? ";transform:rotate(45deg)" : "") + '"></i>level ' + lv.level + "</span>"; }).join("") +
        OUT.map(function (o) { return '<span><i class="swatch" style="background:' + color(o) + '"></i>' + o.name + "</span>"; }).join("");
    }
  }
  function pdp(host) {
    var P = D.pdp; if (!P) return;
    var feats = P.features, grid = document.createElement("div");
    grid.className = feats.length >= 3 ? "grid3" : feats.length === 2 ? "grid2" : "";
    host.appendChild(grid);
    var share = D.shap ? D.shap.shares : null;
    OUT.forEach(function (o, ri) {
      var C = P.curves[o.name]; if (!C) return;
      var all = []; feats.forEach(function (ft) { all = all.concat(C[ft].p10, C[ft].p90); });
      var yd = pad(mn(all), mx(all), 0.05);
      feats.forEach(function (ft, ci) {
        var d = document.createElement("div"); d.className = "fig chart"; grid.appendChild(d);
        var xs = P.grid[ft], c = C[ft], last = ri === OUT.length - 1;
        var f = frame(d, { x: [xs[0], xs[xs.length - 1]], y: yd, w: 300, h: 200, m: { l: ci === 0 ? 56 : 44, b: last ? 40 : 24, t: 22, r: 10 }, ny: 4, nx: 4, xlab: last ? ft : "", ylab: ci === 0 ? trunc(o.name, 18) : "", label: "Partial dependence of " + o.name + " on " + ft });
        var up = xs.map(function (x, i) { return f.X(x) + "," + f.Y(c.p90[i]); }), dn = xs.map(function (x, i) { return f.X(x) + "," + f.Y(c.p10[i]); }).reverse();
        E("path", { d: "M" + up.concat(dn).join("L") + "Z", fill: color(o), "fill-opacity": 0.14 }, f.g);
        E("path", { d: "M" + xs.map(function (x, i) { return f.X(x) + "," + f.Y(c.mean[i]); }).join("L"), fill: "none", stroke: color(o), "stroke-width": 2 }, f.g);
        xs.forEach(function (x, i) { if (i % 2 === 0) E("circle", { cx: f.X(x), cy: f.Y(c.mean[i]), r: 7, fill: "transparent", "data-tip": o.name + " at " + ft + " = " + fmt(x) + ": " + fmt(c.mean[i]) + "\n10th–90th: " + fmt(c.p10[i]) + " to " + fmt(c.p90[i]) }, f.g); });
        var s = share && share[o.name] && share[o.name][ft] != null ? pct(share[o.name][ft]) + " of " + trunc(o.name, 14) : trunc(o.name, 20);
        E("text", { x: f.m.l, y: 13, "font-size": 11, "font-weight": ri === 0 ? 600 : 400, fill: ri === 0 ? "var(--fg)" : "var(--fg2)" }, f.g, ri === 0 ? trunc(ft, 22) + " · " + s : s);
      });
    });
  }
  function cvPlot(host, mini) {
    var cv = D.cv; if (!cv || !cv.rows) return;
    var ok = cv.rows.filter(function (r) { return r.mean != null; });
    if (!ok.length) { note(host, "Every candidate failed to fit."); return; }
    var best = ok.filter(function (r) { return r.name === cv.best; })[0] || ok[0];
    var rows = mini ? ok.slice(0, 5) : cv.rows.slice(0, 25);
    var lim = best.mean + (best.sd || 0), tied = cv.tied || [];
    var hiv = mx(ok.map(function (r) { return r.mean + (r.sd || 0); })) * 1.05, lov = Math.max(0, mn(ok.map(function (r) { return r.mean - (r.sd || 0); })) * 0.9);
    var rh = mini ? 26 : 32, nameMax = mini ? 22 : 34;
    var l = 16 + 6.2 * Math.min(nameMax, mx(rows.map(function (r) { return r.name.length; })));
    var f = frame(host, { x: [lov, hiv], y: [0, rows.length], w: mini ? 460 : 760, h: 40 + rh * rows.length, m: { l: l, t: 6, b: 34, r: mini ? 16 : 90 }, noYgrid: true, xlab: cv.metric + " (lower is better)", nx: 5, label: "Cross-validation scores" });
    E("rect", { x: f.X(lov), y: f.m.t, width: Math.max(0, f.X(lim) - f.X(lov)), height: f.H - f.m.b - f.m.t, fill: "var(--band)" }, f.g);
    rows.forEach(function (c, i) {
      var cy = f.m.t + rh * (i + 0.5), isBest = c === best;
      E("text", { x: f.m.l - 12, y: cy + 4, "text-anchor": "end", "font-size": mini ? 11 : 12, "font-weight": isBest ? 700 : 400, fill: "var(--fg)" }, f.g, trunc(c.name, nameMax));
      if (c.mean == null) { E("text", { x: f.m.l + 4, y: cy + 4, "font-size": 11.5, fill: "var(--muted)" }, f.g, trunc("Failed: " + c.status, 70)); return; }
      var sd = c.sd || 0;
      E("line", { x1: f.X(Math.max(lov, c.mean - sd)), x2: f.X(c.mean + sd), y1: cy, y2: cy, stroke: isBest ? "var(--accent)" : "var(--fg2)", "stroke-width": 2, "stroke-linecap": "round" }, f.g);
      dot(f.g, f.X(c.mean), cy, isBest ? 6 : 5, isBest ? "var(--accent)" : "var(--fg2)", c.name + "\n" + cv.metric + " " + fmt(c.mean) + (c.sd != null ? " ± " + fmt(c.sd) : "") + (c.selected_for && c.selected_for.length ? "\nSelected for " + c.selected_for.join(", ") : ""));
      if (!mini) E("text", { x: f.W - f.m.r + 12, y: cy + 4, "font-size": 11.5, "font-weight": 600, fill: isBest ? "var(--accent)" : "var(--fg)" }, f.g, isBest ? (cv.per_output ? "Best" : "Selected") : tied.indexOf(c.name) >= 0 ? "Tied" : "");
    });
    if (!mini && cv.rows.length > rows.length) note(host, (cv.rows.length - rows.length) + " more candidates are in the table below.");
  }

  /* ---------- draw every slot ---------- */
  var DRAW = {
    parity: function (h, o, m) { parity(h, o, m); }, resid: function (h, o, m) { residuals(h, o, m); },
    hist: function (h, o) { histogram(h, o); }, ordered: function (h, o) { ordered(h, o); },
    calib: function (h, o, m) { calib(h, o, m); }, cov95: function (h, o, m) { cov95(h, m); },
    width: function (h) { width(h); }, dumbbell: function (h, o, m) { dumbbell(h, m); },
    quadrant: function (h, o, m) { quadrant(h, m); }, pdp: function (h) { pdp(h); }, cv: function (h, o, m) { cvPlot(h, m); }
  };
  document.querySelectorAll("[data-chart]").forEach(function (h) {
    var spec = h.getAttribute("data-chart").split(":"), fn = DRAW[spec[0]];
    if (!fn) return;
    var o = spec.length > 1 ? OUT[+spec[1]] : null;
    if (spec.length > 1 && !o) return;
    try { fn(h, o, h.getAttribute("data-mini") === "1"); }
    catch (e) { h.innerHTML = ""; note(h, "Chart could not be drawn: " + e.message); }
  });

  /* ---------- output switcher ---------- */
  var tabs = document.querySelector(".outtabs"), blocks = document.querySelector(".outblocks");
  if (tabs && blocks) tabs.addEventListener("click", function (e) {
    var b = e.target.closest("button[data-out]"); if (!b) return;
    var k = b.getAttribute("data-out");
    tabs.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
    blocks.classList.toggle("single", k !== "all");
    blocks.querySelectorAll(".outblock").forEach(function (x) { x.classList.toggle("on", x.getAttribute("data-out") === k); });
  });
  /* links to a hidden output block switch to it */
  document.addEventListener("click", function (e) {
    var a = e.target.closest('a[href^="#out-"]'); if (!a || !tabs) return;
    var k = a.getAttribute("href").slice(5), b = tabs.querySelector('button[data-out="' + k + '"]');
    if (b) b.click();
  });

  /* ---------- section rail ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".rail a[href^='#']"));
  if ("IntersectionObserver" in window && links.length) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (en) {
        if (!en.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle("on", a.getAttribute("href") === "#" + en.target.id); });
      });
    }, { rootMargin: "-20% 0px -70% 0px" });
    links.forEach(function (a) { var t = document.getElementById(a.getAttribute("href").slice(1)); if (t) io.observe(t); });
  }

  /* ---------- tooltip ---------- */
  var tip = document.getElementById("tip");
  if (tip) document.addEventListener("pointermove", function (e) {
    var t = e.target.closest && e.target.closest("[data-tip]"), txt = t && t.getAttribute("data-tip");
    if (!txt) { tip.hidden = true; return; }
    tip.textContent = txt; tip.hidden = false;
    var r = tip.getBoundingClientRect(), x = e.clientX + 14, y = e.clientY + 14;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  });

  /* ---------- theme and print ---------- */
  var root = document.documentElement, tb = document.getElementById("btn-theme"), MODES = ["system", "light", "dark"];
  function applyTheme(m) {
    if (m === "system") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", m);
    if (tb) tb.textContent = "Theme: " + m;
  }
  var mode = root.getAttribute("data-theme") || "system", saved = null;
  try { saved = localStorage.getItem("mor-report-theme"); } catch (e) { }
  if (MODES.indexOf(saved) >= 0) mode = saved;
  if (MODES.indexOf(mode) < 0) mode = "system";
  applyTheme(mode);
  if (tb) tb.addEventListener("click", function () {
    mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length]; applyTheme(mode);
    try { localStorage.setItem("mor-report-theme", mode); } catch (e) { }
  });
  var pb = document.getElementById("btn-print");
  if (pb) pb.addEventListener("click", function () { window.print(); });
  var closed = [];
  window.addEventListener("beforeprint", function () {
    closed = Array.prototype.slice.call(document.querySelectorAll("details:not([open])"));
    closed.forEach(function (d) { d.open = true; });
  });
  window.addEventListener("afterprint", function () { closed.forEach(function (d) { d.open = false; }); closed = []; });
})();
