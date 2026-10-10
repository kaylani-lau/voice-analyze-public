// The voice map's model, for the browser and node: per-take signal medians and a ridge model per
// axis. A take is { id, label, createdAt, features: { "<signal name>": median } }.
// A model scores -1 (heavy / dark) .. +1 (light / bright).
(function (root) {
  const E = typeof module !== "undefined" && module.exports ? require("./engine.js") : root.VoiceEngine;
  const AXES = [{ key: "weight", pos: "light", neg: "heavy" }, { key: "brightness", pos: "bright", neg: "dark" }];

  const median5 = (a) => { a = a.filter((v) => v != null && isFinite(v)).sort((p, q) => p - q); return a.length >= 5 ? a[a.length >> 1] : null; };
  // Medians over the voiced frames of a take, one per signal the engine computes.
  function featuresOf(frames) {
    const v = frames.filter((f) => f.voiced), out = {};
    for (const s of E.SIGNALS) out[s.name] = median5(v.map(s.val));
    return out;
  }

  // A new session starts after a 10-minute gap between takes.
  function sessions(takes) {
    const sorted = takes.slice().sort((a, b) => Date.parse(a.createdAt) - Date.parse(b.createdAt)), out = {};
    let s = 0;
    sorted.forEach((t, i) => { if (i && Date.parse(t.createdAt) - Date.parse(sorted[i - 1].createdAt) > 600e3) s++; out[t.id] = s; });
    return out;
  }

  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const sdev = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, a.length - 1)); };
  function inv(A) {
    const n = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      const d = M[c][c] || 1e-12; for (let k = 0; k < 2 * n; k++) M[c][k] /= d;
      for (let r = 0; r < n; r++) if (r !== c && M[r][c]) { const f = M[r][c]; for (let k = 0; k < 2 * n; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((r) => r.slice(n));
  }
  // Linear-kernel ridge on standardized features; the penalty is picked by closed-form leave-one-out.
  function ridgeFit(X, y) {
    const n = X.length, p = X[0].length, mu = [], sd = [];
    for (let j = 0; j < p; j++) { const col = X.map((r) => r[j]); mu.push(mean(col)); sd.push(sdev(col) || 1); }
    const Z = X.map((r) => r.map((v, j) => (v - mu[j]) / sd[j])), t = y.map((v) => (v ? 1 : -1)), b = mean(t), tc = t.map((v) => v - b);
    const K = Z.map((a) => Z.map((c) => a.reduce((s, v, k) => s + v * c[k], 0)));
    let best = null;
    for (const lam of [0.3, 1, 3, 10, 30, 100, 300]) {
      const Gi = inv(K.map((r, i) => r.map((v, j) => v + (i === j ? lam : 0)))), a = Gi.map((r) => r.reduce((s, v, j) => s + v * tc[j], 0));
      let ok = 0; for (let i = 0; i < n; i++) { const loo = t[i] - a[i] / Gi[i][i]; if ((loo > 0) === !!y[i]) ok++; }
      if (!best || ok >= best.ok) best = { ok, a, lam };
    }
    const w = Array.from({ length: p }, (_, k) => Z.reduce((s, z, i) => s + best.a[i] * z[k], 0));
    return { w, mu, sd, b, lam: best.lam, looOk: best.ok };
  }
  const scoreRow = (a, r) => a.b + r.reduce((s, v, k) => s + a.w[k] * ((v ?? a.mu[k]) - a.mu[k]) / a.sd[k], 0);

  // Returns null with a reason when there aren't enough takes on each side of each axis.
  function train(takes) {
    for (const ax of AXES) {
      const pos = takes.filter((t) => t.label.includes(ax.pos)).length;
      if (pos < 2 || takes.length - pos < 2) return { error: `Record at least 2 ${ax.pos} and 2 ${ax.neg} takes.` };
    }
    const features = E.SIGNALS.map((s) => s.name).filter((n) => takes.every((t) => t.features[n] != null));
    const X = takes.map((t) => features.map((n) => t.features[n])), ses = sessions(takes), nSes = new Set(Object.values(ses)).size;
    const model = { kind: "ridge", trainedAt: new Date().toISOString(), trainedOn: takes.length, sessions: nSes, features, axes: {} };
    for (const ax of AXES) {
      const y = takes.map((t) => (t.label.includes(ax.pos) ? 1 : 0)), m = ridgeFit(X, y);
      // Accuracy on takes it didn't learn from: whole held-out sessions when there are 2+, else leave-one-out.
      let ok = 0, n = 0;
      if (nSes > 1) {
        for (const s of new Set(Object.values(ses))) {
          const tr = takes.map((_, i) => i).filter((i) => ses[takes[i].id] !== s), ytr = tr.map((i) => y[i]);
          if (!ytr.some(Boolean) || ytr.every(Boolean)) continue;
          const f = ridgeFit(tr.map((i) => X[i]), ytr);
          takes.forEach((t, i) => { if (ses[t.id] === s) { n++; if ((scoreRow(f, X[i]) > 0) === !!y[i]) ok++; } });
        }
      } else { ok = m.looOk; n = takes.length; }
      model.axes[ax.key] = { pos: ax.pos, neg: ax.neg, w: m.w, mu: m.mu, sd: m.sd, b: m.b, lam: m.lam, test: { ok, n, by: nSes > 1 ? "session" : "take" } };
    }
    return model;
  }
  // Scores one take, or a live window's features, as { x: brightness, y: weight }.
  function score(model, features) {
    const r = model.features.map((n) => features[n]);
    return { x: scoreRow(model.axes.brightness, r), y: scoreRow(model.axes.weight, r) };
  }

  const api = { AXES, featuresOf, sessions, train, score, ridgeFit };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.VoiceModel = api;
})(typeof window !== "undefined" ? window : globalThis);
