// Voice analysis engine: one ~43 ms frame in, pitch / weight / resonance / breathiness out.
// Runs in the browser (window.VoiceEngine) and in node (module.exports) for offline checks.
(function (root) {
  const FMIN = 70, FMAX = 600;

  // ---- pitch (YIN) ----
  function yin(x, sr) {
    const tauMin = Math.floor(sr / FMAX);
    const tauMax = Math.min(Math.floor(sr / FMIN), Math.floor(x.length / 2));
    const W = x.length - tauMax;
    const d = new Float64Array(tauMax + 1);
    for (let tau = 1; tau <= tauMax; tau++) {
      let s = 0;
      for (let i = 0; i < W; i++) { const v = x[i] - x[i + tau]; s += v * v; }
      d[tau] = s;
    }
    const dn = new Float64Array(tauMax + 1);
    dn[0] = 1;
    let run = 0;
    for (let tau = 1; tau <= tauMax; tau++) { run += d[tau]; dn[tau] = run > 0 ? d[tau] * tau / run : 1; }
    let tau = -1;
    for (let t = tauMin; t <= tauMax; t++) {
      if (dn[t] < 0.15) { while (t + 1 <= tauMax && dn[t + 1] < dn[t]) t++; tau = t; break; }
    }
    if (tau < 0) {
      tau = tauMin;
      for (let t = tauMin; t <= tauMax; t++) if (dn[t] < dn[tau]) tau = t;
    }
    let lag = tau;
    if (tau > 1 && tau < tauMax) {
      const a = dn[tau - 1], b = dn[tau], c = dn[tau + 1], den = a - 2 * b + c;
      if (den) lag = tau + 0.5 * (a - c) / den;
    }
    // harmonics-to-noise from normalized autocorrelation at the period (as Praat's cc method)
    let xy = 0, xx = 0, yy = 0;
    for (let i = 0; i < W; i++) { xy += x[i] * x[i + tau]; xx += x[i] * x[i]; yy += x[i + tau] * x[i + tau]; }
    const r = Math.min(0.9999, Math.max(1e-4, xy / Math.sqrt(xx * yy + 1e-20)));
    return { f0: sr / lag, aperiodicity: dn[tau], hnr: 10 * Math.log10(r / (1 - r)) };
  }

  // ---- spectrum ----
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
          const ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k + len / 2] = re[i + k] - ar; im[i + k + len / 2] = im[i + k] - ai;
          re[i + k] += ar; im[i + k] += ai;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }

  const NFFT = 8192;
  function spectrumDb(x) {
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT), n = x.length;
    for (let i = 0; i < n; i++) re[i] = x[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
    fft(re, im);
    const db = new Float64Array(NFFT / 2);
    for (let i = 0; i < NFFT / 2; i++) db[i] = 10 * Math.log10(re[i] * re[i] + im[i] * im[i] + 1e-20);
    return db;
  }
  function peakDb(db, sr, f, tol = 0.15) {
    const step = sr / NFFT;
    let best = -Infinity;
    for (let i = Math.floor(f * (1 - tol) / step); i <= Math.ceil(f * (1 + tol) / step) && i < db.length; i++) best = Math.max(best, db[i]);
    return best;
  }
  function bandPower(db, sr, lo, hi) {
    const step = sr / NFFT;
    let s = 0;
    for (let i = Math.ceil(lo / step); i < Math.min(db.length, hi / step); i++) s += Math.pow(10, db[i] / 10);
    return s;
  }

  // ---- formants (LPC on an 11025 Hz copy, like Praat's 5500 Hz ceiling) ----
  // The LPC rate must not depend on the input rate: at 48 kHz an integer decimation lands on
  // 12 kHz, which moved "ee" F2 by ~100 Hz on the same take versus 44.1 kHz.
  const LPC_FS = 11025;
  function resample(x, sr) {
    const ratio = sr / LPC_FS, half = 31 * ratio / 4, fc = 0.45 / ratio;
    const n = Math.max(0, Math.floor((x.length - 1 - 2 * half) / ratio)), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const c = i * ratio + half;
      let s = 0, wsum = 0;
      for (let k = Math.ceil(c - half); k <= Math.floor(c + half); k++) {
        const m = k - c;
        const sinc = m === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * m) / (Math.PI * m);
        const h = sinc * (0.54 + 0.46 * Math.cos(Math.PI * m / half));
        s += h * x[k]; wsum += h;
      }
      y[i] = s / wsum;
    }
    return y;
  }
  function polyRoots(c) {
    // Durand-Kerner on z^n + c1 z^(n-1) + ... + cn
    const n = c.length - 1, re = [], im = [];
    for (let i = 0; i < n; i++) { const a = 2 * Math.PI * i / n + 0.4; re.push(0.9 * Math.cos(a)); im.push(0.9 * Math.sin(a)); }
    for (let it = 0; it < 300; it++) {
      let moved = 0;
      for (let i = 0; i < n; i++) {
        let pr = 1, pi = 0;
        for (let k = 1; k <= n; k++) { const t = pr * re[i] - pi * im[i] + c[k]; pi = pr * im[i] + pi * re[i]; pr = t; }
        let dr = 1, di = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const ar = re[i] - re[j], ai = im[i] - im[j], t = dr * ar - di * ai;
          di = dr * ai + di * ar; dr = t;
        }
        const den = dr * dr + di * di || 1e-30, qr = (pr * dr + pi * di) / den, qi = (pi * dr - pr * di) / den;
        re[i] -= qr; im[i] -= qi; moved = Math.max(moved, Math.abs(qr) + Math.abs(qi));
      }
      if (moved < 1e-9) break;
    }
    return re.map((r, i) => [r, im[i]]);
  }
  function formants(x, sr) {
    const fs = LPC_FS, y = resample(x, sr), n = y.length;
    const pre = Math.exp(-2 * Math.PI * 50 / fs);
    for (let i = n - 1; i > 0; i--) y[i] -= pre * y[i - 1];
    for (let i = 0; i < n; i++) y[i] *= 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1));
    const p = 10, r = new Float64Array(p + 1);
    for (let k = 0; k <= p; k++) { let s = 0; for (let i = k; i < n; i++) s += y[i] * y[i - k]; r[k] = s; }
    if (r[0] <= 0) return [];
    r[0] *= 1 + 1e-9;
    let a = new Float64Array(p + 1), err = r[0];
    a[0] = 1;
    for (let i = 1; i <= p; i++) {
      let acc = r[i];
      for (let j = 1; j < i; j++) acc += a[j] * r[i - j];
      const k = -acc / err, b = a.slice();
      for (let j = 1; j < i; j++) b[j] = a[j] + k * a[i - j];
      b[i] = k; a = b; err *= 1 - k * k;
    }
    return polyRoots(Array.from(a))
      .filter(([re, im]) => im > 0)
      .map(([re, im]) => ({ f: Math.atan2(im, re) * fs / (2 * Math.PI), bw: -Math.log(Math.hypot(re, im)) * fs / Math.PI }))
      .filter(o => o.f > 50 && o.f < fs / 2 - 50)
      .sort((u, v) => u.f - v.f);
  }

  // Iseli & Alwan correction: remove the boost a formant (F, B) gives the harmonic at f.
  function formantBoostDb(f, F, B, fs) {
    const r = Math.exp(-Math.PI * B / fs), w = 2 * Math.PI / fs, c = (x) => r * r + 1 - 2 * r * Math.cos(w * x);
    return 10 * Math.log10((c(F) * c(F)) / (c(f + F) * c(f - F)));
  }

  // Hawks & Miller (1995) formant bandwidth from frequency and pitch, as VoiceSauce uses;
  // steadier than per-frame LPC bandwidths.
  function hawksMillerBw(F, f0) {
    const k = F < 500
      ? [165.327516, -6.73636734e-1, 1.80874446e-3, -4.52201682e-6, 7.49514000e-9, -4.70219241e-12]
      : [15.8146139, 8.10159009e-2, -9.79728215e-5, 5.28725064e-8, -1.07099364e-11, 7.91528509e-16];
    let b = 0;
    for (let i = 0; i < k.length; i++) b += k[i] * Math.pow(F, i);
    return (1 + 0.25 * (f0 - 132) / 88) * b;
  }

  // ---- extra spectral signals for the trained models ----
  const melCache = {};
  function melBank(sr) {
    if (melCache[sr]) return melCache[sr];
    const nb = 26, lo = 50, hi = Math.min(8000, sr / 2), step = sr / NFFT;
    const mel = (f) => 2595 * Math.log10(1 + f / 700), inv = (m) => 700 * (Math.pow(10, m / 2595) - 1);
    const pts = Array.from({ length: nb + 2 }, (_, i) => inv(mel(lo) + (i * (mel(hi) - mel(lo))) / (nb + 1)));
    return (melCache[sr] = Array.from({ length: nb }, (_, b) => {
      const w = [];
      for (let k = Math.floor(pts[b] / step); k <= Math.ceil(pts[b + 2] / step); k++) {
        const f = k * step, v = f < pts[b + 1] ? (f - pts[b]) / (pts[b + 1] - pts[b]) : (pts[b + 2] - f) / (pts[b + 2] - pts[b + 1]);
        if (v > 0) w.push([k, v]);
      }
      return w;
    }));
  }
  function extraSignals(db, sr, f0, F1, F2, F3) {
    const step = sr / NFFT, H1 = peakDb(db, sr, f0, 0.1);
    const harmAt = (F) => peakDb(db, sr, Math.max(1, Math.round(F / f0)) * f0, 0.08);
    let num = 0, den = 0;
    for (let i = Math.ceil(50 / step); i < 5000 / step; i++) { const p = Math.pow(10, db[i] / 10); num += p * i * step; den += p; }
    const band = (a, b) => bandPower(db, sr, a, b);
    const out = {
      centroid: num / den,
      ring: 10 * Math.log10(band(2000, 4000) / band(50, 1000)),
      air: 10 * Math.log10(band(5000, Math.min(8000, sr / 2)) / band(50, 1000)),
      h1a1: F1 ? H1 - harmAt(F1) : null,
      h1a2: F2 ? H1 - harmAt(F2) : null,
      h1a3: F3 ? H1 - harmAt(F3) : null,
      fmean: F1 && F2 && F3 ? Math.cbrt(F1 * F2 * F3) : null,
      disp: F1 && F3 ? (F3 - F1) / 2 : null,
    };
    // MFCC 1-12; c0 is overall level, so it's left out.
    const bank = melBank(sr), logE = bank.map((w) => Math.log(w.reduce((s, [k, v]) => s + v * Math.pow(10, db[k] / 10), 0) + 1e-20));
    for (let c = 1; c <= 12; c++) out["mfcc" + c] = logE.reduce((s, v, b) => s + v * Math.cos((Math.PI * c * (b + 0.5)) / logE.length), 0);
    // Cepstral peak prominence: cepstrum peak in the pitch range above a regression line.
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
    for (let i = 0; i < NFFT / 2; i++) { re[i] = db[i]; re[NFFT - 1 - i] = db[i]; }
    fft(re, im);
    const qa = Math.floor(sr / 600), qb = Math.min(NFFT / 2 - 1, Math.ceil(sr / 60)), c = (q) => Math.abs(re[q]) / NFFT;
    let pk = qa, sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
    for (let q = qa; q <= qb; q++) { if (c(q) > c(pk)) pk = q; sx += q; sy += c(q); sxx += q * q; sxy += q * c(q); n++; }
    const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    out.cpp = c(pk) - ((sy - slope * sx) / n + slope * pk);
    return out;
  }
  // Per-take signals the trained models choose from; each is the median over voiced frames.
  const wIndex = (f) => (f.h1h2c != null && f.h2h4c != null ? (f.h1h2c + f.h2h4c) / 2 : null);
  const SIGNALS = [
    ["F1", (f) => f.F1], ["F2", (f) => f.F2], ["F3", (f) => f.F3], ["formant mean", (f) => f.fmean], ["formant spacing", (f) => f.disp],
    ["centroid", (f) => f.centroid], ["tilt (alpha)", (f) => f.alpha], ["ring 2-4k", (f) => f.ring], ["air 5-8k", (f) => f.air],
    ["H1-H2 raw", (f) => f.h1h2], ["H1*-H2*", (f) => f.h1h2c], ["H2*-H4*", (f) => f.h2h4c], ["weight index", wIndex],
    ["H1-A1", (f) => f.h1a1], ["H1-A2", (f) => f.h1a2], ["H1-A3", (f) => f.h1a3], ["HNR", (f) => f.hnr], ["CPP", (f) => f.cpp], ["pitch", (f) => f.f0],
    ...Array.from({ length: 12 }, (_, i) => ["MFCC " + (i + 1), (f) => f["mfcc" + (i + 1)]]),
  ].map(([name, val]) => ({ name: "all " + name, val }));

  // ---- one frame ----
  function analyzeFrame(x, sr) {
    let ss = 0;
    for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const level = 10 * Math.log10(ss / x.length + 1e-20);
    const out = { level, voiced: false };
    if (level < -60) return out;
    const p = yin(x, sr);
    out.aperiodicity = p.aperiodicity;
    if (p.aperiodicity > 0.25 || p.f0 < FMIN || p.f0 > FMAX) return out;
    const db = spectrumDb(x), f0 = p.f0;
    const fm = formants(x, sr);
    const F1 = fm[0], F2 = fm[1], F3 = fm[2];
    const H1 = peakDb(db, sr, f0), H2 = peakDb(db, sr, 2 * f0), H4 = peakDb(db, sr, 4 * f0);
    let h1h2c = null, h2h4c = null;
    // When F1 lands within 1.25x of pitch the tracker has usually locked onto a harmonic;
    // correcting with it would swing weight by 10+ dB, so report weight as unreadable instead.
    const f1Locked = !F1 || F1.f < 1.25 * f0;
    if (F1 && F2 && !f1Locked) {
      const corr = (f) => formantBoostDb(f, F1.f, hawksMillerBw(F1.f, f0), sr) + formantBoostDb(f, F2.f, hawksMillerBw(F2.f, f0), sr);
      h1h2c = (H1 - corr(f0)) - (H2 - corr(2 * f0));
      h2h4c = (H2 - corr(2 * f0)) - (H4 - corr(4 * f0));
    }
    Object.assign(out, {
      voiced: true,
      f0,
      hnr: p.hnr,
      h1h2: H1 - H2,
      h1h2c, h2h4c,
      alpha: 10 * Math.log10(bandPower(db, sr, 1000, 5000) / bandPower(db, sr, 50, 1000)),
      F1: F1 && F1.f, F2: F2 && F2.f, F3: F3 && F3.f,
      // H2 sitting on F1 inflates H2 and makes the voice read heavier than it is.
      weightUnreliable: f1Locked || Math.abs(2 * f0 - F1.f) < 0.15 * F1.f,
      weightUnreadable: f1Locked,
    }, extraSignals(db, sr, f0, F1 && F1.f, F2 && F2.f, F3 && F3.f));
    return out;
  }

  // ---- live source (browser only) ----
  // `gen` changes on every start and stop, so a start still waiting on the mic or a clip
  // gives up instead of running alongside the newer one. start() resolves false when superseded.
  let ctx = null, raf = 0, cleanup = null, gen = 0;
  async function start(src, onFrame, onEnd, opts = {}) {
    stop();
    const my = ++gen, c = new AudioContext();
    ctx = c;
    const an = c.createAnalyser();
    an.fftSize = 2048;
    an.smoothingTimeConstant = 0;
    let stream = null, el = null;
    if (src.kind === "mic") {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
      if (my !== gen) { stream.getTracks().forEach((t) => t.stop()); return false; }
      const mic = c.createMediaStreamSource(stream);
      mic.connect(an);
      if (opts.onAudio) {
        // Raw-audio tap so a page can save what it heard; routed through a muted gain so it runs.
        const tap = c.createScriptProcessor(4096, 1, 1), mute = c.createGain();
        mute.gain.value = 0;
        mic.connect(tap); tap.connect(mute); mute.connect(c.destination);
        const sr = c.sampleRate;
        tap.onaudioprocess = (e) => { if (my === gen) opts.onAudio(new Float32Array(e.inputBuffer.getChannelData(0)), sr); };
      }
    } else {
      el = new Audio(src.url);
      c.createMediaElementSource(el).connect(an);
      an.connect(c.destination);
      el.onended = () => { if (my !== gen) return; stop(); onEnd && onEnd(); };
      try { await el.play(); } catch (e) { if (my !== gen) return false; throw e; }
      if (my !== gen) { el.pause(); return false; }
    }
    const buf = new Float32Array(an.fftSize), t0 = c.currentTime;
    const tick = () => {
      if (my !== gen) return;
      an.getFloatTimeDomainData(buf);
      const fr = analyzeFrame(buf, c.sampleRate);
      fr.t = c.currentTime - t0;
      onFrame(fr);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    cleanup = () => {
      cancelAnimationFrame(raf);
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (el) el.pause();
    };
    return true;
  }
  function stop() {
    gen++;
    if (cleanup) cleanup();
    cleanup = null;
    if (ctx && ctx.state !== "closed") ctx.close();
    ctx = null;
  }

  const api = { analyzeFrame, start, stop, spectrumDb, peakDb, bandPower, fft, NFFT, SIGNALS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.VoiceEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
