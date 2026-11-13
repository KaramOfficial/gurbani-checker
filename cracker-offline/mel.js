/* Log-mel featurization matching the Python validation pipeline
 * (librosa.feature.melspectrogram: sr=16000, n_fft=512, hop=160,
 * win=400 hann, 80 mels, fmin=0, fmax=8000, htk mel scale, power=1.0
 * magnitude, center=False), then log(x + 2^-24) and per-feature
 * (per mel-bin) normalization over time — the input format the
 * IndicConformer CTC model was exported with.
 * Exposes window.MelFeaturizer.compute(Float32Array) -> {data, frames}
 * where data is Float32Array [80 * frames] in [bin][time] order. */
(function (global) {
  "use strict";
  var SR = 16000, N_FFT = 512, HOP = 160, WIN = 400, N_MELS = 80;

  // --- radix-2 FFT (in-place, real input) ---
  function fft(re, im) {
    var n = re.length;
    for (var i = 1, j = 0; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        var tr = re[i]; re[i] = re[j]; re[j] = tr;
        var ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    for (var len = 2; len <= n; len <<= 1) {
      var ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (var i = 0; i < n; i += len) {
        var cwr = 1, cwi = 0;
        for (var k = 0; k < len / 2; k++) {
          var ur = re[i + k], ui = im[i + k];
          var vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
          var vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
          var nwr = cwr * wr - cwi * wi; cwi = cwr * wi + cwi * wr; cwr = nwr;
        }
      }
    }
  }

  // --- mel filterbank, librosa-compatible (htk scale, slaney norm) ---
  function hzToMel(f) { return 2595 * Math.log10(1 + f / 700); }
  function melToHz(m) { return 700 * (Math.pow(10, m / 2595) - 1); }
  var filters = null; // [N_MELS][N_FFT/2+1]
  function buildFilters() {
    var nBins = N_FFT / 2 + 1;
    var melMin = hzToMel(0), melMax = hzToMel(8000);
    var pts = [];
    for (var i = 0; i < N_MELS + 2; i++) pts.push(melToHz(melMin + (melMax - melMin) * i / (N_MELS + 1)));
    filters = [];
    for (var m = 0; m < N_MELS; m++) {
      var row = new Float32Array(nBins);
      var f0 = pts[m], f1 = pts[m + 1], f2 = pts[m + 2];
      var enorm = 2.0 / (f2 - f0);
      for (var b = 0; b < nBins; b++) {
        var freq = b * SR / N_FFT;
        var w = 0;
        if (freq >= f0 && freq <= f1) w = (freq - f0) / (f1 - f0);
        else if (freq > f1 && freq <= f2) w = (f2 - freq) / (f2 - f1);
        row[b] = w * enorm;
      }
      filters.push(row);
    }
  }

  var hann = null;
  function buildHann() {
    hann = new Float32Array(WIN);
    for (var i = 0; i < WIN; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / WIN);
  }

  function compute(samples) {
    if (!filters) buildFilters();
    if (!hann) buildHann();
    var nFrames = Math.max(0, Math.floor((samples.length - WIN) / HOP) + 1);
    var nBins = N_FFT / 2 + 1;
    var out = new Float32Array(N_MELS * nFrames);
    var re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
    var mag = new Float64Array(nBins);
    for (var t = 0; t < nFrames; t++) {
      var off = t * HOP;
      for (var i = 0; i < N_FFT; i++) {
        re[i] = i < WIN ? samples[off + i] * hann[i] : 0;
        im[i] = 0;
      }
      fft(re, im);
      for (var b = 0; b < nBins; b++) mag[b] = Math.sqrt(re[b] * re[b] + im[b] * im[b]);
      for (var m = 0; m < N_MELS; m++) {
        var row = filters[m], sum = 0;
        for (var k = 0; k < nBins; k++) sum += row[k] * mag[k];
        out[m * nFrames + t] = Math.log(sum + 5.960464477539063e-08); // log(x + 2^-24), as validated
      }
    }
    // --- per-feature normalization over time ---
    for (var m2 = 0; m2 < N_MELS; m2++) {
      var base = m2 * nFrames, mean = 0, s2 = 0, i2;
      for (i2 = 0; i2 < nFrames; i2++) mean += out[base + i2];
      mean /= Math.max(1, nFrames);
      for (i2 = 0; i2 < nFrames; i2++) { var d = out[base + i2] - mean; s2 += d * d; }
      var std = Math.sqrt(s2 / Math.max(1, nFrames)) + 1e-5;
      for (i2 = 0; i2 < nFrames; i2++) out[base + i2] = (out[base + i2] - mean) / std;
    }
    return { data: out, frames: nFrames };
  }

  global.MelFeaturizer = { compute: compute, N_MELS: N_MELS };
})(typeof window !== "undefined" ? window : globalThis);
