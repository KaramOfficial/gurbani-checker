/* Cracker aligner — tracks position in a known text from a stream of
 * imperfectly recognized words. Beam of competing positions; each incoming
 * word votes. Matching is on a folded Gurmukhi form (nukta folded, addak
 * dropped, punctuation/vishraam stripped) with character-level Levenshtein
 * similarity, plus merged-word (ladiwar) candidates.
 * Works in browser (window.CrackerAligner) and Node (module.exports). */
(function (global) {
  "use strict";

  var NUKTA_FOLD = {
    "\u0a5a": "\u0a17", // khha -> ga
    "\u0a5b": "\u0a1c", // za -> ja
    "\u0a59": "\u0a16", // kha-nukta -> kha
    "\u0a5e": "\u0a2b", // fa -> pha
    "\u0a36": "\u0a38", // sha -> sa
    "\u0a33": "\u0a32"  // lla -> la
  };

  function foldWord(w) {
    if (!w) return "";
    var t = String(w).normalize("NFC");
    var out = "";
    for (var i = 0; i < t.length; i++) {
      var ch = t[i];
      if (NUKTA_FOLD[ch]) ch = NUKTA_FOLD[ch];
      if (ch === "\u0a71") continue;            // addak: dropped in folded form
      if (/[\s;.,\u0964\u0965\u00ab\u00bb'"\-:]/.test(ch)) continue; // punct / vishraam marks
      if (/[0-9\u0a66-\u0a6f]/.test(ch)) continue; // digits (unspoken markers)
      if (ch === "\u0a1e") ch = "\u0a28";        // nya -> na
      if (ch === "\u0a19") ch = "\u0a28";        // nga -> na
      out += ch;
    }
    return out;
  }

  // normalized Levenshtein similarity in [0,1]
  function sim(a, b) {
    if (a === b) return 1;
    if (!a.length || !b.length) return 0;
    var la = a.length, lb = b.length;
    var prev = new Array(lb + 1), cur = new Array(lb + 1);
    for (var j = 0; j <= lb; j++) prev[j] = j;
    for (var i = 1; i <= la; i++) {
      cur[0] = i;
      for (var k = 1; k <= lb; k++) {
        var cost = a[i - 1] === b[k - 1] ? 0 : 1;
        cur[k] = Math.min(prev[k] + 1, cur[k - 1] + 1, prev[k - 1] + cost);
      }
      var tmp = prev; prev = cur; cur = tmp;
    }
    return 1 - prev[lb] / Math.max(la, lb);
  }

  /* Tracker over target words: [{w (original), line}] */
  function Tracker(words) {
    this.words = words;
    this.folded = words.map(function (x) { return foldWord(x.w); });
    // beam: Map wordIdx(next expected) -> {score, matched, lastAdvance}
    this.beam = new Map();
    this.beam.set(0, { score: 0, matched: 0 });
    this.cursor = 0;          // best tentative next-word index
    this.lockedCursor = -1;   // last locked next-word index
    this.state = "searching";
    this.eventsSeen = 0;
    this.lowConfStreak = 0;
    this.sinceAdvance = 0;
    this.lastCursor = 0;
    // sequence-anchor machinery: folded word -> all its positions, and a
    // rolling history of heard words, so a reciter who jumps to a totally
    // different part of the text is re-found from the word SEQUENCE alone
    this.posIndex = {};
    for (var pi = 0; pi < this.folded.length; pi++) {
      var fw = this.folded[pi];
      if (fw) { (this.posIndex[fw] = this.posIndex[fw] || []).push(pi); }
    }
    this.history = [];   // last folded heard words, oldest first
    this.wideEvents = 0; this.jumpCooldown = 0; // after softReset: unrestricted re-acquisition window
    // jump-stability state: a distant jump must be proposed twice in a
    // row before it commits, and jumping BACK toward where we just came
    // from faces a higher bar for a while — together these stop the
    // first-line/last-line refrain twins from ping-ponging the cursor
    this.pendingJump = null;   // {end, at, count}
    this.lastJumpFrom = -1;    // cursor position before the last committed jump
    this.lastJumpAt = -1000000; // eventsSeen at the last committed jump
  }

  // avg sim of heardSeq (folded, length n) against target words [start, start+n)
  Tracker.prototype.seqScore = function (start, heardSeq) {
    var n = heardSeq.length, total = 0;
    for (var i = 0; i < n; i++) {
      if (start + i < 0 || start + i >= this.folded.length) return 0;
      total += sim(heardSeq[i], this.folded[start + i]);
    }
    return total / n;
  };

  // Best DISTANT sequence match for the last n heard words, plus how well
  // those same words fit exactly where the cursor already expects them.
  Tracker.prototype.distantMatch = function (n) {
    var H = this.history;
    if (H.length < n) return null;
    var seq = H.slice(H.length - n);
    var lastWord = seq[n - 1];
    var ends = this.posIndex[lastWord] || [];
    var bestStart = -1, bestScore = 0;
    for (var ei = 0; ei < ends.length; ei++) {
      var start = ends[ei] - (n - 1);
      if (start < 0) continue;
      if (Math.abs(start + n - this.cursor) <= 15) continue; // not distant
      var sc = this.seqScore(start, seq);
      if (sc > bestScore) { bestScore = sc; bestStart = start; }
    }
    if (bestStart < 0) return null;
    return { start: bestStart, score: bestScore, local: this.seqScore(this.cursor - n, seq) };
  };

  // Jump detection, stay-first. Anand Sahib's pauris open and close with
  // near-identical lines, so a short heard sequence often matches a
  // distant refrain twin almost perfectly while a word or two of
  // recognition noise dents the local fit — the old rule snapped on that
  // and then snapped back, and the screen jumped. Now:
  //  (1) LONG context is tried first (last 8-10 words = the tail of the
  //      previous sentence, which the twins do NOT share);
  //  (2) a short-window jump is vetoed when the same words also fit well
  //      somewhere near the current position (refrain ambiguity);
  //  (3) any jump must be proposed on two consecutive events before it
  //      commits — one noisy burst can no longer teleport the cursor;
  //  (4) after a jump, jumping BACK toward the pre-jump position needs
  //      the long-context proof for the next 40 events.
  // A genuine relocation still re-acquires: it matches long, matches
  // twice, and its local fit stays poor — it just takes ~1 extra word.
  Tracker.prototype.checkJump = function () {
    if (this.jumpCooldown > 0) { this.jumpCooldown--; return false; }
    var H = this.history;
    if (H.length < 5) return false;
    var cand = null, li, si, n, m;
    var longNs = [10, 9, 8];
    for (li = 0; li < longNs.length && !cand; li++) {
      n = longNs[li];
      if (H.length < n) continue;
      m = this.distantMatch(n);
      if (!m) continue;
      if (m.local >= 0.85) { this.pendingJump = null; return false; } // long local fit is good: stay
      if (m.score >= 0.95 && m.score >= m.local + 0.25) cand = { start: m.start, n: n, strong: true };
    }
    if (!cand) {
      var shortNs = [6, 5];
      for (si = 0; si < shortNs.length && !cand; si++) {
        n = shortNs[si];
        m = this.distantMatch(n);
        if (!m) continue;
        if (m.local >= 0.8) { this.pendingJump = null; return false; }
        // refrain ambiguity: the same words also fit near where we are
        var seq = H.slice(H.length - n), nearBest = 0;
        for (var st = Math.max(0, this.cursor - n - 8); st <= this.cursor - n + 8; st++) {
          var nsc = this.seqScore(st, seq);
          if (nsc > nearBest) nearBest = nsc;
        }
        if (nearBest >= 0.85) continue; // explained near here — a twin, not a move
        var need = this.state === "locked" ? 0.95 : 0.9;
        if (m.score >= need && m.score >= m.local + 0.35) cand = { start: m.start, n: n, strong: false };
      }
    }
    if (!cand) { this.pendingJump = null; return false; }
    var targetEnd = cand.start + cand.n;
    // (4) no cheap ride back to where the last jump came from
    if (!cand.strong && this.lastJumpFrom >= 0 &&
        this.eventsSeen - this.lastJumpAt < 40 &&
        Math.abs(targetEnd - this.lastJumpFrom) <= 25) {
      this.pendingJump = null;
      return false;
    }
    // (3) two consecutive proposals, tracking the same moving endpoint
    var p = this.pendingJump;
    if (p && Math.abs(targetEnd - (p.end + (this.eventsSeen - p.at))) <= 2) {
      p.count++; p.end = targetEnd; p.at = this.eventsSeen;
    } else {
      this.pendingJump = p = { end: targetEnd, at: this.eventsSeen, count: 1 };
    }
    if (p.count < 2) return false;
    // commit
    var from = this.cursor;
    this.beam = new Map();
    this.beam.set(targetEnd, { score: 4, matched: cand.n + 2 });
    this.cursor = targetEnd;
    this.lastCursor = targetEnd;
    this.anchorIdx = targetEnd;
    this.anchorEvents = this.eventsSeen;
    this.sinceAdvance = 0;
    this.state = "tentative";
    this.jumpCooldown = 30; // no second jump for a while — prevents thrash
    this.lastJumpFrom = from;
    this.lastJumpAt = this.eventsSeen;
    this.pendingJump = null;
    return true;
  };

  // After Stop/Start or a long stall: forget banked evidence but remember
  // roughly where we were; open a short unrestricted re-acquisition window
  Tracker.prototype.softReset = function () {
    var at = this.cursor;
    this.beam = new Map();
    this.beam.set(at, { score: 0, matched: 0 });
    this.state = "searching";
    this.lowConfStreak = 0;
    this.sinceAdvance = 0;
    this.history = [];
    this.pendingJump = null;
    this.lastJumpFrom = -1;
    this.lastJumpAt = -1000000;
    this.wideEvents = 12;
    return this.snapshot();
  };

  Tracker.prototype.feed = function (rawWord) {
    var inw = foldWord(rawWord);
    if (!inw) return this.snapshot();
    this.eventsSeen++;
    this.history.push(inw);
    if (this.history.length > 14) this.history.shift();
    if (this.checkJump()) return this.snapshot();
    var next = new Map();
    var self = this;
    // exponential forgetting: old evidence fades so a stalled hypothesis
    // cannot squat on its banked score forever
    this.beam.forEach(function (hyp) { hyp.score *= 0.90; });
    function put(idx, score, matched) {
      if (idx < 0 || idx > self.words.length) return;
      var cur = next.get(idx);
      if (!cur || score > cur.score) next.set(idx, { score: score, matched: matched });
    }
    this.beam.forEach(function (hyp, idx) {
      var s = hyp.score, m = hyp.matched;
      // stay (word unrelated / insertion): penalty, position unchanged
      put(idx, s - 1.0, m);
      if (idx < self.words.length) {
        var s1 = sim(inw, self.folded[idx]);
        if (s1 >= 0.55) put(idx + 1, s + 2 * s1, m + 1);            // match next word
        else if (s1 >= 0.3) put(idx + 1, s - 0.6 + s1, m);          // weak substitute, advance cautiously
        // skip one target word then match
        if (idx + 1 < self.words.length) {
          var s2 = sim(inw, self.folded[idx + 1]);
          if (s2 >= 0.55) put(idx + 2, s - 1.6 + 2 * s2, m + 1);
          // ladiwar: incoming word = two target words merged
          var s3 = sim(inw, self.folded[idx] + self.folded[idx + 1]);
          if (s3 >= 0.6) put(idx + 2, s + 2 * s3, m + 1);
        }
        // loop-back: reciter repeats current line's first word etc. -> allow
        // small backward jump to the start of the current line
        var lineStart = idx;
        while (lineStart > 0 && self.words[lineStart - 1].line === self.words[idx].line) lineStart--;
        if (lineStart < idx) {
          var s4 = sim(inw, self.folded[lineStart]);
          if (s4 >= 0.6) put(lineStart + 1, s - 1.5 + 2 * s4, m + 1);
        }
      }
    });
    // global re-acquisition: if the best hypothesis has not advanced for
    // several events (stuck / lost), seed fresh matches anywhere in the text
    var arr = Array.from(next.entries()).sort(function (a, b) { return b[1].score - a[1].score; });
    var bestIdx = arr.length ? arr[0][0] : 0;
    if (bestIdx > this.lastCursor) { this.sinceAdvance = 0; this.lastCursor = bestIdx; }
    else this.sinceAdvance++;
    var wide = this.wideEvents > 0;
    if (wide) this.wideEvents--;
    if (arr.length && (wide || arr[0][1].score < 1.0 || this.sinceAdvance >= 6)) {
      // Re-acquisition is restricted: near the current position, OR on a
      // rare word (<=3 occurrences in the whole text) which is a safe
      // global anchor. Common repeated words may NOT teleport the cursor —
      // that is exactly how repeated refrains trap the tracker. During the
      // short window after a softReset (user pressed Listen again), the
      // restriction lifts: the user may have moved anywhere on purpose.
      if (!this.freq) {
        this.freq = {};
        for (var fi = 0; fi < this.folded.length; fi++) this.freq[this.folded[fi]] = (this.freq[this.folded[fi]] || 0) + 1;
      }
      var anchor = arr[0][0];
      for (var k = 0; k < this.words.length; k++) {
        var near = Math.abs(k - anchor) <= 40;
        var rare = (this.freq[this.folded[k]] || 99) <= 3;
        if (!wide && !near && !rare) continue;
        var sg = sim(inw, this.folded[k]);
        if (sg >= 0.8) {
          var cur2 = next.get(k + 1);
          var sc = arr[0][1].score - 1.0; // re-acquisition costs a little evidence
          if (!cur2 || sc > cur2.score) next.set(k + 1, { score: sc, matched: 3 });
        }
      }
      this.sinceAdvance = 0;
      arr = Array.from(next.entries()).sort(function (a, b) { return b[1].score - a[1].score; });
    }
    // NOTE: a pace prior was tried here (penalize hypotheses far from the
    // event-count expectation) and REJECTED by measurement — with drops
    // and merges, event count underestimates true position, so the prior
    // punished the correct path (synthetic within-1 fell 95% -> 15%).
    // Pace belongs in the CTC engine, where time is the axis and a path
    // cannot teleport. Word-event tracking stays evidence-only.
    arr = Array.from(next.entries()).sort(function (a, b) { return b[1].score - a[1].score; });
    // prune to top 20, but keep position diversity: at most 4 hypotheses
    // per 10-word bucket, so one repeated passage cannot flood the beam
    // and evict the true position
    var bucketCount = {}, kept = [];
    for (var bi = 0; bi < arr.length && kept.length < 20; bi++) {
      var bk = Math.floor(arr[bi][0] / 10);
      if ((bucketCount[bk] || 0) >= 4) continue;
      bucketCount[bk] = (bucketCount[bk] || 0) + 1;
      kept.push(arr[bi]);
    }
    this.beam = new Map(kept);
    if (arr.length) {
      // Sticky cursor (display hysteresis): when a refrain twin runs
      // neck-and-neck with the true position through an identical
      // stretch, the reported cursor stays with the incumbent unless the
      // challenger beats every near-incumbent hypothesis by a clear
      // margin (2.5 ≈ more than a word of evidence). The beam itself is
      // untouched — only which hypothesis is surfaced. When the lines
      // diverge, the true path outscores the stalled twin within a word
      // or two and the cursor follows.
      var chosen = arr[0];
      for (var ci = 0; ci < arr.length; ci++) {
        if (Math.abs(arr[ci][0] - this.cursor) <= 4) {
          if (arr[0][1].score - arr[ci][1].score <= 2.5) chosen = arr[ci];
          break;
        }
      }
      this.cursor = chosen[0];
      var best = arr[0][1];
      var second = arr.length > 1 ? arr[1][1].score : -Infinity;
      var gap = best.score - second;
      if (best.matched >= 4 && gap >= 2.0) {
        this.state = "locked";
        this.lockedCursor = this.cursor;
        this.lowConfStreak = 0;
      } else if (best.matched >= 2) {
        this.state = this.state === "locked" && gap < 0.5 ? "lost" : "tentative";
        if (this.state === "lost") this.lowConfStreak++; else this.lowConfStreak = 0;
      } else {
        this.state = "searching";
      }
    }
    return this.snapshot();
  };

  Tracker.prototype.snapshot = function () {
    var idx = Math.max(0, Math.min(this.cursor, this.words.length - 1));
    return {
      state: this.state,
      wordIdx: this.cursor,
      line: this.words.length ? this.words[idx].line : 0,
      lockedWordIdx: this.lockedCursor,
      eventsSeen: this.eventsSeen
    };
  };

  // manual set: user taps a line/word
  Tracker.prototype.setPosition = function (wordIdx) {
    this.beam = new Map();
    this.beam.set(wordIdx, { score: 3, matched: 4 });
    this.cursor = wordIdx;
    this.lockedCursor = wordIdx;
    this.state = "locked";
    this.anchorIdx = wordIdx;
    this.anchorEvents = this.eventsSeen;
    this.lastCursor = wordIdx;
    this.pendingJump = null;
    this.lastJumpFrom = -1;
    this.lastJumpAt = -1000000;
    this.sinceAdvance = 0;
    return this.snapshot();
  };

  var api = { foldWord: foldWord, sim: sim, Tracker: Tracker };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.CrackerAligner = api;
})(typeof window !== "undefined" ? window : globalThis);
