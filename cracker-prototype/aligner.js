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
    // nudge policy state (v3): while a good lock is fresh, words that fit
    // a distant place instead of here first earn ONE gentle nudge, and a
    // jump then needs either a pause before the new passage (pausePending,
    // set by the page via notePause) or persistent proof (many proposals)
    this.pausePending = false;
    this.wasLockedAt = -1000000; // eventsSeen when the tracker last held a lock
    this.nudgeAt = -1000000;
    this.episodeNudged = false;
    this._nudge = false;
    // near-move (lurch) detection: the beam follows skips of a line or
    // two silently — a single feed that moves the REPORTED position by
    // 5+ words while a lock is fresh is a skip/repeat, and must buzz
    this.lastRep = -1;      // last reported (snapshot) wordIdx
    this.lurchCalm = 99;    // feeds since the last big lurch
    this.justJumped = false; // a committed jump is deliberate — no buzz for it
    // ---- lock layer (v4) ----
    this.mode = "following"; // following | locked | diverged
    this.anchor = -1;        // reported position while locked/diverged (next expected word)
    this.frozenAnchor = -1;
    this.spanWords = 0;      // beam-advanced words in the current lock run
    this.spanFrom = -1;      // beam idx where the current lock run began
    this.prevBeamIdx = 0;
    this.beamGap = 0;
    this.missRun = 0;
    this.winMiss = [];       // last 6 walk outcomes while locked (1 = miss)
    this.resumeRun = 0;
    this.resumeAnchor = -1;
    this.beamRun = 0;        // consecutive feeds the beam tracked consistently (the shadow)
    this.beamRunPos = -1;
    this.escapeMode = "manual"; // manual | suggest | auto — set by the page
    this._suggest = null;
    this._jumped = false;
    this.buzzLatched = false;
    this.pendingRestart = -1;
    this.graceEvents = 0;   // post-network-gap words that may not accuse
    this.lagging = false;   // page's lag monitor: connection is degraded
    this.dropoutFlag = false; // a mid-speech recognizer restart ate words
    this.dropWatchPos = -1; // shadow position under dropout observation
    this.dropWatchRun = 0;  // sequential shadow words seen there
    this.heardSince = []; // folded words heard recently (bounded window)
  }

  // The page calls this when the reciter paused before the incoming
  // words (an utterance gap): a passage that starts after a pause is
  // allowed to move the tracker quickly — it reads as a deliberate
  // restart from somewhere else, not a mid-flow mistake.
  Tracker.prototype.notePause = function () {
    this.pausePending = true;
  };

  // A long gap in RESULTS ARRIVING (the online engine on a slow
  // connection): the words that land next are late, batched, and often
  // garbled interim revisions. They may still move the tracker when
  // they fit, but they must never trigger a divergence or a buzz —
  // the network misbehaving is not the reciter going off-track.
  Tracker.prototype.noteLongGap = function () {
    this.graceEvents = 6;
  };

  // The page's lag monitor (event-level packet signals, AI-Mode-reviewed
  // design v2.1): while the connection is degraded, divergence must be
  // BLATANT to buzz, and proven-flow catch-ups replace accusations.
  Tracker.prototype.noteLag = function (on) {
    this.lagging = !!on;
  };

  // A recognizer restart while speech was active: the audio during the
  // gap was EATEN by the network, not skipped by the reciter. The flag
  // lives until the next lock engagement; it suppresses the buzz for
  // this episode and enables the dropout catch-up in the diverged state.
  Tracker.prototype.noteDropout = function () {
    this.dropoutFlag = true;
    this.lagging = true;
  };

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
      if (m.local >= 0.85) { this.pendingJump = null; this.episodeNudged = false; return false; } // long local fit is good: stay
      if (m.score >= 0.95 && m.score >= m.local + 0.25) cand = { start: m.start, n: n, strong: true, local: m.local };
    }
    if (!cand) {
      var shortNs = [6, 5];
      for (si = 0; si < shortNs.length && !cand; si++) {
        n = shortNs[si];
        m = this.distantMatch(n);
        if (!m) continue;
        if (m.local >= 0.8) { this.pendingJump = null; this.episodeNudged = false; return false; }
        // refrain ambiguity: the same words also fit near where we are
        var seq = H.slice(H.length - n), nearBest = 0;
        for (var st = Math.max(0, this.cursor - n - 8); st <= this.cursor - n + 8; st++) {
          var nsc = this.seqScore(st, seq);
          if (nsc > nearBest) nearBest = nsc;
        }
        if (nearBest >= 0.85) continue; // explained near here — a twin, not a move
        var need = this.state === "locked" ? 0.95 : 0.9;
        if (m.score >= need && m.score >= m.local + 0.35) cand = { start: m.start, n: n, strong: false, local: m.local };
      }
    }
    if (!cand) { this.pendingJump = null; this.episodeNudged = false; return false; }
    var targetEnd = cand.start + cand.n;
    // (4) no cheap ride back to where the last jump came from
    if (!cand.strong && this.lastJumpFrom >= 0 &&
        this.eventsSeen - this.lastJumpAt < 40 &&
        Math.abs(targetEnd - this.lastJumpFrom) <= 25) {
      this.pendingJump = null;
      return false;
    }
    // (5) NUDGE + GATED COMMIT (v3, user direction): a held lock is a
    // commitment. If words arrive that fit a distant place instead of
    // here, the first strong proposal earns ONE gentle nudge (the page
    // vibrates / softly beeps — "this isn't the expected sentence") and
    // the tracker STAYS. The jump itself then needs one of:
    //   - a pause before this passage (deliberate restart), or
    //   - persistent proof: the same move proposed again and again
    //     (he kept reciting the new place — it wasn't a slip).
    // Without a recent lock there is nothing to protect: 2 proposals,
    // as before (cold start / right after Listen is pressed again).
    var protectedRun = this.state === "locked" || (this.eventsSeen - this.wasLockedAt) < 30;
    var gated = this.pausePending;
    this.pausePending = false; // the pause is spent on this evaluation
    var p = this.pendingJump;
    var isNew = !(p && Math.abs(targetEnd - (p.end + (this.eventsSeen - p.at))) <= 2);
    if (isNew) {
      this.pendingJump = p = { end: targetEnd, at: this.eventsSeen, count: 1 };
      if (this.mode === "following" && this.graceEvents <= 0 && protectedRun && !gated && cand.local < 0.5 &&
          !this.episodeNudged && this.eventsSeen - this.nudgeAt >= 15) {
        this._nudge = true;
        this.nudgeAt = this.eventsSeen;
        this.episodeNudged = true;
      }
    } else {
      p.count++; p.end = targetEnd; p.at = this.eventsSeen;
    }
    var needCount = gated ? 2 : (protectedRun ? 7 : 2);
    if (p.count < needCount) return false;
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
    this.episodeNudged = false;
    this.justJumped = true;
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
    this.pausePending = false;
    this.wasLockedAt = -1000000;
    this.episodeNudged = false;
    this.lastRep = -1;
    this.lurchCalm = 99;
    this.mode = "following";
    this.anchor = -1;
    this._suggest = null;
    this.buzzLatched = false;
    this.spanFrom = -1;
    this.spanWords = 0;
    this.winMiss = [];
    this.missRun = 0;
    this.wideEvents = 12;
    return this.snapshot();
  };

  Tracker.prototype.feed = function (rawWord) {
    var inw = foldWord(rawWord);
    if (!inw) return this.snapshot();
    this.eventsSeen++;
    this._nudge = false;
    this._jumped = false;
    this.history.push(inw);
    if (this.history.length > 14) this.history.shift();
    this.heardSince.push(inw);
    if (this.heardSince.length > 60) this.heardSince.shift();
    if (this.checkJump()) {
      this.lockStep(inw);
      var js = this.snapshot();
      this.lastRep = js.wordIdx;
      this.justJumped = false;
      return js;
    }
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
      this.beamGap = gap;
      if (best.matched >= 4 && gap >= 2.0) {
        this.state = "locked";
        this.lockedCursor = this.cursor;
        this.wasLockedAt = this.eventsSeen;
        this.lowConfStreak = 0;
      } else if (best.matched >= 2) {
        this.state = this.state === "locked" && gap < 0.5 ? "lost" : "tentative";
        if (this.state === "lost") this.lowConfStreak++; else this.lowConfStreak = 0;
      } else {
        this.state = "searching";
      }
    }
    this.lockStep(inw);
    return this.snapshot();
  };

  // ---- lock layer (v4) ----
  // Modes: "following" (the beam drives the display), "locked" (the
  // display advances ONLY sequentially from the anchor; the beam keeps
  // running silently as a shadow), "diverged" (words stopped fitting
  // the anchor: display frozen at the last confirmed word, ONE latched
  // buzz, shadow keeps watching). The lock engages only on
  // discriminating evidence — a sustained beam lock whose span
  // contains rare words — never on a refrain alone. Escape from
  // diverged is governed by escapeMode: "manual" (default — only the
  // user's own action moves us), "suggest" (a confident shadow is
  // offered; the user taps to take it), "auto" (a confident shadow
  // moves us silently).
  Tracker.prototype.lineStartOf = function (idx) {
    if (idx < 0) idx = 0;
    if (idx >= this.words.length) idx = this.words.length - 1;
    var ln = this.words[idx].line, s = idx;
    while (s > 0 && this.words[s - 1].line === ln) s--;
    return s;
  };

  Tracker.prototype.spanHasRare = function (from, to) {
    if (!this.freq) {
      this.freq = {};
      for (var fi = 0; fi < this.folded.length; fi++) this.freq[this.folded[fi]] = (this.freq[this.folded[fi]] || 0) + 1;
    }
    for (var i = Math.max(0, from); i < Math.min(to, this.folded.length); i++) {
      if ((this.freq[this.folded[i]] || 99) <= 90) return true; // a word used in <5% of positions
    }
    return false;
  };

  Tracker.prototype.engageLock = function (at) {
    this.mode = "locked";
    this.anchor = at;
    this.divCause = null;
    this.pendingRestart = -1;
    this.missRun = 0;
    this.winMiss = [];
    this.resumeRun = 0;
    this.stagRun = 0;
    this.buzzLatched = false;
    this._suggest = null;
    this.dropoutFlag = false;
    this.dropWatchPos = -1;
    this.dropWatchRun = 0;
    this.heardSince = [];
  };

  // Does the shadow's last-8 window contain a discriminating word that
  // does NOT appear within ±2 lines of the anchor? If not, a "leap"
  // there is a refrain twin, not a move (AI Mode round 2, rule B fix).
  Tracker.prototype.shadowWindowForeignRare = function (beamIdx) {
    if (!this.freq) this.spanHasRare(0, 0);
    var aLine = this.words[Math.max(0, Math.min(this.anchor - 1, this.words.length - 1))].line;
    var near = {};
    for (var i = 0; i < this.words.length; i++) {
      if (Math.abs(this.words[i].line - aLine) <= 2) near[this.folded[i]] = true;
    }
    for (var k = Math.max(0, beamIdx - 8); k < Math.min(beamIdx, this.folded.length); k++) {
      var w = this.folded[k];
      if ((this.freq[w] || 99) <= 90 && !near[w]) return true;
    }
    return false;
  };

  // Ordered fuzzy coverage: how much of the text span [from, to) is
  // accounted for, in order, by the words heard since the anchor last
  // advanced? Greedy in-order matching with the usual word similarity
  // (partial credit) — the "did he SAY those lines?" test.
  Tracker.prototype.spanCoverage = function (from, to) {
    var heard = this.heardSince, hi = 0, credit = 0, disc = false;
    if (!this.freq) this.spanHasRare(0, 0);
    for (var s = from; s < to; s++) {
      var sw = this.folded[s], bestJ = -1, bestSim = 0;
      for (var j = hi; j < Math.min(hi + 6, heard.length); j++) {
        var sm = sim(heard[j], sw);
        if (sm > bestSim) { bestSim = sm; bestJ = j; }
      }
      if (bestJ >= 0 && bestSim >= 0.68) {
        credit += bestSim;
        hi = bestJ + 1;
        if ((this.freq[sw] || 99) <= 90) disc = true;
      }
    }
    return { cov: (to - from) > 0 ? credit / (to - from) : 0, disc: disc };
  };

  // SILENT CATCH-UP (all modes, incl. manual): if the shadow is ahead
  // of the anchor and the heard words PROVE the intervening lines were
  // said — he flowed, the network lagged — glide to the shadow with no
  // buzz and no escape semantics. Compound span rules (AI Mode v2.1):
  // 1 line: coverage >= 0.65, no rare-word demand. 2-3 lines, short
  // span (<8 words): coverage >= 0.75 + a rare word in the span or the
  // landing line's head. 2-3 lines, span >= 8: coverage >= 0.60 + a
  // rare word in the span. Beyond 3 lines / 45 words: not eligible.
  Tracker.prototype.tryCoverageCatchup = function (beamIdx) {
    var from = this.mode === "diverged" ? this.frozenAnchor : this.anchor;
    if (from < 0 || beamIdx <= from) return false;
    var spanLen = beamIdx - from;
    if (spanLen > 45) return false;
    var linesCrossed = this.words[beamIdx - 1].line - this.words[Math.max(0, from - 1)].line;
    if (linesCrossed < 1 || linesCrossed > 3) return false;
    var c = this.spanCoverage(from, beamIdx);
    var ok = false;
    if (linesCrossed === 1) ok = c.cov >= 0.65;
    else if (spanLen < 8) ok = c.cov >= 0.75 && (c.disc || this.spanHasRare(beamIdx, beamIdx + 4));
    else ok = c.cov >= 0.60 && c.disc;
    if (!ok) return false;
    this.engageLock(beamIdx);
    return true;
  };

  Tracker.prototype.lockStep = function (inw) {
    var beamIdx = this.cursor, k;
    var inGrace = this.graceEvents > 0;
    if (this.graceEvents > 0) this.graceEvents--;
    // shadow bookkeeping: how consistently is the beam tracking?
    if (this.state === "locked" || this.state === "tentative") {
      var adv = beamIdx - this.beamRunPos;
      if (this.beamRunPos >= 0 && adv >= 0 && adv <= 3) this.beamRun++;
      else if (beamIdx !== this.beamRunPos) this.beamRun = 0;
    } else if (beamIdx !== this.beamRunPos) {
      this.beamRun = 0;
    }
    this.beamRunPos = beamIdx;

    if (this.mode === "following") {
      if (this.state === "locked") {
        if (this.spanFrom < 0) { this.spanFrom = this.prevBeamIdx; this.spanWords = 0; }
        this.spanWords += Math.max(0, beamIdx - this.prevBeamIdx);
        var quick = beamIdx <= 15 && this.spanWords >= 5;
        if ((this.spanWords >= 10 && this.spanHasRare(this.spanFrom, beamIdx)) || quick) {
          this.engageLock(beamIdx);
        }
      } else if (this.spanWords > 0) {
        // a single flickery feed must not wipe the run — decay instead
        this.spanWords = Math.max(0, this.spanWords - 2);
        if (this.spanWords === 0) this.spanFrom = -1;
      }
      this.prevBeamIdx = beamIdx;
      return;
    }

    if (this.mode === "locked") {
      // a restart landing is only TENTATIVE until the beam confirms
      // it with a sequential advance — a one-event beam wobble onto a
      // line start must not drag the display backward
      if (this.pendingRestart >= 0) {
        var pr = this.pendingRestart;
        this.pendingRestart = -1;
        if (beamIdx > pr && beamIdx <= pr + 3) {
          this.anchor = beamIdx;
          this.stagRun = 0;
          this.prevBeamIdx = beamIdx;
          return;
        }
      }
      // Guarded following: the beam's scoring is the robust matcher
      // (it absorbs splits, merges and sound-alike substitutions that
      // a strict word-by-word walk cannot — measured on the real
      // decode). While locked we follow the beam ONLY through its
      // sequential movement. A LEAP (the beam's chosen position jumps
      // 5+ words forward, or 3+ back without landing on a line start)
      // is a skip/repeat/relocation: refuse it, freeze, buzz once.
      // A stagnating beam whose recent words fit nowhere near the
      // anchor is the other divergence: he is saying something else.
      var delta = beamIdx - this.anchor;
      var lsCur = this.lineStartOf(this.anchor - 1);
      var lsPrev = lsCur > 0 ? this.lineStartOf(lsCur - 1) : -1;
      var restartLanding = Math.abs(beamIdx - (lsCur + 1)) <= 1 || (lsPrev >= 0 && Math.abs(beamIdx - (lsPrev + 1)) <= 1);
      if (delta >= -2 && delta <= 4) {
        // Calm display: while locked the shown position NEVER steps
        // backward for a wobble (recognizer revisions make the beam
        // jitter -1/-2 constantly) and glides forward at most 2 words
        // per heard word, so the highlight walks instead of hopping.
        // A genuine line restart (below) is the only backward move.
        if (beamIdx > this.anchor) this.anchor = Math.min(beamIdx, this.anchor + 2);
        this.stagRun = 0;
      } else if (delta < -2 && restartLanding) {
        this.pendingRestart = beamIdx; // line restart? — confirm next event
        this.stagRun = 0;
      } else if (delta > 4 || delta < -2) {
        // A beam leap is ambiguous: the beam also leaps when it
        // REVIVES near the truth after its local hypotheses died of
        // noise. The words themselves arbitrate: if the last stretch
        // still substantially fits HERE, it was a revival — follow.
        // If it fits THERE (>=0.85) and not here, HE moved.
        // If it fits neither, call it stagnation (healable).
        var H8 = this.history, n8 = Math.min(8, H8.length), seq8 = H8.slice(-n8);
        var fitHere = this.seqScore(this.anchor - n8, seq8);
        var fitThere = this.seqScore(beamIdx - n8, seq8);
        if (fitHere >= 0.55 || inGrace) {
          // revival: follow, but glide — never a visible hop,
          // forward or back. (In a post-gap grace window a leap is
          // presumed to be the late batch landing, not a move by him.)
          this.anchor = beamIdx > this.anchor ? Math.min(beamIdx, this.anchor + 4) : Math.max(beamIdx, this.anchor - 2);
          this.stagRun = 0;
        } else if (this.tryCoverageCatchup(beamIdx)) {
          // the heard words prove the intervening lines were SAID —
          // he flowed, the delivery lagged. Silent catch-up (already
          // re-locked inside), no buzz, no divergence.
          this.prevBeamIdx = beamIdx;
          return;
        } else if (this.dropoutFlag) {
          // A mid-speech restart may have EATEN the intervening
          // words. Don't accuse yet: watch the shadow — if it
          // sustains a sequential run at the new place with a
          // foreign discriminating word in the window, the network
          // ate the gap and we rejoin silently. If the run breaks,
          // the flag stays and the normal verdicts resume below on
          // later leaps once the watch gives up (run reset).
          if (this.dropWatchPos >= 0 && beamIdx >= this.dropWatchPos && beamIdx <= this.dropWatchPos + 3) {
            this.dropWatchRun++;
          } else {
            this.dropWatchRun = 1;
          }
          this.dropWatchPos = beamIdx; // the window slides with the beam
          if (this.dropWatchRun >= 5 && this.shadowWindowForeignRare(beamIdx)) {
            this.engageLock(beamIdx);
          }
          this.stagRun = 0;
          this.prevBeamIdx = beamIdx;
          return;
        } else if (this.lagging) {
          // Blatant-only while the connection is degraded: a leap
          // divergence now needs an extreme fit THERE, near-zero fit
          // HERE, and a discriminating word in the shadow's window
          // that doesn't live near the anchor (refrain-twin veto).
          // Anything less freezes quietly — no buzz on suspicion.
          if (fitThere >= 0.9 && fitHere < 0.35 && this.shadowWindowForeignRare(beamIdx)) {
            this.mode = "diverged";
            this.divCause = "leap";
            this.frozenAnchor = this.anchor;
            this.resumeRun = 0;
            if (!this.buzzLatched && !this.dropoutFlag && this.eventsSeen - this.nudgeAt >= 30) { this._nudge = true; this.nudgeAt = this.eventsSeen; this.buzzLatched = true; }
          }
          this.stagRun = 0;
          this.prevBeamIdx = beamIdx;
          return;
        } else {
          this.mode = "diverged";
          this.divCause = fitThere >= 0.85 ? "leap" : "stagnation";
          this.frozenAnchor = this.anchor;
          this.resumeRun = 0;
          if (!this.buzzLatched && !this.dropoutFlag && this.eventsSeen - this.nudgeAt >= 30) { this._nudge = true; this.nudgeAt = this.eventsSeen; this.buzzLatched = true; }
          this.prevBeamIdx = beamIdx;
          return;
        }
      }
      // stagnation: position not advancing and the last 5 heard words
      // do not fit the text just behind the anchor
      if (beamIdx <= this.anchor) {
        var H = this.history;
        var fit = H.length >= 4 ? this.seqScore(this.anchor - Math.min(5, H.length), H.slice(-5)) : 1;
        if (fit < 0.5 && inGrace) {
          this.stagRun = 0; // late/garbled batch — not evidence against him
        } else if (fit < 0.5) {
          this.stagRun = (this.stagRun || 0) + 1;
          // while lagging, stagnation must be blatant too: a single
          // lag flush can burn 10 stagnant events in a second
          if (this.stagRun >= (this.lagging ? 12 : 5)) {
            this.mode = "diverged";
            this.divCause = "stagnation"; // the EVIDENCE failed (recognizer burst), not necessarily him
            this.frozenAnchor = this.anchor;
            this.resumeRun = 0;
            if (!this.buzzLatched && !this.dropoutFlag && this.eventsSeen - this.nudgeAt >= 30) { this._nudge = true; this.nudgeAt = this.eventsSeen; this.buzzLatched = true; }
          }
        } else {
          this.stagRun = 0;
        }
      } else {
        this.stagRun = 0;
      }
      this.prevBeamIdx = beamIdx;
      return;
    }

    // SILENT CATCH-UP from diverged: if the words heard since prove
    // the intervening lines were actually said (he flowed, delivery
    // lagged), rejoin the shadow quietly — in every escape mode.
    if (this.tryCoverageCatchup(beamIdx)) return;
    // DROPOUT catch-up: a mid-speech restart ate the intervening
    // audio; if the shadow now sustains a strong run at its new
    // position WITH a discriminating word in the matched stretch,
    // rejoin silently instead of treating it as a skip.
    if (this.dropoutFlag && this.beamRun >= 8 &&
        beamIdx > this.frozenAnchor && beamIdx <= this.frozenAnchor + 45 &&
        (this.state === "locked" || this.state === "tentative") &&
        this.shadowWindowForeignRare(beamIdx)) {
      this.engageLock(beamIdx);
      return;
    }
    // diverged: display frozen; the beam (shadow) decides recovery.
    // A STAGNATION divergence heals automatically once the shadow is
    // clearly tracking again nearby — the recognizer burst passed and
    // the beam never truly lost him. A LEAP divergence never heals on
    // its own: he actually went somewhere else, so only a return to
    // the frozen place, his chosen escape mode, or his own tap moves us.
    if (this.divCause === "stagnation" && this.beamRun >= 6 &&
        beamIdx >= this.frozenAnchor - 14 && beamIdx <= this.frozenAnchor + 45 &&
        (this.state === "locked" || this.state === "tentative")) {
      // snap back to the frozen word when the shadow is within a few
      // words of it — the display then doesn't move AT ALL on a heal
      this.engageLock(Math.abs(beamIdx - this.frozenAnchor) <= 3 ? this.frozenAnchor : beamIdx);
      return;
    }
    // a local resume is the beam returning near the frozen anchor and
    // advancing sequentially again (he came back / restarted the line)
    var nearFrozen = beamIdx >= this.frozenAnchor - 14 && beamIdx <= this.frozenAnchor + 6;
    var seqDelta = beamIdx - this.prevBeamIdx;
    this.prevBeamIdx = beamIdx;
    if (nearFrozen && seqDelta >= 0 && seqDelta <= 3 && (this.state === "locked" || this.state === "tentative")) {
      this.resumeRun++;
      if (this.resumeRun >= 3) {
        this.engageLock(Math.abs(beamIdx - this.frozenAnchor) <= 3 ? this.frozenAnchor : beamIdx);
        return;
      }
    } else {
      this.resumeRun = 0;
    }
    var shadowDist = Math.abs(beamIdx - this.frozenAnchor) > 15 &&
      (this.state === "locked" || this.state === "tentative");
    if (shadowDist && this.beamGap >= 2.5 && this.escapeMode === "auto" && this.beamRun >= 12) {
      this.engageLock(beamIdx);
      this._jumped = true;
    } else if (shadowDist && this.beamGap >= 2.5 && this.escapeMode === "suggest" && this.beamRun >= 8) {
      // one suggestion per episode: the shown target is FROZEN (a
      // finger may be moving toward it); accepting still lands on the
      // freshest confident shadow — see acceptSuggestion
      if (!this._suggest) {
        this._suggest = {
          wordIdx: beamIdx,
          line: this.words[Math.min(beamIdx, this.words.length - 1)].line,
          streak: this.beamRun
        };
      }
    } else if (this.beamRun < 4) {
      this._suggest = null;
    }
  };

  Tracker.prototype.setEscapeMode = function (m) {
    if (m === "auto" || m === "suggest" || m === "manual") this.escapeMode = m;
    return this.snapshot();
  };

  Tracker.prototype.acceptSuggestion = function () {
    if (this._suggest) {
      // land on the freshest confident shadow when there is one (it
      // kept tracking while the popup stood still); otherwise the
      // frozen target the user actually saw
      var at = this._suggest.wordIdx;
      if (this.beamRun >= 8 && (this.state === "locked" || this.state === "tentative")) at = this.cursor;
      this._suggest = null;
      this.beamRunPos = at;
      this.engageLock(at);
    }
    return this.snapshot();
  };

  // the "I'm somewhere else" action: land on the shadow if it holds a
  // real track (only meaningful while diverged), else re-open search
  Tracker.prototype.unlockToShadow = function () {
    this._suggest = null;
    this.buzzLatched = false;
    if (this.mode === "diverged" &&
        (this.state === "locked" || this.state === "tentative") && this.beamRun >= 6 &&
        Math.abs(this.cursor - this.anchor) > 4) {
      this.engageLock(this.cursor);
    } else {
      this.mode = "following";
      this.spanFrom = -1; this.spanWords = 0;
    }
    return this.snapshot();
  };

  Tracker.prototype.snapshot = function () {
    var idx = this.mode === "following" ? this.cursor : this.anchor;
    if (idx < 0) idx = this.cursor;
    var cidx = Math.max(0, Math.min(idx, this.words.length - 1));
    return {
      state: this.state,
      mode: this.mode,
      wordIdx: idx,
      line: this.words.length ? this.words[cidx].line : 0,
      lockedWordIdx: this.lockedCursor,
      eventsSeen: this.eventsSeen,
      nudge: this._nudge === true,
      jumped: this._jumped === true,
      suggest: this._suggest
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
    this.wasLockedAt = this.eventsSeen;
    this.pausePending = false;
    this.episodeNudged = false;
    this.lastRep = wordIdx;
    this.lurchCalm = 99;
    this.mode = "locked";
    this.anchor = wordIdx;
    this.frozenAnchor = wordIdx;
    this.missRun = 0;
    this.winMiss = [];
    this.buzzLatched = false;
    this._suggest = null;
    this.beamRunPos = wordIdx;
    this.sinceAdvance = 0;
    return this.snapshot();
  };

  var api = { foldWord: foldWord, sim: sim, Tracker: Tracker };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.CrackerAligner = api;
})(typeof window !== "undefined" ? window : globalThis);
