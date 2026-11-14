/* Cracker prototype service worker — offline after first load.
 * Caches the app shell and, at runtime, the Bani JSON + reference MP3
 * (cross-origin, GitHub Pages serves CORS *), so Listen & Hint keep
 * working with no network. */
var CACHE = "cracker-v18";
var SHELL = ["./", "./index.html", "./aligner.js"];

self.addEventListener("install", function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) { return c.addAll(SHELL); }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener("activate", function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.map(function (k) { if (k !== CACHE) return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
self.addEventListener("fetch", function (e) {
  var url = e.request.url;
  var isData = url.indexOf("anand_sahib.json") !== -1 || url.indexOf("anand_sahib.mp3") !== -1;
  if (isData || url.indexOf(self.location.origin) === 0) {
    e.respondWith(caches.match(e.request).then(function (hit) {
      if (hit) return hit;
      return fetch(e.request).then(function (res) {
        var copy = res.clone();
        caches.open(CACHE).then(function (c) { c.put(e.request, copy); });
        return res;
      });
    }));
  }
});
