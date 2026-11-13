/* Cracker OFFLINE service worker — cache-first for the shell, the Bani
 * data, the hint track, the ONNX Runtime CDN files and the 138 MB model,
 * so after the first visit everything (including recognition) works
 * with no network at all. */
var CACHE = "cracker-offline-v4";
var SHELL = [
  "./",
  "./index.html",
  "./aligner.js",
  "./mel.js",
  "./tokens.js",
  "./vendor/ort.wasm.min.js",
  "./vendor/ort-wasm-simd-threaded.wasm",
  "./vendor/ort-wasm-simd-threaded.mjs",
  "https://karamofficial.github.io/gurbani-checker/data/anand-sahib/anand_sahib.json",
  "https://karamofficial.github.io/gurbani-checker/data/anand-sahib/anand_sahib.mp3"
];
self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (c) { return c.addAll(SHELL); }).then(function () { return self.skipWaiting(); })
  );
});
self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) { if (k !== CACHE) return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});
function isCacheable(url) {
  return url.indexOf("huggingface.co/surindersinghssj") !== -1 ||
         url.indexOf("gurbani-checker/data/anand-sahib/") !== -1 ||
         url.indexOf("cracker-offline/") !== -1;
}
self.addEventListener("fetch", function (e) {
  var url = e.request.url;
  if (e.request.method !== "GET" || !isCacheable(url)) return;
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then(function (hit) {
      if (hit) return hit;
      return fetch(e.request).then(function (resp) {
        if (resp && (resp.status === 200 || resp.type === "opaque")) {
          var copy = resp.clone();
          caches.open(CACHE).then(function (c) { c.put(e.request, copy); });
        }
        return resp;
      });
    })
  );
});
