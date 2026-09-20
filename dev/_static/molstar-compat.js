/*
 * nbsphinx loads require.js on workshop pages. Mol*'s UMD bundle sees its
 * global `define.amd`, registers as an AMD module, and consequently never
 * defines `window.molstar`, although the notebooks use the browser global.
 *
 * Temporarily hide AMD only while a notebook inserts the pinned Mol* bundle.
 * require.js is restored immediately after it loads, so ordinary notebook
 * output and any other AMD consumers are unaffected.
 */
(function () {
  "use strict";

  var appendChild = HTMLHeadElement.prototype.appendChild;
  var molstarBundle = /cdn\.jsdelivr\.net\/npm\/molstar@[^/]+\/build\/viewer\/molstar\.js(?:[?#].*)?$/;

  HTMLHeadElement.prototype.appendChild = function (node) {
    if (!node || node.tagName !== "SCRIPT" || !molstarBundle.test(node.src)) {
      return appendChild.call(this, node);
    }

    var define = window.define;
    var amd = define && define.amd;
    if (!amd) return appendChild.call(this, node);

    // UMD chooses the browser-global branch while this script is evaluated.
    try {
      delete define.amd;
    } catch (_) {
      define.amd = undefined;
    }

    function restoreAmd() {
      define.amd = amd;
    }
    node.addEventListener("load", restoreAmd, { once: true });
    node.addEventListener("error", restoreAmd, { once: true });
    return appendChild.call(this, node);
  };
})();
