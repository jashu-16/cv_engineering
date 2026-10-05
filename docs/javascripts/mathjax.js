window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['ams', 'boldsymbol', 'mathtools', 'color', 'physics', 'textmacros']}
  },
  loader: {
    load: ['[tex]/ams', '[tex]/boldsymbol', '[tex]/mathtools', '[tex]/color', '[tex]/physics']
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    if (MathJax.startup && MathJax.startup.output && MathJax.startup.output.clearCache) {
      MathJax.startup.output.clearCache();
    }
    if (MathJax.typesetClear) {
      MathJax.typesetClear();
    }
    if (MathJax.texReset) {
      MathJax.texReset();
    }
    MathJax.typesetPromise();
  }
});
