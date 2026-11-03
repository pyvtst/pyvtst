// Draws the plotly figures of a page in the theme's mode, and redraws them
// when the reader toggles light and dark mode. Each figure is a
// <div class="pyvtst-figure" data-figure="name"> followed by a JSON
// <script data-figure-spec="name"> holding {"light": spec, "dark": spec}.
(() => {
  const config = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"],
  };
  const specs = new Map();

  const mode = () =>
    document.documentElement.dataset.theme === "dark" ? "dark" : "light";

  function spec(name) {
    if (!specs.has(name)) {
      const node = document.querySelector(`script[data-figure-spec="${name}"]`);
      specs.set(name, JSON.parse(node.textContent));
    }
    return specs.get(name)[mode()];
  }

  function draw() {
    for (const div of document.querySelectorAll(".pyvtst-figure")) {
      const current = mode();
      if (div.dataset.drawn === current) continue;
      const { data, layout } = spec(div.dataset.figure);
      // Keeps the reader's zoom and 3D camera across a theme change.
      layout.uirevision = div.dataset.figure;
      Plotly.react(div, data, layout, config);
      div.dataset.drawn = current;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    draw();
    new MutationObserver(draw).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  });
})();
