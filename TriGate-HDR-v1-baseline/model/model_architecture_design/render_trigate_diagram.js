const { chromium } = require("playwright");
const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

async function renderDiagram() {
  const inputPath = path.join(__dirname, "TriGateHDRUnifiedArchitecture.tsx");
  const outputPngPath = path.join(__dirname, "TriGateHDRUnifiedArchitecture.png");

  console.log("Transpiling TSX with esbuild...");
  const source = fs.readFileSync(inputPath, "utf8").replace(
    /import\s+React\s+from\s+["']react["'];?\s*/g,
    ""
  );
  const transformed = await esbuild.transform(source, {
    loader: "tsx",
    format: "iife",
    jsxFactory: "React.createElement",
    jsxFragment: "React.Fragment",
    target: "es2019",
    globalName: "TriGateDiagram",
  });

  console.log("Launching Chromium...");
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 3200, height: 2300 },
    deviceScaleFactor: 3,
  });
  page.on("console", (msg) => console.log("[browser]", msg.text()));
  page.on("pageerror", (err) => console.error("[pageerror]", err.message));

  const htmlContent = `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8" />
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <style>
          html, body {
            margin: 0;
            background: #f8fafc;
          }
          body {
            padding: 0;
          }
        </style>
      </head>
      <body>
        <div id="root"></div>
        <script>${transformed.code}</script>
        <script>
          (function () {
            if (!window.TriGateDiagram || !window.TriGateDiagram.default) {
              throw new Error("TriGateDiagram component did not load.");
            }
            const root = ReactDOM.createRoot(document.getElementById("root"));
            root.render(React.createElement(window.TriGateDiagram.default));
          })();
        </script>
      </body>
    </html>
  `;

  await page.setContent(htmlContent, { waitUntil: "load" });
  await page.waitForSelector("svg", { timeout: 30000 });

  console.log("Capturing PNG...");
  const svg = await page.$("svg");
  await svg.screenshot({ path: outputPngPath, type: "png" });

  await browser.close();
  console.log(`Success: ${outputPngPath}`);
}

renderDiagram().catch((err) => {
  console.error("Render failed:", err);
  process.exit(1);
});

