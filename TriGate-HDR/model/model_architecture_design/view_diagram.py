import webbrowser
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    tsx_file = base / "TriGateHDRUnifiedArchitecture.tsx"
    html_file = base / "diagram_viewer.html"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>TriGate HDR Architecture</title>
  <style>
    body {{ margin: 0; background: #080c16; color: #e2e8f0; font-family: monospace; }}
    .box {{ padding: 24px; }}
    pre {{ white-space: pre-wrap; background: #0b1220; padding: 16px; border-radius: 8px; }}
  </style>
</head>
<body>
  <div class="box">
    <h2>TriGate HDR Architecture Diagram Source</h2>
    <p>Open this TSX in your React environment to render the diagram:</p>
    <pre>{tsx_file}</pre>
  </div>
</body>
</html>"""

    html_file.write_text(html, encoding="utf-8")
    webbrowser.open(html_file.as_uri())
    print(f"Opened viewer: {html_file}")


if __name__ == "__main__":
    main()

