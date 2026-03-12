from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #f2f8fb;
  --surface: #ffffff;
  --surface-soft: #f7fbfe;
  --text: #0f2c36;
  --muted: #4f6b75;
  --line: #d4e3ea;
  --accent: #0e7aa7;
  --accent-2: #17a2b8;
  --ok: #0f8f5f;
  --warn: #c47c00;
  --error: #b93030;
  --shadow: 0 8px 26px rgba(20, 61, 77, 0.08);
  --radius-xl: 16px;
  --radius-md: 12px;
}

body, .gradio-container {
  background:
    radial-gradient(circle at 10% 10%, rgba(23,162,184,0.10), transparent 40%),
    radial-gradient(circle at 90% 0%, rgba(14,122,167,0.08), transparent 36%),
    var(--bg);
  color: var(--text);
  font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
}

.gradio-container {
  max-width: 1600px !important;
  padding-bottom: 20px;
}

.hero-card {
  margin: 8px 0 14px;
  padding: 16px 18px;
  border-radius: var(--radius-xl);
  border: 1px solid var(--line);
  background: linear-gradient(115deg, #ffffff 0%, #f1f8fc 100%);
  box-shadow: var(--shadow);
  display: grid;
  grid-template-columns: 90px 1fr;
  gap: 14px;
  align-items: center;
}

.hero-logo-wrap {
  width: 82px;
  height: 82px;
  border-radius: 20px;
  background: linear-gradient(145deg, #e6f4fb 0%, #ffffff 100%);
  border: 1px solid #d1e6f0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.hero-logo {
  width: 64px;
  height: 64px;
  object-fit: contain;
}

.hero-content h1 {
  margin: 0;
  font-size: 2rem;
  line-height: 1.1;
  letter-spacing: 0.2px;
}

.hero-subtitle {
  margin: 6px 0 10px;
  color: var(--muted);
  font-size: 1.03rem;
}

.hero-guide {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.hero-guide span {
  background: #eef7fc;
  border: 1px solid #d2e8f3;
  color: #255467;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 0.87rem;
  font-weight: 500;
}

#layout-main {
  display: grid;
  grid-template-columns: minmax(350px, 400px) minmax(640px, 1fr);
  gap: 14px;
  align-items: start;
}

#controls-panel {
  position: sticky;
  top: 10px;
}

.panel-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow);
  padding: 8px 12px;
}

.gr-accordion {
  border: 1px solid #d6e5ec !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden;
  margin: 8px 0 !important;
}

.gr-accordion > .label-wrap {
  background: #f6fbff !important;
  color: #1d4a5c !important;
  font-weight: 600 !important;
}

.gr-accordion .prose, .gr-accordion .wrap {
  font-size: 0.95rem;
}

.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow);
  padding: 12px 14px;
}

.card-title {
  font-weight: 700;
  font-size: 0.98rem;
  color: #174355;
  margin-bottom: 10px;
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(150px, 1fr));
  gap: 8px;
}

.meta-grid > div {
  background: var(--surface-soft);
  border: 1px solid #d7eaf2;
  border-radius: 10px;
  padding: 8px;
  display: flex;
  flex-direction: column;
}

.meta-k {
  color: var(--muted);
  font-size: 0.78rem;
  margin-bottom: 3px;
}

.meta-v {
  font-weight: 600;
  color: #153b4a;
  font-size: 0.92rem;
}

.pred-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  padding: 6px 11px;
  font-size: 0.83rem;
  font-weight: 600;
  border: 1px solid transparent;
}

.chip-strong {
  background: #e7f5ff;
  border-color: #c6e7fb;
  color: #0f5f86;
}

.chip-soft {
  background: #f0f7fa;
  border-color: #d6eaf2;
  color: #25556a;
}

.prob-row {
  display: grid;
  grid-template-columns: 90px 1fr 62px;
  align-items: center;
  gap: 8px;
  margin: 7px 0;
}

.prob-label {
  font-weight: 600;
  font-size: 0.9rem;
  color: #1e4f62;
}

.prob-track {
  height: 10px;
  border-radius: 999px;
  background: #e7f0f4;
  overflow: hidden;
}

.prob-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

.prob-fill.p2 {
  background: linear-gradient(90deg, #68a6bf, #88c4d6);
}

.prob-val {
  text-align: right;
  font-size: 0.84rem;
  color: #2c5b6f;
  font-weight: 600;
}

.logits-line {
  margin-top: 9px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.logits-line span {
  color: var(--muted);
  font-size: 0.78rem;
}

.logits-line code {
  font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.76rem;
  line-height: 1.35;
  padding: 8px;
  border-radius: 9px;
  border: 1px solid #d9e7ee;
  background: #f7fbfe;
  color: #174456;
}

.label-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(180px, 1fr));
  gap: 10px;
}

.label-k {
  color: var(--muted);
  font-size: 0.78rem;
  margin-bottom: 5px;
}

.badge-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.badge {
  background: #f1f8fd;
  border: 1px solid #d5e9f2;
  border-radius: 999px;
  padding: 5px 10px;
  display: inline-flex;
  gap: 6px;
}

.badge-k {
  color: #456977;
  font-size: 0.79rem;
}

.badge-v {
  color: #1f4c5f;
  font-weight: 600;
  font-size: 0.79rem;
}

.banner {
  border-radius: 12px;
  padding: 9px 11px;
  border: 1px solid;
  font-size: 0.9rem;
}

.banner-ok {
  background: #e8f8ef;
  border-color: #b8e9cc;
  color: #156745;
}

.banner-warn {
  background: #fff8e9;
  border-color: #f0d9a4;
  color: #8a5a00;
}

.banner-error {
  background: #fdeeee;
  border-color: #f2c5c5;
  color: #8c2a2a;
}

.banner-sub {
  font-size: 0.79rem;
  opacity: 0.9;
}

.stage-card {
  border-radius: 12px;
  border: 1px solid var(--line);
  background: var(--surface);
  padding: 9px 12px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 2px;
}

.stage-dot {
  width: 9px;
  height: 9px;
  border-radius: 999px;
}

.stage-running .stage-dot { background: var(--warn); box-shadow: 0 0 0 4px rgba(196,124,0,0.15); }
.stage-done .stage-dot { background: var(--ok); box-shadow: 0 0 0 4px rgba(15,143,95,0.15); }
.stage-idle .stage-dot { background: #7a97a3; }
.stage-error .stage-dot { background: var(--error); box-shadow: 0 0 0 4px rgba(185,48,48,0.14); }
.stage-text { font-weight: 600; color: #264f61; }

.legend-gradient {
  height: 13px;
  border-radius: 999px;
  background: linear-gradient(90deg, #2b83f6 0%, #29d1f2 25%, #9ce65f 50%, #f4d53f 75%, #e74c3c 100%);
  border: 1px solid #d4e4ec;
}

.legend-labels {
  margin-top: 6px;
  display: flex;
  justify-content: space-between;
  color: #496977;
  font-size: 0.78rem;
}

.help-list {
  font-size: 0.9rem;
  color: #274d5e;
}

.help-list ul {
  margin: 7px 0 0 0;
  padding-left: 17px;
}

.help-list li {
  margin: 4px 0;
}

.copy-status {
  min-height: 22px;
  font-size: 0.84rem;
  padding: 4px 0;
}

.copy-ok { color: #0f7a52; }
.copy-error { color: #a53333; }

.chip-front { background: #e9f7ef; border-color: #bee6cd; color: #156a43; }
.chip-back { background: #fbeceb; border-color: #f2c5c0; color: #90332e; }
.chip-left { background: #edf2fe; border-color: #cad7fd; color: #2f4fa8; }
.chip-right { background: #fff5e9; border-color: #f2dec0; color: #8f5a1f; }
.chip-ambiguous, .chip-unknown { background: #edf3f7; border-color: #d3e0e7; color: #476878; }

.placeholder {
  color: #56717d;
  font-size: 0.9rem;
}

.app-footer {
  margin-top: 14px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
  color: #4d6974;
  font-size: 0.82rem;
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
}

.app-footer a {
  color: #176184;
  text-decoration: none;
}

.app-footer a:hover {
  text-decoration: underline;
}

@media (max-width: 1100px) {
  #layout-main {
    grid-template-columns: 1fr;
  }
  #controls-panel {
    position: static;
  }
  .hero-card {
    grid-template-columns: 1fr;
    text-align: left;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
    scroll-behavior: auto !important;
  }
}
"""


def path_to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def build_head_html(favicon_path: Path, extra_script: str = "") -> str:
    favicon_data_uri = path_to_data_uri(favicon_path)
    return (
        f'<link rel="icon" type="image/png" href="{favicon_data_uri}" />\n'
        f"{extra_script}\n"
    )
