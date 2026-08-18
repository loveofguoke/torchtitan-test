# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared visual theme for exploratory GLM parity HTML reports."""

PARITY_REPORT_CSS = r"""
:root {
  --pass: #17803d;
  --boundary: #b45309;
  --fail: #b42318;
  --trace: #087e8b;
  --ink: #17202a;
  --muted: #64748b;
  --line: #d8dee9;
  --panel: #f8fafc;
  --header: #eef2f7;
  --link: #155eef;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: #fff;
  font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: 1600px; margin: auto; padding: 24px; }
h1, h2, h3 { scroll-margin-top: 64px; }
h1 { margin-top: 0; }
section { margin: 28px 0; scroll-margin-top: 64px; }
nav.report-toc {
  display: flex;
  gap: 18px;
  align-items: center;
  overflow-x: auto;
  padding: 10px 24px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, .97);
  backdrop-filter: blur(6px);
}
nav.report-toc strong { white-space: nowrap; }
nav.report-toc a { color: var(--link); text-decoration: none; white-space: nowrap; }
nav.report-toc a:hover { text-decoration: underline; }
.hero {
  padding: 22px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
}
.hero p:last-child { margin-bottom: 0; }
.overall, .status-pill {
  display: inline-block;
  padding: 5px 11px;
  border-radius: 999px;
  color: #fff;
  font-weight: 800;
  letter-spacing: .02em;
}
.overall.pass, .status-pill.pass { background: var(--pass); }
.overall.fail, .status-pill.fail { background: var(--fail); }
.status-pill.boundary { background: var(--boundary); }
.status-pill.trace { background: var(--trace); }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-top: 16px;
}
.summary-card {
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-radius: 9px;
  background: #fff;
}
.summary-card b { display: block; margin-top: 2px; font-size: 21px; }
.summary-card span { color: var(--muted); font-size: 12px; text-transform: uppercase; }
.section-summary {
  display: inline-block;
  padding: 4px 9px;
  border-radius: 999px;
  font-weight: 800;
}
.section-summary.pass { color: var(--pass); background: #eaf7ee; }
.section-summary.fail { color: var(--fail); background: #fff0ee; }
.metric-note, .note { color: var(--muted); }
details {
  margin: 12px 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}
summary { cursor: pointer; padding: 9px 11px; font-weight: 700; }
details > :not(summary) { margin-left: 11px; margin-right: 11px; }
details > :last-child { margin-bottom: 11px; }
.topk-help { background: var(--panel); }
.topk-help li { margin: 6px 0; }
.configuration-scroll, .parity-table-scroll {
  max-height: 72vh;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
}
table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  background: #fff;
}
th, td {
  padding: 8px 10px;
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  vertical-align: top;
  text-align: left;
}
th:last-child, td:last-child { border-right: 0; }
tr:last-child td { border-bottom: 0; }
thead th {
  position: sticky;
  top: 0;
  z-index: 10;
  background: var(--header);
  box-shadow: 0 1px 0 var(--line);
}
.configuration { min-width: 720px; }
.parity-table { min-width: 1280px; }
.parity-table td:first-child { white-space: nowrap; font-weight: 650; }
.parity-table tbody tr:hover { background: #f8fbff; }
.column-controls {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 9px 0;
  background: #fff;
}
.column-controls button {
  cursor: pointer;
  padding: 5px 11px;
  border: 1px solid #94a3b8;
  border-radius: 999px;
  background: #e2e8f0;
  color: #1e293b;
  font: inherit;
  font-weight: 650;
}
.column-controls button:hover { border-color: var(--link); }
.column-controls button[aria-pressed="true"] {
  border-color: #1d4ed8;
  background: #2563eb;
  color: #fff;
}
.hide-path .col-path, .hide-dtype .col-dtype,
.hide-value .col-value, .hide-topk .col-topk,
.hide-metric .col-metric, .hide-diagnostic .col-diagnostic { display: none; }
.expandable-cell { min-width: 220px; max-width: 520px; }
.expandable-cell details { margin: 0; background: var(--panel); }
.expandable-cell pre {
  margin: 0 10px 10px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: 12px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace;
}
td.pass { color: var(--pass); font-weight: 800; }
td.boundary { color: var(--boundary); font-weight: 800; }
td.fail { color: var(--fail); font-weight: 800; }
td.trace { color: var(--trace); font-weight: 800; }
.explosion-row { background: #fff7e6; }
.chart-card, .error-chart {
  margin: 18px 0;
  padding: 14px;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
}
.error-chart h3 { margin-top: 0; }
.error-chart svg { width: 100%; min-width: 760px; height: auto; overflow: visible; }
.axis { stroke: #64748b; stroke-width: 1.2; }
.grid { stroke: #e5e7eb; stroke-width: 1; }
.error-chart-legend {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 7px 18px;
  margin-top: 8px;
}
.error-chart-legend-item { display: flex; align-items: center; gap: 8px; min-width: 0; }
.error-chart-swatch { width: 22px; height: 3px; flex: 0 0 22px; }
@media (max-width: 720px) {
  main { padding: 14px; }
  nav.report-toc { padding: 9px 14px; }
  .hero { padding: 16px; }
}
"""
