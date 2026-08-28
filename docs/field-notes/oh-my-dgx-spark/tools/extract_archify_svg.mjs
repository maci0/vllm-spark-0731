#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const [inputPath, outputPath] = process.argv.slice(2);

if (!inputPath || !outputPath) {
  console.error('usage: extract_archify_svg.mjs <archify.html> <output.svg>');
  process.exit(2);
}

const html = fs.readFileSync(inputPath, 'utf8');
const match = html.match(/<svg\b[\s\S]*?<\/svg>/i);
if (!match) {
  throw new Error(`no SVG found in ${inputPath}`);
}

let svg = match[0];
const viewBox = svg.match(/viewBox="0 0 ([0-9.]+) ([0-9.]+)"/i);
const dimensions = viewBox ? ` width="${viewBox[1]}" height="${viewBox[2]}"` : '';
svg = svg.replace(/^<svg\b([^>]*)>/i, (_opening, attributes) => {
  let next = attributes;
  if (!/\bxmlns=/.test(next)) next += ' xmlns="http://www.w3.org/2000/svg"';
  if (!/\bwidth=/.test(next)) next += dimensions;
  if (!/\bheight=/.test(next)) next += dimensions ? ` height="${viewBox[2]}"` : '';
  return `<svg${next}>`;
});

const style = `
  :root, svg {
    --bg: #020617;
    --grid: #1e293b;
    --text: #ffffff;
    --text-muted: #94a3b8;
    --text-dim: #475569;
    --arrow: #64748b;
    --arrow-emphasis: #34d399;
    --mask: #0f172a;
    --frontend-fill: rgba(8, 51, 68, 0.4);
    --frontend-stroke: #22d3ee;
    --backend-fill: rgba(6, 78, 59, 0.4);
    --backend-stroke: #34d399;
    --database-fill: rgba(76, 29, 149, 0.4);
    --database-stroke: #a78bfa;
    --cloud-stroke: #fbbf24;
    --security-stroke: #fb7185;
    --messagebus-fill: rgba(251, 146, 60, 0.3);
    --messagebus-stroke: #fb923c;
    --external-fill: rgba(30, 41, 59, 0.5);
    --external-stroke: #94a3b8;
  }
  svg { background: var(--bg); font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', ui-monospace, monospace; }
  /* The Archify viewer is interactive and intentionally compact. WikiDocs
     displays these static assets at a narrower reading width, so use a
     readable print scale for the exported SVG without changing the source
     architecture or its validated geometry. */
  svg text[data-node-label] { font-size: 16px !important; }
  svg text[data-detail="context"] { font-size: 12px !important; }
  svg text[data-detail="fine"] { font-size: 10px !important; }
  svg g[data-edge-from] > text { font-size: 11px !important; }
  svg text[data-boundary-label] { font-size: 12px !important; }
  svg g[data-legend] text { font-size: 11px !important; }
  svg g[data-legend] > text { font-size: 13px !important; }
  .archify-background { fill: var(--bg); }
  .c-grid { stroke: var(--grid); fill: none; }
  .c-mask { fill: var(--mask); stroke: none; }
  .c-frontend { fill: var(--frontend-fill); stroke: var(--frontend-stroke); }
  .c-backend { fill: var(--backend-fill); stroke: var(--backend-stroke); }
  .c-database { fill: var(--database-fill); stroke: var(--database-stroke); }
  .c-cloud { fill: rgba(120, 53, 15, 0.3); stroke: var(--cloud-stroke); }
  .c-security { fill: rgba(136, 19, 55, 0.4); stroke: var(--security-stroke); }
  .c-messagebus { fill: var(--messagebus-fill); stroke: var(--messagebus-stroke); }
  .c-external { fill: var(--external-fill); stroke: var(--external-stroke); }
  .c-security-group { fill: transparent; stroke: var(--security-stroke); stroke-dasharray: 4,4; }
  .c-region { fill: rgba(251, 191, 36, 0.05); stroke: var(--cloud-stroke); stroke-dasharray: 8,4; }
  .c-lane { fill: rgba(15, 23, 42, 0.22); stroke: #334155; stroke-dasharray: 6,6; }
  .t-primary { fill: var(--text); }
  .t-muted { fill: var(--text-muted); }
  .t-dim { fill: var(--text-dim); }
  .t-frontend { fill: var(--frontend-stroke); }
  .t-backend { fill: var(--backend-stroke); }
  .t-database { fill: var(--database-stroke); }
  .t-cloud { fill: var(--cloud-stroke); }
  .t-security { fill: var(--security-stroke); }
  .t-messagebus { fill: var(--messagebus-stroke); }
  .t-external { fill: var(--external-stroke); }
  .a-default { stroke: var(--arrow); fill: none; }
  .a-emphasis { stroke: var(--arrow-emphasis); fill: none; }
  .a-security { stroke: var(--security-stroke); fill: none; stroke-dasharray: 5,5; }
  .a-dashed { stroke: var(--database-stroke); fill: none; stroke-dasharray: 4,4; }
  .m-default { fill: var(--arrow); }
  .m-emphasis { fill: var(--arrow-emphasis); }
  .m-security { fill: var(--security-stroke); }
  .m-dashed { fill: var(--database-stroke); }
  svg .semantic-sigil { fill: none; stroke: currentColor; stroke-width: 1.35; stroke-linecap: round; stroke-linejoin: round; opacity: 0.76; }
  svg .semantic-sigil .sigil-fill { fill: currentColor; stroke: none; }
  svg .s-frontend { color: var(--frontend-stroke); }
  svg .s-backend { color: var(--backend-stroke); }
  svg .s-database { color: var(--database-stroke); }
  svg .s-cloud { color: var(--cloud-stroke); }
  svg .s-security { color: var(--security-stroke); }
  svg .s-messagebus { color: var(--messagebus-stroke); }
  svg .s-external { color: var(--external-stroke); }
`;

svg = svg.replace(/(<svg\b[^>]*>)/i, `$1\n  <style><![CDATA[${style}  ]]></style>\n  <rect class="archify-background" width="100%" height="100%"/>`);

// Archify's inline SVG is emitted inside HTML, where valueless data-* attributes
// are legal. A standalone SVG is XML, so those attributes must have values for
// GitHub and WikiDocs image parsers to accept the file.
svg = svg.replace(/\s(data-[a-z0-9-]+)(?=\s|\/?>)/gi, ' $1="true"');

// Keep the readable scale in presentation attributes as well as CSS. Some
// repository image renderers sanitize embedded styles but preserve SVG
// presentation attributes.
const readableFontSizes = { 7: 10, 8: 11, 9: 12, 10: 11, 11: 16, 12: 13 };
svg = svg.replace(/font-size="(7|8|9|10|11|12)"/g, (_match, value) => (
  `font-size="${readableFontSizes[value]}"`
));

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${svg}\n`, 'utf8');
console.log(`extracted ${outputPath}`);
