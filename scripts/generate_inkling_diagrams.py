#!/usr/bin/env python3
"""Render code-labelled SVGs from the architecture and exact image fixtures."""
from pathlib import Path
import base64
from html import escape
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/images/inkling2"
OUT.mkdir(parents=True, exist_ok=True)
STYLE = '''<style>
text { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; fill:#1f2d42; text-anchor:middle; dominant-baseline:middle; font-size:25px }
.box { fill:#fff; stroke:#536780; stroke-width:2 }
.accent { fill:#fff4e9; stroke:#b64c10; stroke-width:2.5 }
.cache { fill:#edf6ff; stroke:#28699d; stroke-width:2.5 }
.wire { fill:none; stroke:#536780; stroke-width:2.5; marker-end:url(#arrow) }
.weights { fill:none; stroke:#b64c10; stroke-width:2.5; stroke-dasharray:6 5; marker-end:url(#orange) }
</style><defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#536780"/></marker>
<marker id="orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#b64c10"/></marker></defs>'''


def box(x, y, width, height, label, kind="box", size=25):
    return f'<rect class="{kind}" x="{x}" y="{y}" width="{width}" height="{height}" rx="14"/><text x="{x+width/2}" y="{y+height/2}" style="font-size:{size}px">{escape(label)}</text>'


def path(value, kind="wire"):
    return f'<path class="{kind}" d="{value}"/>'


def save(name, height, title, content):
    value = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 {height}" role="img" aria-labelledby="title"><title id="title">{escape(title)}</title>{STYLE}<rect width="600" height="{height}" rx="20" fill="#f7f9fd"/>{content}</svg>\n'
    (OUT / name).write_text(value)


architecture = ''.join([
    box(30, 30, 240, 70, 'ChatMessage[]'), box(330, 30, 240, 70, 'Inkling2Image'),
    path('M150 100V140'), path('M450 100V140'),
    box(30, 145, 240, 70, 'input_ids'), box(330, 145, 240, 70, 'vision_tower'),
    path('M150 215V245H240V270'), path('M450 215V245H360V270'),
    box(150, 275, 300, 70, 'inputs_embeds'), path('M300 345V390'),
    '<rect x="20" y="395" width="560" height="370" rx="20" fill="#eef1f8" stroke="#536780" stroke-width="2"/>',
    '<text x="300" y="424">Inkling2Model</text>',
    box(50, 460, 250, 65, '0…5'), box(50, 560, 250, 70, '(6…17) × 3', 'accent'),
    box(50, 665, 250, 65, '18…23'), path('M175 525V555'), path('M175 630V660'),
    box(360, 460, 190, 70, 'θ[24]', 'accent'),
    path('M360 495H330V493H305', 'weights'), path('M330 495V595H305', 'weights'),
    path('M330 595V697H305', 'weights'),
    box(360, 660, 190, 70, 'State[48]', 'cache'),
    '<text x="455" y="580" style="font-size:23px">8 / 128</text>',
    path('M300 765V810'), box(125, 815, 350, 70, 'generateStream()'),
    path('M300 885V930'), box(180, 935, 240, 65, 'String')])
save('architecture.svg', 1020, 'Inkling2Model · 24 θ · 48 State · 8/128', architecture)

decision = ''.join([
    box(160, 30, 280, 70, 'request'), path('M300 100V140'),
    box(130, 145, 340, 70, 'Inkling2Chat'), path('M300 215V255'),
    box(180, 260, 240, 70, 'JSON'), path('M300 330V370'),
    box(100, 375, 400, 75, 'parse_proposal()', 'accent'), path('M300 450V490'),
    '<polygon points="300,495 465,575 300,655 135,575" fill="#edf6ff" stroke="#28699d" stroke-width="2.5"/>',
    '<text x="300" y="575">validated</text>',
    path('M135 575H95V725'), path('M465 575H505V725'),
    '<text x="95" y="675">True</text><text x="505" y="675">False</text>',
    box(20, 730, 240, 75, 'action'), box(340, 730, 240, 75, '"clarify"', 'cache')])
save('decision-flow.svg', 830, 'Inkling2Chat · JSON · parse_proposal()', decision)

fixtures = ROOT / 'Tests/Qwen3ChatTests/Fixtures/Inkling2'
cards = []
for index, (name, answer) in enumerate([('quality-count.png', '3'), ('quality-spatial.png', 'Red square'),
                                         ('quality-ocr.png', 'MEET AT 4 PM'), ('quality-chart.png', 'B')]):
    x, y = 20 + (index % 2) * 300, 20 + (index // 2) * 315
    data = (fixtures / name).read_bytes()
    width, height = struct.unpack('>II', data[16:24])
    assert width > 0 and height > 0
    encoded = base64.b64encode(data).decode()
    cards += [f'<rect class="box" x="{x}" y="{y}" width="260" height="285" rx="16"/>',
              f'<image x="{x+12}" y="{y+12}" width="236" height="205" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{encoded}"/>',
              f'<text x="{x+130}" y="{y+251}" style="font-size:25px">{escape(answer)}</text>']
save('examples.svg', 655, '3 · Red square · MEET AT 4 PM · B', ''.join(cards))
print('Wrote architecture, decision-flow and exact-fixture SVGs to', OUT)
