"""Build PROJECT_GUIDE.pdf -- a beginner's walkthrough of the whole project.

This is deliberately separate from PROJECT.md.  PROJECT.md is the formal report
and assumes the reader knows what an eigenvector is; this guide assumes nothing
and builds up from "what is PageRank" to "why our result is what it is",
explaining every symbol, every dataset, every file and every figure.

Run with:   ./.venv/bin/python docs/make_guide.py
"""

from __future__ import annotations

import pathlib

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "PROJECT_GUIDE.pdf"

# --------------------------------------------------------------------------
# Fonts.  ReportLab's built-in fonts have no Greek glyphs, so lambda/beta/mu
# would render as solid black boxes.  DejaVu ships with matplotlib and has the
# full Greek range plus the maths symbols and subscripts this document needs.
# --------------------------------------------------------------------------
TTF = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, fn in [
    ("Body", "DejaVuSans.ttf"),
    ("Body-B", "DejaVuSans-Bold.ttf"),
    ("Body-I", "DejaVuSans-Oblique.ttf"),
    ("Mono", "DejaVuSansMono.ttf"),
    ("Mono-B", "DejaVuSansMono-Bold.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(TTF / fn)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I")

PAGE_W, PAGE_H = A4
MARGIN = 1.7 * cm
TEXT_W = PAGE_W - 2 * MARGIN

NAVY = colors.HexColor("#0B3C5D")
BLUE = colors.HexColor("#0072B2")
ORANGE = colors.HexColor("#D55E00")
GREEN = colors.HexColor("#009E73")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#F4F6F8")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName="Body-B", fontSize=25, leading=30,
                            textColor=NAVY, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=12.5, leading=17,
                               textColor=GREY, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="Body-B", fontSize=17, leading=21,
                         textColor=NAVY, spaceBefore=18, spaceAfter=9),
    "h2": ParagraphStyle("h2", fontName="Body-B", fontSize=12.5, leading=16,
                         textColor=BLUE, spaceBefore=12, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="Body-B", fontSize=10.5, leading=14,
                         textColor=colors.black, spaceBefore=9, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14.2,
                           alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.6, leading=14.2,
                             leftIndent=14, bulletIndent=4, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="Body-I", fontSize=8.3, leading=11.5,
                              textColor=GREY, alignment=TA_CENTER, spaceAfter=10),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10.2,
                           backColor=LIGHT, borderPadding=6, leftIndent=2,
                           spaceBefore=4, spaceAfter=8),
    "note": ParagraphStyle("note", fontName="Body", fontSize=9.2, leading=13.2,
                           leftIndent=9, rightIndent=9, spaceBefore=4, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.2, leading=11),
    "cellb": ParagraphStyle("cellb", fontName="Body-B", fontSize=8.2, leading=11),
    "cellm": ParagraphStyle("cellm", fontName="Mono", fontSize=7.8, leading=11),
}


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# Content helpers
# --------------------------------------------------------------------------
story: list = []


def h1(t):
    story.append(PageBreak())
    story.append(Paragraph(t, S["h1"]))


def h1_same(t):
    story.append(Paragraph(t, S["h1"]))


def h2(t):
    story.append(Paragraph(t, S["h2"]))


def h3(t):
    story.append(Paragraph(t, S["h3"]))


def p(t):
    story.append(Paragraph(t, S["body"]))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="•"))
    story.append(Spacer(1, 5))


def code(t, title=None):
    if title:
        story.append(Paragraph(f'<font face="Mono-B" size="8">{esc(title)}</font>',
                               S["h3"]))
    story.append(Preformatted(t.rstrip("\n"), S["code"]))


def note(t, colour=BLUE, label="NOTE"):
    tbl = Table([[Paragraph(f'<font face="Body-B" color="{colour.hexval()}">{label}</font>'
                            f'<br/>{t}', S["note"])]],
                colWidths=[TEXT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, colour),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 7))


def table(rows, widths=None, header=True, mono_cols=(), align_right=()):
    data = []
    for ri, row in enumerate(rows):
        out = []
        for ci, cell in enumerate(row):
            st = S["cellb"] if (header and ri == 0) else (
                S["cellm"] if ci in mono_cols else S["cell"])
            out.append(Paragraph(str(cell), st))
        data.append(out)
    if widths is None:
        widths = [TEXT_W / len(rows[0])] * len(rows[0])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3EAF0")),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.9, NAVY)]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 9))


def figure(path, caption, width=None, zoom_hint=False):
    f = ROOT / path
    if not f.exists():
        p(f"<i>[missing figure: {esc(path)}]</i>")
        return
    from PIL import Image as PILImage
    with PILImage.open(f) as im:
        iw, ih = im.size
    w = width or TEXT_W
    h = w * ih / iw
    cap = caption
    if zoom_hint:
        cap += (f'<br/><font size="7.4">Open <font face="Mono">'
                f'{esc(path.replace(".png", ".pdf"))}</font> to zoom in.</font>')
    story.append(KeepTogether([Image(str(f), width=w, height=h),
                               Spacer(1, 3),
                               Paragraph(cap, S["caption"])]))


# --------------------------------------------------------------------------
# Page furniture
# --------------------------------------------------------------------------
def decorate(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(MARGIN, 1.05 * cm,
                          "Accelerating PageRank with Dynamic Momentum  ·  "
                          "a beginner's guide")
        canvas.drawRightString(PAGE_W - MARGIN, 1.05 * cm, str(doc.page))
        canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.4 * cm, PAGE_W - MARGIN, 1.4 * cm)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=1.9 * cm,
                          title="Accelerating PageRank with Dynamic Momentum "
                                "- a beginner's guide",
                          author="Dip Saha")
    frame = Frame(MARGIN, 1.9 * cm, TEXT_W, PAGE_H - MARGIN - 1.9 * cm, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(story)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")


# ==========================================================================
# COVER
# ==========================================================================
story.append(Spacer(1, 3.2 * cm))
story.append(Paragraph("Accelerating PageRank with<br/>Dynamic Momentum Power Iteration", S["title"]))
story.append(Spacer(1, 0.5 * cm))
story.append(Paragraph("A beginner's guide to the whole project:<br/>"
                       "the idea, the code, the data, and every figure explained",
                       S["subtitle"]))
story.append(Spacer(1, 1.4 * cm))
figure("figures/explain/explain_tiny_web.png", "", width=TEXT_W)
story.append(Spacer(1, 0.7 * cm))
table([
    ["Course", "CSE 402 — Numerical Analysis Sessional, Section C"],
    ["Base paper", "Austin, Pollock &amp; Zhu (2024), <i>Dynamically accelerating the "
                   "power iteration with momentum</i>, Numer. Linear Algebra Appl. 31(6), e2584"],
    ["Group", "Nakib Arman (2105128) · Ali Asif Khan (2105131) · Shariar Al Kabir (2105132) · "
              "Dip Saha (2105138) · MD. Mehedi Hasan Mim (2105142)"],
    ["Presenter", "Dip Saha"],
    ["Code", "7 library modules · 7 experiments · 19 unit tests · <font face='Mono'>make all</font>"],
], widths=[3.2 * cm, TEXT_W - 3.2 * cm], header=False)

# ==========================================================================
h1("How to read this guide")
p("This guide is written for someone who has <b>not</b> yet understood the project. It starts "
  "from “what is PageRank” and ends at “why our answer came out the way it did”. "
  "Nothing is assumed except comfort with matrices and vectors.")
p("It is deliberately different from <font face='Mono'>PROJECT.md</font>, which is the formal "
  "report and assumes you already know the theory. If you only have time for one, read this "
  "one first.")
table([
    ["Part", "What it answers"],
    ["1. The problem", "What is PageRank, what is an eigenvector here, why is it slow?"],
    ["2. The idea", "What is momentum, what does β do, how does the paper guess β?"],
    ["3. Symbols", "Every letter in the code and the report, in one dictionary."],
    ["4. The data", "Which graphs, where from, what the raw files look like."],
    ["5. The code", "Every file, what it does, with real excerpts."],
    ["6. The figures", "All 13 figures, what is plotted and what to look for."],
    ["7. The result", "What we found and, in plain words, why."],
    ["8. Running it", "Commands, and what to say when someone asks."],
], widths=[3.6 * cm, TEXT_W - 3.6 * cm])
note("There are <b>two different things called <i>d</i></b> in this project, and mixing them up "
     "is the single most common confusion. <b>d</b> (no subscript) is the <i>damping factor</i>, "
     "usually 0.85. <b>d<sub>k</sub></b> (with a subscript) is the <i>residual</i> at step k, a "
     "measure of how wrong the current answer still is. Part 3 lists both.",
     ORANGE, "READ THIS FIRST")

# ==========================================================================
h1("Part 1 — The problem, from zero")

h2("1.1  What PageRank actually computes")
p("Imagine a tiny web of four pages that link to each other. We want a single number per page "
  "saying how important it is. PageRank's rule is circular on purpose: "
  "<b>a page is important if important pages link to it</b>, and a page splits its own "
  "importance equally among the pages it links to.")
figure("figures/explain/explain_tiny_web.png",
       "<b>Figure A.</b> Left: four pages and their links. Middle: the same thing written as a "
       "matrix P, where column j says how page j splits its vote. Right: the resulting scores. "
       "Notice A wins even though C has more incoming links — because C's only out-link hands "
       "<i>all</i> of C's score to A, while A splits its score two ways.")
p("That last point is the whole idea of PageRank and worth repeating: it is not a popularity "
  "count. A single link from a page that links nowhere else is worth more than many links from "
  "pages that link everywhere.")

h3("The matrix P")
p("Column j of P describes page j's outgoing links. If page j has 3 out-links, each gets "
  "1/3. So <b>every column adds up to 1</b> — this is called <i>column-stochastic</i>. "
  "In the figure, column A is (0, 0.5, 0.5, 0) because A links to B and C.")

h3("Dangling nodes")
p("Page D links nowhere, so its column is all zeros and it would leak importance out of the "
  "system. Such pages are called <b>dangling nodes</b>. The fix is to pretend a dangling page "
  "links to <i>every</i> page equally. Real data is full of these: our citation graph has 2,393 "
  "of them (6.9% of all papers).")

h3("The damping factor d")
p("There is one more ingredient. A surfer who only ever follows links can get stuck in a loop. "
  "So with probability <b>d</b> (typically 0.85) the surfer follows a link, and with probability "
  "<b>1 − d</b> they jump to a completely random page. That jump is called "
  "<i>teleportation</i>, and it guarantees a unique answer exists.")
p("Putting the three pieces together gives the <b>Google matrix</b>:")
code("G  =  d · ( P + (1/n) e aᵀ )  +  ((1 − d)/n) e eᵀ\n"
     "        ↑                ↑                  ↑\n"
     "     follow a link   dangling-node fix   teleport to a random page")
p("Here <b>e</b> is a column of all ones, <b>a</b> marks which pages are dangling, and <b>n</b> "
  "is the number of pages. The PageRank scores are the vector <b>x</b> satisfying "
  "<b>G x = x</b>.")

h2("1.2  Why this is an eigenvector problem")
p("<b>G x = x</b> says: applying G to x gives back x itself. In linear algebra language, x is an "
  "<i>eigenvector</i> of G with <i>eigenvalue</i> 1. Because G is column-stochastic, 1 is always "
  "its largest eigenvalue, and the matching eigenvector is exactly the PageRank vector. "
  "So <b>computing PageRank = finding the dominant eigenvector of G</b>.")
note("This is why the base paper is relevant at all. The paper is about finding dominant "
     "eigenvectors faster. PageRank <i>is</i> a dominant-eigenvector problem, on a gigantic "
     "sparse matrix. That is the “cross-domain application” in our title.", BLUE)

h2("1.3  Power iteration: just multiply, over and over")
p("The oldest method for this is embarrassingly simple. Start with any guess x₀. Multiply by "
  "G. Normalise so it does not blow up. Repeat.")
code("x  =  starting guess\n"
     "repeat:\n"
     "    x  =  G x            # one matrix-vector product\n"
     "    x  =  x / ||x||      # keep it a sensible size\n"
     "until x stops changing")
p("Why does this work? Write your starting guess as a mixture of all the eigenvectors of G. "
  "Each multiplication by G multiplies the i-th ingredient by its eigenvalue λ<sub>i</sub>. "
  "The largest eigenvalue is λ₁ = 1, so after k steps the ingredient for "
  "λ<sub>2</sub> has shrunk by (λ<sub>2</sub>/λ<sub>1</sub>)<sup>k</sup> relative "
  "to the winner, the one for λ<sub>3</sub> by even more, and so on. Everything except the "
  "dominant eigenvector dies away.")

h2("1.4  Why it is slow: the spectral gap")
p("The speed is decided by a single number, the ratio of the two largest eigenvalues:")
code("r  =  | λ₂ / λ₁ |            ← called the spectral ratio, or the gap")
p("Each iteration kills only a fraction r of the remaining error. If r = 0.5 the error halves "
  "every step and you are done in 30 steps. If r = 0.99 you need about 2,000 steps for the same "
  "accuracy. <b>The closer r is to 1, the slower everything gets.</b>")
figure("figures/explain/explain_spectral_gap.png",
       "<b>Figure B.</b> Left: the leftover error after k steps is r<sup>k</sup>. Right: how many "
       "steps are needed to reach 10<sup>−8</sup>. The cost explodes as r approaches 1.")
note("For the Google matrix there is a theorem (Haveliwala &amp; Kamvar, 2003) that "
     "|λ₂| ≤ <b>d</b>, the damping factor. So choosing d = 0.99 to get a more "
     "accurate ranking directly makes the computation about 14× slower. That tension is "
     "exactly what our project set out to fix.", GREEN, "WHY WE CARE")


# ==========================================================================
h1("Part 2 — The idea: momentum")

h2("2.1  The intuition")
p("Plain power iteration takes one step, looks around, takes another step. It has no memory. "
  "If ten steps in a row all pushed in roughly the same direction, it still takes the eleventh "
  "step as timidly as the first.")
p("<b>Momentum</b> gives it memory. Think of pushing a child on a swing: you do not fight the "
  "swing, you add a small push <i>in the direction it is already going</i>. The method does the "
  "same by subtracting a multiple of the <i>previous</i> iterate:")
code("plain:      uₖ₊₁  =  G xₖ\n"
     "momentum:   uₖ₊₁  =  G xₖ  −  (β / hₖ) xₖ₋₁\n"
     "                          ↑\n"
     "                    the memory term")
p("β (beta) controls how much memory. β = 0 gives back plain power iteration. Too much "
  "β and the method becomes unstable. The whole game is choosing β well.")
note("The extra term costs a subtraction and a scaling — no extra matrix-vector product. "
     "That matters: on our biggest graph one matrix-vector product touches 2.3 million edges, "
     "whereas the momentum term touches 281,903 numbers. This is why we count "
     "<b>matrix-vector products</b>, not seconds, when comparing methods.", BLUE)

h2("2.2  The trick that makes it analysable: the augmented matrix")
p("Here is the clever step in the theory. The momentum iteration looks like it depends on two "
  "previous vectors, which is awkward. But if you stack the current and previous iterate into "
  "one long vector, it becomes an ordinary <i>plain</i> power iteration on a bigger matrix:")
code("Aβ  =  ⎡  A   −βI ⎤          applied to   ⎡ xₖ   ⎤\n"
     "        ⎣  I     0  ⎦                       ⎣ xₖ₋₁ ⎦")
p("So momentum is not a new kind of algorithm at all — it is the same old power iteration on a "
  "cleverly chosen larger matrix. And the speed of a power iteration is decided by the "
  "eigenvalues of the matrix it runs on. Each eigenvalue λ of G becomes <b>two</b> "
  "eigenvalues of Aβ:")
code("μ±  =  ( λ  ±  √(λ² − 4β) ) / 2        ← equation (2.7) in the paper")
note("<b>Everything in this project follows from that one formula.</b> If you remember one line "
     "from this guide, make it this one. Our result, the failure, and the fix are all just "
     "consequences of what μ does as λ and β change.", NAVY, "THE KEY FORMULA")

h2("2.3  What β does — the “flattening” picture")
p("Multiply the two roots together and you always get β (this is Vieta's formula: for "
  "μ² − λμ + β = 0 the product of the roots is the constant term). So "
  "<b>|μ₊| × |μ₋| = β, always.</b> The only question is how the two "
  "split.")
p("If λ is <b>real</b> and small enough that λ² &lt; 4β, the square root is "
  "imaginary, the two roots are complex conjugates, and conjugates have <i>equal</i> size. Their "
  "product is β, so each must be exactly √β. They cannot split.")
figure("figures/explain/explain_momentum_fold.png",
       "<b>Figure C.</b> Left: with momentum, every real eigenvalue below a threshold gets "
       "squashed onto one flat floor at √β, while λ₁ = 1 stays high. That gap "
       "is the acceleration. Right: an <i>imaginary</i> eigenvalue is not squashed — it climbs, "
       "and past the dotted line it beats λ₁ and the method converges to the wrong "
       "answer.")
p("The left panel is the acceleration in one picture. Without momentum a rival mode at "
  "λ = 0.85 survives at strength 0.85 and only barely dies away. With momentum it is "
  "crushed to 0.43, while the winner stays at 0.76. The bigger that gap, the faster you finish.")
p("Choosing β = λ₂²/4 puts the flat floor exactly at the highest rival, which "
  "is the best you can do. That is the paper's <b>optimal β</b>. The resulting speed is")
code("ρ(r)  =  r / ( 1 + √(1 − r²) )        versus  r  without momentum")
table([
    ["r = |λ₂/λ₁|", "ρ(r) with momentum", "predicted speed-up"],
    ["0.85", "0.5567", "3.60×"],
    ["0.90", "0.6268", "4.43×"],
    ["0.95", "0.7239", "6.30×"],
    ["0.99", "0.8676", "14.13×"],
    ["0.999", "0.9562", "44.72×"],
], widths=[5 * cm, 5.5 * cm, TEXT_W - 10.5 * cm])
p("The worse the problem, the more momentum helps. That is exactly why PageRank looked like a "
  "perfect target.")

h2("2.4  The paper's contribution: guessing β without knowing λ₂")
p("There is a catch. β = λ₂²/4 needs λ₂, and finding λ₂ is "
  "about as hard as the original problem. Earlier methods either assumed you knew it or spent "
  "extra matrix-vector products estimating it.")
p("The paper's idea is to <b>watch how fast the method is actually converging and work backwards</b>. "
  "If the residual shrinks by a factor ρ each step, and theory says the best possible "
  "shrink factor is ρ = r/(1+√(1−r²)), then you can invert that relation to "
  "recover r:")
code("ρₖ  =  min( dₖ₊₁ / dₖ , 1 )        ← the observed shrink factor\n"
     "rₖ₊₁ =  2ρₖ / (1 + ρₖ²)          ← invert the formula to get r\n"
     "βₖ   =  νₖ² · rₖ² / 4              ← and hence β")
p("This costs nothing: dₖ and νₖ are already computed to check convergence. "
  "That is <b>Algorithm 3.1</b>, the paper's method, and it is genuinely elegant — it tunes "
  "itself from information it was collecting anyway.")
note("Hold on to this, because it is where the project turns. The self-tuning rule assumes the "
     "residual behaves the way a <i>real</i> spectrum makes it behave. When we apply it to "
     "PageRank, that assumption breaks, and the rule turns from an asset into the cause of "
     "failure. Part 7 tells that story.", ORANGE, "FORESHADOWING")

# ==========================================================================
h1("Part 3 — Every symbol, in one place")
p("This is the dictionary to keep open while reading the code or the report. The names in the "
  "code deliberately match the paper's notation so the two can be read side by side.")

h2("3.1  The iteration variables")
table([
    ["Symbol", "In code", "What it is"],
    ["x<sub>k</sub>", "x", "The current guess at the answer, scaled to length 1. After "
                           "convergence this <i>is</i> the PageRank vector."],
    ["x<sub>k−1</sub>", "x_prev", "The previous guess. Only momentum needs this — it is "
                                      "the “memory”."],
    ["u<sub>k+1</sub>", "u", "The new vector before scaling, i.e. G x<sub>k</sub> minus the "
                             "momentum term."],
    ["v<sub>k+1</sub>", "v_next", "The result of the matrix-vector product G x<sub>k</sub>. "
                                  "Computed once per iteration — this is the expensive line."],
    ["h<sub>k</sub>", "h", "The length of u<sub>k</sub>, used to rescale. Prevents numbers "
                           "growing or shrinking out of range."],
    ["ν<sub>k</sub>", "nu", "The <b>Rayleigh quotient</b> ⟨G x, x⟩ — the current "
                                 "estimate of the eigenvalue λ₁. For PageRank it "
                                 "should approach 1."],
    ["d<sub>k</sub>", "d", "The <b>residual</b>, ||G x − ν x||. How wrong we still are. "
                           "We stop when this drops below 10<sup>−12</sup>."],
    ["ρ<sub>k</sub>", "rho", "The observed shrink factor d<sub>k+1</sub>/d<sub>k</sub>. "
                                  "How fast the residual is actually falling."],
    ["r<sub>k</sub>", "r", "The <i>estimate</i> of the true spectral ratio, obtained by "
                           "inverting the rate formula from ρ<sub>k</sub>."],
    ["β<sub>k</sub>", "beta", "The momentum parameter. 0 = no momentum. Too large = "
                                   "instability."],
], widths=[2.3 * cm, 2.1 * cm, TEXT_W - 4.4 * cm])
note("<b>The two d's.</b> <b>d</b> without a subscript is the damping factor (0.85). "
     "<b>d<sub>k</sub></b> with a subscript is the residual at step k. They are unrelated. The "
     "paper uses both and so do we, because changing the notation would make the code harder to "
     "check against the paper.", ORANGE, "COMMON CONFUSION")

h2("3.2  The matrix and spectral quantities")
table([
    ["Symbol", "In code", "What it is"],
    ["G", "GoogleMatrix", "The Google matrix. Never built explicitly — it is dense and would "
                          "need 10<sup>11</sup> numbers for our biggest graph."],
    ["P", "G.P", "The sparse link matrix: P[i,j] = 1/outdeg(j) if page j links to page i."],
    ["d", "G.d", "The damping factor, 0.85 by default."],
    ["n", "G.n", "Number of pages/nodes."],
    ["a", "G.dangling", "Boolean mask: true for pages with no out-links."],
    ["e", "—", "A column of all ones. Never stored; it appears only in formulas."],
    ["λ₁", "—", "Largest eigenvalue of G. Always exactly <b>1</b> for a Google matrix."],
    ["λ₂", "lambda2_measured", "Second largest. Bounded by d; we measure whether the "
                                        "bound is tight."],
    ["r", "r", "The spectral ratio |λ₂/λ₁|. For PageRank this equals d when "
               "the bound is tight."],
    ["μ±", "mu_magnitude()", "Eigenvalues of the augmented matrix, from equation (2.7). "
                                       "These decide whether momentum helps or destroys."],
    ["Aβ", "—", "The augmented matrix. An analysis device only; never built in code."],
], widths=[2.3 * cm, 2.9 * cm, TEXT_W - 5.2 * cm])

h2("3.3  The five methods we compare")
table([
    ["Name in code", "What it is"],
    ["<font face='Mono'>power</font>", "Algorithm 1.1 — plain power iteration. The baseline "
                                       "everything is measured against."],
    ["<font face='Mono'>static_d</font>", "Algorithm 1.2 with β = d²/4 — the paper's "
                                          "optimal fixed β, which for PageRank we get for "
                                          "free because λ₁ = 1 and λ₂ ≤ d."],
    ["<font face='Mono'>dynamic_paper</font>", "Algorithm 3.1 exactly as printed in the paper."],
    ["<font face='Mono'>dynamic_safe</font>", "Our extension: Algorithm 3.1 plus the safeguards "
                                              "of Part 7."],
    ["<font face='Mono'>dynamic_guard</font>", "The same plus an extra-cautious rule that "
                                               "retreats to plain power iteration as soon as "
                                               "momentum stops paying."],
], widths=[3.8 * cm, TEXT_W - 3.8 * cm])


# ==========================================================================
h1("Part 4 — The data")

h2("4.1  Where it comes from")
p("Two sources, for two different jobs.")
h3("SNAP — Stanford Large Network Dataset Collection")
p("Real graphs, downloaded from <font face='Mono'>snap.stanford.edu/data</font>. These are the "
  "PageRank problems. They are cached in <font face='Mono'>data/</font> on first use, so you "
  "only download once.")
h3("SuiteSparse Matrix Collection")
p("The abstract benchmark matrices the base paper itself used, fetched via the "
  "<font face='Mono'>ssgetpy</font> package. We use these to <i>reproduce the paper</i> before "
  "trusting our code on anything new.")

h2("4.2  Every dataset we use")
table([
    ["Dataset", "Nodes", "Edges", "Kind", "What we use it for"],
    ["<font face='Mono'>wiki-Vote</font>", "7,115", "103,689", "social",
     "Small enough to debug on, and the one graph where |λ₂| is far below d — a "
     "useful counter-example."],
    ["<font face='Mono'>cit-HepPh</font>", "34,546", "421,578", "citation",
     "<b>Main citation case.</b> arXiv high-energy-physics papers citing each other. This is the "
     "dataset named in our proposal."],
    ["<font face='Mono'>web-Stanford</font>", "281,903", "2,312,497", "web",
     "<b>Main web case.</b> Pages on stanford.edu and the links between them."],
    ["<font face='Mono'>ca-HepPh</font>", "12,008", "118,521", "collaboration",
     "<b>Symmetric control.</b> Undirected co-authorship: two authors are joined if they wrote a "
     "paper together."],
    ["<font face='Mono'>cit-HepTh</font>", "27,770", "352,807", "citation", "Available, unused."],
    ["<font face='Mono'>web-NotreDame</font>", "325,729", "1,497,134", "web", "Available, unused."],
    ["<font face='Mono'>web-Google</font>", "875,713", "5,105,039", "web",
     "Supported; the practical ceiling for a laptop."],
    ["<font face='Mono'>Kuu, Muu</font>", "7,102", "—", "SuiteSparse",
     "Paper's test suite 1. Both symmetric positive definite, r ≈ 0.998."],
    ["<font face='Mono'>ash292, bcspwr06</font>", "292 / 1,454", "—", "SuiteSparse",
     "Paper's test suite 2. Symmetric but indefinite."],
], widths=[3.4 * cm, 1.7 * cm, 1.9 * cm, 2.0 * cm, TEXT_W - 9.0 * cm])

h2("4.3  What the raw file actually looks like")
p("A SNAP file is a gzipped text file of number pairs, with a comment header. Nothing exotic:")
code("# Directed graph: cit-HepPh.txt\n"
     "# Paper citation network of Arxiv High Energy Physics category\n"
     "# Nodes: 34546 Edges: 421578\n"
     "# FromNodeId    ToNodeId\n"
     "9907233   9301253\n"
     "9907233   9504304\n"
     "9907233   9505235\n"
     "...", "data/cit-HepPh.txt.gz  (decompressed)")
p("Each line means “paper 9907233 cites paper 9301253”. Two things make this harder "
  "than it looks:")
bullets([
    "<b>The IDs are not 0,1,2,…</b> They are arXiv identifiers with big gaps. We renumber "
    "them to 0..n−1 with <font face='Mono'>np.unique(..., return_inverse=True)</font> and "
    "keep the original IDs so results can be reported in the source's own numbering.",
    "<b>The dumps contain self-loops and duplicate edges.</b> Both are artefacts; a paper citing "
    "itself is not a real link. We drop both.",
])

h2("4.4  From a file to a ranking")
figure("figures/explain/explain_pipeline.png",
       "<b>Figure D.</b> The five stages every experiment goes through. Only the fourth stage "
       "differs between the methods we compare — everything else is shared, which is what "
       "makes the comparison fair.")

h2("4.5  Why we never build G")
p("For <font face='Mono'>web-Stanford</font>, G is 281,903 × 281,903. Storing it densely "
  "would need about 636 <b>gigabytes</b>. But G is mostly the sparse matrix P plus two rank-one "
  "corrections, so we apply it as a formula instead:")
code("y = d·(P @ x)                      # sparse, ~2.3M multiply-adds\n"
     "  + (d/n)·(a·x)·e                   # dangling-node fix, one dot product\n"
     "  + ((1−d)/n)·(e·x)·e               # teleportation,     one dot product")
p("That is one sparse matrix-vector product plus two dot products, and it runs in about 3 "
  "milliseconds. We count it as <b>one matrix-vector product</b>.")
note("Most PageRank code simplifies the last two terms to a single constant, because plain power "
     "iteration keeps the entries of x summing to 1. <b>Momentum breaks that.</b> Its iterates "
     "are scaled to length 1 in the 2-norm and can contain <i>negative</i> entries, so the sum "
     "is neither 1 nor even positive. Using the shortcut gives a silently wrong operator and a "
     "silently wrong ranking. This cost us real debugging time and is now locked down by a unit "
     "test that checks the identity on negative and zero vectors.", ORANGE, "TRAP 1")

# ==========================================================================
h1("Part 5 — The code, file by file")
p("About 3,600 lines, laid out so that each file has one job.")
code("src/                              the library\n"
     "  iterations.py       (698)   the three algorithms + our safeguards\n"
     "  google_matrix.py    (236)   the Google matrix as a formula\n"
     "  datasets.py         (241)   downloading and caching graphs\n"
     "  spectrum.py         (168)   the μ(λ) analysis of Part 7\n"
     "  metrics.py          (115)   ranking quality: precision@k, Kendall tau\n"
     "  pagerank_methods.py  (76)   the five configurations, defined once\n"
     "  plotting.py          (66)   one shared figure style\n"
     "\n"
     "experiments/                      one file per phase, each runnable alone\n"
     "  exp1_reproduce.py    reproduce the paper          (a PASS/FAIL gate)\n"
     "  exp2_pagerank.py     validate G, then benchmark   (a PASS/FAIL gate)\n"
     "  exp3_damping.py      sweep the damping factor d\n"
     "  exp4_ranking.py      ranking quality vs residual\n"
     "  exp5_spectrum.py     why β = d²/4 fails\n"
     "  exp6_symmetric.py    the symmetric control\n"
     "  exp7_normality.py    the single-variable proof\n"
     "\n"
     "tests/                 19 unit tests\n"
     "results/               CSV output + run.log\n"
     "figures/               PDF for LaTeX, PNG for slides")

h2("5.1  src/iterations.py — the three algorithms")
p("The most important file, and the one design decision worth understanding: "
  "<b>this file knows nothing about PageRank</b>. Every function takes an arbitrary linear "
  "operator and only ever computes <font face='Mono'>A @ x</font>.")
p("That is why the same code runs on the paper's SuiteSparse matrices and on our Google matrix "
  "with no changes — and it is why, when the PageRank results came out negative, we could "
  "<i>prove</i> the solver was correct rather than argue about it.")
code("def power_iteration(A, v0=None, tol=1e-12, max_iter=2000, ...):\n"
     "    matvec = _as_matvec(A)\n"
     "    ...\n"
     "    for k in range(1, max_iter + 1):\n"
     "        h = np.linalg.norm(v_next)\n"
     "        x = v_next / h\n"
     "        v_next = matvec(x)                    # the single matvec\n"
     "        matvecs += 1\n"
     "        nu = float(v_next @ x)                # Rayleigh quotient\n"
     "        d  = float(np.linalg.norm(v_next - nu * x))   # residual\n"
     "        if d < tol:\n"
     "            converged = True\n"
     "            break", "Algorithm 1.1, the whole loop")
p("The momentum version adds exactly one line before the rescale, and the dynamic version adds "
  "three more to update β:")
code("# Algorithm 1.2 - static momentum\n"
     "u = v_next - (beta / h) * x_prev\n"
     "\n"
     "# Algorithm 3.1 - dynamic momentum: beta is recomputed every step\n"
     "beta = nu * nu * r * r / 4.0\n"
     "u = v_next - (beta / h) * x_prev\n"
     "...\n"
     "rho = min(d / d_prev, 1.0)\n"
     "r   = 2.0 * rho / (1.0 + rho * rho)")
p("Every routine returns an <font face='Mono'>IterationResult</font> carrying the full history "
  "— residual per step, β per step, matvec count, wall-clock, and flags for whether it "
  "converged or diverged. The experiment scripts do nothing but call these and tabulate.")

h2("5.2  src/google_matrix.py — G without building G")
code("class GoogleMatrix:\n"
     "    def matvec(self, x):\n"
     "        self.count += 1\n"
     "        x = np.asarray(x).ravel()\n"
     "        y = self.d * (self.P @ x)\n"
     "        dangling_mass = x[self.dangling].sum()\n"
     "        total_mass    = x.sum()\n"
     "        y += (self.d * dangling_mass + (1.0 - self.d) * total_mass) / self.n\n"
     "        return y")
p("Three details worth noting. The <font face='Mono'>count</font> makes every experiment's "
  "matvec tally automatic rather than hand-maintained. <font face='Mono'>with_damping(d)</font> "
  "re-uses the sparse structure so sweeping d does not rebuild the matrix. And the input dtype "
  "is preserved, because the eigenvalue solver probes the operator with <i>complex</i> vectors "
  "— quietly casting to float would corrupt the analysis in Part 7.")

h2("5.3  The supporting modules")
table([
    ["File", "What it gives you"],
    ["<font face='Mono'>datasets.py</font>",
     "Downloads and caches SNAP and SuiteSparse data. Also holds a synthetic graph generator "
     "with power-law out-degrees and dangling nodes, so the whole project still runs with no "
     "internet (<font face='Mono'>--offline</font>)."],
    ["<font face='Mono'>metrics.py</font>",
     "Ranking-quality measures. The paper only ever looks at the residual; these ask the "
     "question a search engine actually cares about — is the <i>order</i> right? "
     "precision@k, Kendall tau, ℓ1 error."],
    ["<font face='Mono'>spectrum.py</font>",
     "The analysis machinery for Part 7: μ(λ) from equation (2.7), the true convergence "
     "rate as a function of β, and the stability limit. It samples eigenvalues by largest "
     "magnitude <i>and</i> largest/smallest imaginary part — for a reason explained in "
     "Part 7."],
    ["<font face='Mono'>pagerank_methods.py</font>",
     "Defines the five configurations once, so every experiment compares identical settings."],
    ["<font face='Mono'>plotting.py</font>",
     "One colour-blind-safe palette and one style, so every figure in the report matches."],
], widths=[4.0 * cm, TEXT_W - 4.0 * cm])

h2("5.4  The gates")
p("Two experiments print an explicit <b>PASS/FAIL</b> and exit non-zero on failure.")
bullets([
    "<b>Phase 1 gate</b> (<font face='Mono'>exp1</font>): does our code reproduce the paper on "
    "the paper's own matrices? If not, nothing downstream means anything.",
    "<b>Phase 2 gate</b> (<font face='Mono'>exp2</font>): is our Google matrix correct? Checked "
    "against NetworkX <i>and</i> against its own fixed point ||Gx − x||₁ &lt; "
    "10<sup>−12</sup>, which needs no other library to be trusted.",
])


# ==========================================================================
h1("Part 6 — Every figure explained")
p("Thirteen figures. For each one: what is plotted, and what you are supposed to notice. "
  "All exist as both PDF (for LaTeX) and PNG (for slides) in "
  "<font face='Mono'>figures/</font>.")
note("Several figures are wide and appear small here. Open the matching "
     "<font face='Mono'>.pdf</font> in <font face='Mono'>figures/</font> to zoom in — they "
     "are vector graphics, so they stay sharp at any magnification.", BLUE)

h2("6.1  fig1_rate_comparison — how much momentum could help")
figure("figures/fig1_rate_comparison.png",
       "Reproduction of Figure 1 of the paper. The black curve is the accelerated rate "
       "ρ(r); the thin curves are r<sup>p</sup> for various p.", zoom_hint=True)
p("<b>Read it as:</b> momentum turns a rate of r into a rate of ρ(r). The black curve "
  "sitting below r<sup>6</sup> for r &gt; 0.945 means that in that regime one momentum step is "
  "worth about six plain steps. <b>Look for:</b> the curve dropping further below the thin lines "
  "as r → 1. This is pure theory — no data yet.")

h2("6.2  fig3_matrix1 — the reproduction, and β tuning itself")
figure("figures/fig3_matrix1.png",
       "Left: convergence on diag(1000:-1:1), the paper's benchmark. Right: the β chosen "
       "automatically by Algorithm 3.1, against the optimal value.", zoom_hint=True)
p("<b>Left, look for:</b> the green curve (dynamic) plunging while the blue (plain power) barely "
  "moves. 683 matvecs versus 27,619 — a 40× speed-up, against 44.7× predicted.")
p("<b>Right, look for:</b> the green line settling onto the black dashed line. That dashed line "
  "is λ₂²/4, which the algorithm was <i>never told</i>. It found it from residual "
  "ratios alone. This panel is the paper's central claim, working.")

h2("6.3  exp1_suitesparse — the same on the paper's real matrices")
figure("figures/exp1_suitesparse.png",
       "Kuu, Muu, ash292 and bcspwr06 from the SuiteSparse collection.", zoom_hint=True)
p("<b>Look for:</b> dynamic (green) at or below static (orange) on all four. Our measured "
  "spectral ratios match the paper's published values to four decimals, which is independent "
  "evidence we built the right matrices.")

h2("6.4  exp2_pagerank_convergence — the first PageRank result")
figure("figures/exp2_pagerank_convergence.png",
       "One row per graph. Left column: residual against iteration. Right column: the β "
       "chosen, against d²/4 and against the ¼ barrier.", zoom_hint=True)
p("<b>Left, look for:</b> the pink curve (Algorithm 3.1 as published) <i>flattening out and "
  "turning upward</i> instead of falling. It is not slow — it is diverging. Plain power "
  "iteration (blue) wins on all three graphs.")
p("<b>Right, look for:</b> the pink β line climbing to the red dotted line at 0.25. "
  "Section 2 of the paper proves convergence is impossible at or above β = λ₁²/4 "
  "= ¼. The algorithm drives itself there. This panel is the failure, caught in the act.")

h2("6.5  exp3_damping_sweep — measured against predicted")
figure("figures/exp3_damping_sweep.png",
       "Left: speed-up against the damping factor, with the theoretical curve dashed. "
       "Right: cost as the spectral gap closes.", zoom_hint=True)
p("<b>Look for:</b> the gap between the dashed theory curve and the measured points, and the red "
  "parity line at 1.0. Almost everything sits <i>below</i> parity — slower than plain power "
  "iteration. The exception is cit-HepPh at d = 0.95 and 0.99, where our safeguarded version "
  "does reach 2.3× and 4.1×.")

h2("6.6  exp4_ranking_quality — the practical punchline")
figure("figures/exp4_ranking_quality.png",
       "Left column: the eigenvalue residual. Right column: precision@100, i.e. how much of the "
       "true top-100 we have found.", zoom_hint=True)
p("<b>Look for:</b> the right-hand curves hitting 1.0 almost immediately, while the left-hand "
  "curves keep falling for hundreds more steps. At d = 0.99 the top-100 ranking is correct after "
  "<b>7</b> matrix-vector products; driving the residual to 10<sup>−12</sup> takes "
  "<b>2,005</b>.")
note("This is arguably the most useful finding in the project. For PageRank, accelerating the "
     "tail of the residual optimises a quantity nobody reads. Whatever the momentum debate, "
     "<i>the ranking was already right</i>.", GREEN, "WHY IT MATTERS")

h2("6.7  exp5_rate_vs_beta — the paper's β is out of bounds")
figure("figures/exp5_rate_vs_beta.png",
       "True convergence rate as a function of β, computed from the sampled spectrum. "
       "The × marks the paper's β = d²/4.", zoom_hint=True)
p("<b>Look for:</b> the red dotted line at rate = 1. Anything above it diverges. At d = 0.95 and "
  "0.99 the × sits <i>above</i> that line. The paper's optimal parameter is not merely "
  "sub-optimal for PageRank — it is outside the stability region.")

h2("6.8  exp5_spectrum_plane — the smoking gun")
figure("figures/exp5_spectrum_plane.png",
       "The eigenvalues of G plotted in the complex plane, coloured by |μ(λ)|. Modes "
       "circled in red overtake the dominant one.", zoom_hint=True)
p("<b>Look for:</b> <i>where</i> the red circles are. They are not near the outer dashed circle "
  "where the big eigenvalues live. They are clustered near the centre, just off the horizontal "
  "axis — small eigenvalues with a large imaginary part.")
p("That is the counter-intuitive heart of the project. The modes that destroy the method are the "
  "ones a normal eigenvalue solver would never even return, because it returns the "
  "<i>largest</i> ones. If you show one figure in the presentation, show this one.")

h2("6.9  exp6_symmetric_control — proof the code is fine")
figure("figures/exp6_symmetric_control.png",
       "The same three solvers on two symmetric ranking problems.", zoom_hint=True)
p("<b>Look for:</b> momentum working normally here — 3.58× against 3.79× "
  "predicted on the right-hand panel. That panel uses the <b>same cit-HepPh data</b> as the "
  "failing PageRank run, just made symmetric (co-citation: how many papers cite both i and j). "
  "Same data, symmetric instead of directed, and the acceleration appears.")

h2("6.10  exp7_normality — the controlled proof")
figure("figures/exp7_normality.png",
       "Three panels holding everything fixed except one variable.", zoom_hint=True)
p("A sharp examiner can object to 6.9: the co-citation control changes the symmetry <i>and</i> "
  "the matrix <i>and</i> r all at once. So how do we know which one matters? This figure answers "
  "that by changing exactly one thing.")
bullets([
    "<b>Panel A:</b> λ₁ = 1 and |λ₂| = r are held fixed, so the dashed "
    "predicted speed-up is the same on every point. Only the <i>angle</i> of the subdominant "
    "eigenvalues is rotated. The circles (Algorithm 3.1) exist only at angle 0; everywhere else "
    "it diverged.",
    "<b>Panel B:</b> the blue power-iteration line is almost flat — the problems are "
    "equally hard throughout, so nothing can be blamed on difficulty. Meanwhile the momentum "
    "methods collapse.",
    "<b>Panel C:</b> the orange curve is the strongest rival mode, the blue line is the winner. "
    "Where they cross, theory says the static method must fail — and the orange square "
    "(measured) sits exactly there. Algorithm 3.1 (pink diamond) fails even <i>earlier</i>, for "
    "a different reason.",
])
note("Momentum goes from 3.14× to divergence after as little as a <b>3.6° "
     "rotation</b> of the "
     "spectrum, with every other quantity pinned. One variable in, one outcome out. This is the "
     "strongest evidence in the project and takes 20 seconds to re-run.", NAVY, "THE PROOF")

h2("6.11  The teaching figures")
p("Figures A–D earlier in this guide (<font face='Mono'>figures/explain/</font>) were made for "
  "this document, not for the report: the tiny four-page web, the spectral-gap cost curves, the "
  "momentum “flattening” picture, and the pipeline diagram. They are the ones to use "
  "if you need to explain the project to someone in two minutes.")

# ==========================================================================
h1("Part 7 — What we found, and why")

h2("7.1  The short version")
p("<b>The method does not make PageRank faster.</b> Plain power iteration wins on every real "
  "graph we tried. Algorithm 3.1 exactly as published <i>diverges</i> on all of them.")
p("That is a negative result, but it is not a failed project, for three reasons: we can prove "
  "our implementation is correct, we can explain precisely why the method fails, and we can fix "
  "it. Those three things are the project.")

h2("7.2  Failure mode A: the self-tuning rule turns on itself")
p("Recall from Part 2 that Algorithm 3.1 estimates r by watching the residual shrink. The "
  "formula includes a safety clamp, <font face='Mono'>min(ρ, 1)</font>, for the case where "
  "the residual <i>grows</i>. On a symmetric matrix the residual falls monotonically and that "
  "clamp never fires.")
p("The Google matrix is not symmetric. Its residual oscillates. The moment one step has "
  "d<sub>k+1</sub> &gt; d<sub>k</sub>, the clamp gives ρ = 1, so")
code("r  =  2·1 / (1 + 1²)  =  1        and therefore\n"
     "β  =  ν²·1²/4  →  ¼   ← exactly the value where convergence stops")
p("Once there, the residual grows, ρ stays pinned at 1, and the method never recovers. On "
  "<font face='Mono'>wiki-Vote</font> the residual bottoms out near 10<sup>−3</sup> around "
  "iteration 20 and climbs back to 0.15 — <b>2,000 iterations end further from the answer "
  "than 20 did.</b>")

h2("7.3  Failure mode B: complex eigenvalues (the real cause)")
p("Bounding r fixes mode A but not this. Go back to the key formula and the Vieta observation "
  "from Part 2: the two roots always multiply to β.")
bullets([
    "If λ is <b>real</b>, the roots are complex conjugates. Conjugates have equal size. "
    "Their product is β, so each is exactly √β. <b>They cannot split.</b>",
    "If λ is <b>complex</b>, they are not conjugates. They split — one grows, the "
    "other shrinks to keep the product at β.",
])
p("For a purely imaginary λ = iy the larger root is (|y| + √(y²+4β))/2, "
  "which grows without bound in |y|. Concretely, at d = 0.99 on "
  "<font face='Mono'>wiki-Vote</font>:")
table([
    ["eigenvalue λ", "|λ|", "|μ| after momentum", ""],
    ["1 (the winner)", "1.000", "0.5705", "what we want to find"],
    ["0.90 (real)", "0.900", "0.4950", "squashed — harmless"],
    ["0.23 (real)", "0.230", "0.4950", "squashed — harmless"],
    ["0.0102 + 0.2312i", "<b>0.231</b>", "<b>0.6239</b>", "<b>beats the winner</b>"],
], widths=[5.2 * cm, 2.2 * cm, 3.6 * cm, TEXT_W - 11.0 * cm])
p("An eigenvalue of size <b>0.23</b> — utterly harmless to plain power iteration — "
  "overtakes the dominant mode and the iteration converges to the wrong eigenvector. On "
  "<font face='Mono'>wiki-Vote</font> at d = 0.99, 126 of 179 sampled modes do this.")

h2("7.4  Where the complex eigenvalues come from: direction")
code("directed 4-cycle    A→B→C→D→A   eigenvalues:  +1, −1, +i, −i     ← complex\n"
     "the same cycle undirected              eigenvalues:  +2, −2,  0,  0     ← all real")
p("Directed cycles produce roots of unity. Real web and citation graphs are full of directed "
  "cycles of every length, so the spectrum of G spreads off the real axis — we measure "
  "|Im λ| up to 0.41 on cit-HepPh. Undirected graphs give symmetric matrices, whose "
  "eigenvalues are real by theorem. <b>That is the whole causal chain:</b>")
code("directed graph → non-symmetric G → complex eigenvalues\n"
     "               → the two μ roots split → a tail mode overtakes λ₁ → divergence")

h2("7.5  Our fix")
p("The safeguarded variant uses only information PageRank gives away for free, plus feedback "
  "from the residual:")
table([
    ["Safeguard", "Uses", "Fixes"],
    ["Cap β below λ₁²/4", "λ₁ = 1 exactly, by stochasticity",
     "Keeps β strictly inside the convergence region."],
    ["Bound r ≤ d", "the theorem |λ₂| ≤ d", "Kills failure mode A."],
    ["Backtracking", "nothing — pure residual feedback",
     "Handles mode B, which no advance bound can prevent."],
    ["Slow growth", "nothing",
     "Retreat alone overshoots; alternating shrink and grow keeps β just below the "
     "stability edge."],
    ["Backtrack budget", "nothing",
     "Turns it into a guarantee: worst case is power iteration plus 60 matvecs."],
], widths=[3.5 * cm, 4.8 * cm, TEXT_W - 8.3 * cm])
p("It converges on every graph and damping factor we tested, and on 27 of 27 matrices in the "
  "controlled sweep, reaching 4.08× on cit-HepPh at d = 0.99 where the published algorithm "
  "diverges.")

h2("7.6  Honest limitations")
bullets([
    "The spectral rates come from a <i>sample</i> of the spectrum, so they are lower bounds. A "
    "sampled rate above 1 proves divergence; a sampled rate below 1 does not prove convergence. "
    "The “achievable speed-up” numbers are therefore optimistic, and "
    "<font face='Mono'>exp5</font> measures its own predictions and prints "
    "SAMPLE INCOMPLETE where they do not hold up.",
    "Three real graphs, one per family. The controlled sweep in "
    "<font face='Mono'>exp7</font> compensates by establishing the mechanism on a synthetic "
    "family instead of relying on the graph sample.",
    "The safeguard's constants were tuned on these graphs. The <i>guarantee</i> does not depend "
    "on them; the <i>acceleration</i> does.",
    "We did not implement DMPower, a competing method the paper also compares against. It would "
    "not change any conclusion, since the winner throughout is plain power iteration.",
])

# ==========================================================================
h1("Part 8 — Running it, and answering questions")

h2("8.1  Commands")
code("make setup     # create .venv and install pinned dependencies\n"
     "make test      # 19 unit tests, about 1 second\n"
     "make all       # every experiment; ~15 min, downloads ~90 MB the first time\n"
     "make quick     # offline subset on synthetic graphs, no network needed\n"
     "\n"
     "make phase1 ... make phase7      # one phase at a time\n"
     "\n"
     "# every experiment takes flags:\n"
     "./.venv/bin/python experiments/exp2_pagerank.py --graphs wiki-Vote --offline")

h2("8.2  What to check if you only have a minute")
code("make test                                  →  19 passed\n"
     "make phase1   →  Phase 1 gate: PASS -- proceed to Phase 2\n"
     "make phase2   →  Phase 2 gate: PASS\n"
     "make phase7   →  3/3 exact agreement ... Non-normality is the cause.")

h2("8.3  Likely questions, and the honest answers")
table([
    ["Question", "Answer"],
    ["“Your proposal promised to measure how much <i>faster</i> it gets. It got slower.”",
     "The promise was to <i>measure</i>, and we did — on more graphs and more damping "
     "factors than proposed. The measured answer is that it does not get faster, and we can show "
     "exactly why."],
    ["“How do you know your code is not just buggy?”",
     "Three ways. We reproduce the paper's own benchmarks at 40.4× versus 44.7× "
     "predicted. The symmetric control on the same data gives 3.58× versus 3.79×. And "
     "19 unit tests, including a Google matrix checked against NetworkX <i>and</i> against its "
     "own fixed point."],
    ["“Maybe the symmetric control differs for some other reason.”",
     "That is exactly why <font face='Mono'>exp7</font> exists. It pins λ₁ and "
     "|λ₂| and rotates only the angle of the spectrum. Power iteration varies 8–13% "
     "across the whole sweep; momentum goes from 3.14× to divergence after as little "
     "as 3.6°."],
    ["“Is the result specific to your choice of β?”",
     "No. <font face='Mono'>exp5</font> scans <i>all</i> β and reports the best achievable "
     "rate, then verifies it empirically. Even the best β gives only 1.26–1.61× "
     "measured."],
    ["“So the whole thing was pointless?”",
     "The opposite. We found the method's hidden assumption (a real spectrum), showed it fails "
     "on an important real problem, explained the mechanism quantitatively, fixed it, and found "
     "that for ranking the residual was the wrong thing to optimise anyway."],
], widths=[5.6 * cm, TEXT_W - 5.6 * cm])

h2("8.4  The thirty-second summary")
note("The paper's method accelerates the power iteration by reshaping the matrix spectrum so "
     "every rival mode gets crushed onto a single small circle. That reshaping only works if the "
     "eigenvalues are <b>real</b>. The Google matrix is built from a <b>directed</b> graph, so "
     "its eigenvalues are complex, and complex ones escape the circle instead of being crushed "
     "— some of them overtaking the very mode we are trying to find. We prove this with a "
     "controlled experiment, fix it with safeguards that use only facts PageRank already gives "
     "us, and show that for ranking purposes the answer was correct after 7 iterations anyway.",
     NAVY, "SAY THIS")

build()
