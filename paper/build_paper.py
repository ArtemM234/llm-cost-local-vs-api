from __future__ import annotations

import csv
import datetime as dt
import math
import re
import statistics as st
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
OUT = ROOT / "submission" / "Matyrin_Popov_Local_LLM_vs_Claude_API.docx"

AUTHOR = "Artem Matyrin, Andrii Popov"
STUDENT_ID = "76671, 75718"
REPO_URL = "https://github.com/ArtemM234/llm-cost-local-vs-api"
TITLE = "Local Language Model or Claude API: The Cost of a Correct Answer on a Home Computer"

BODY_FONT, HEAD_FONT = "Times New Roman", "Arial"


def load(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


summ = {r["model_name"]: r for r in load("summary.csv")}
grid = load("breakeven.csv")
rows = load("results.csv")
H, S, Q = summ["haiku-5.5"], summ["sonnet-5.5"], summ["qwen3-8b"]


def f(r: dict[str, str], k: str) -> float:
    return float(r[k])


def be(api: str, pricing: str, life: int, p_el: float) -> dict[str, str]:
    return next(r for r in grid if r["api"] == api and r["api_pricing"] == pricing
                and int(r["lifetime_years"]) == life and math.isclose(float(r["electricity_usd_kwh"]), p_el))


acc = {k: f(v, "acc") for k, v in summ.items()}
E_wh = f(Q, "energy_wh_net_mean")
t_q = f(Q, "t_task_s")
v_max = 86400 / t_q
per_correct = {k: f(v, "cost_usd_mean") / f(v, "acc") for k, v in (("haiku", H), ("sonnet", S))}
per_correct_batch = {k: f(v, "cost_usd_batch_mean") / f(v, "acc") for k, v in (("haiku", H), ("sonnet", S))}
owned = {p: E_wh / 1000 * p / acc["qwen3-8b"] for p in (0.10, 0.20, 0.35)}
hw_day = 2000 / (4 * 365)
v_son = float(be("sonnet-5.5", "standard", 4, 0.20)["v_star_per_day"])
v_hai = float(be("haiku-5.5", "standard", 4, 0.20)["v_star_per_day"])
v_hai_cheap = {life: float(be("haiku-5.5", "standard", life, 0.10)["v_star_per_day"]) for life in (4, 5)}
son_std = [float(r["v_star_per_day"]) for r in grid if r["api"] == "sonnet-5.5" and r["api_pricing"] == "standard"]
son_bat = [float(r["v_star_per_day"]) for r in grid if r["api"] == "sonnet-5.5" and r["api_pricing"] == "batch"]

qrows = [r for r in rows if r["model_name"] == "qwen3-8b"]
tps = [float(r["gen_tokens_per_s"]) for r in qrows]
tps_med, tps_min, tps_max = st.median(tps), min(tps), max(tps)
avg_w = st.fmean(float(r["avg_power_w"]) for r in qrows)
idle_w = st.fmean(float(r["idle_power_w"]) for r in qrows)
prompt_tps = st.median(float(r["prompt_tokens_per_s"]) for r in qrows)
weights_gb = 5.225388164
bandwidth = 504.0
ceiling = bandwidth / weights_gb
api_spend = sum(float(r["cost_usd"]) for r in rows if r["backend"] == "anthropic")
failed_all = sorted({r["task_id"] for r in rows if r["correct"] != "1"}
                    & {r["task_id"] for r in rows if r["correct"] != "1" and r["model_name"] == "haiku-5.5"}
                    & {r["task_id"] for r in rows if r["correct"] != "1" and r["model_name"] == "sonnet-5.5"}
                    & {r["task_id"] for r in rows if r["correct"] != "1" and r["model_name"] == "qwen3-8b"})
assert failed_all == ["gsm8k/119", "gsm8k/184"], failed_all

def v_star(c_api: float, acc_api: float, e_wh: float, p: float = 0.20) -> float:
    d = acc["qwen3-8b"] * c_api / acc_api - e_wh / 1000 * p
    return hw_day / d if d > 0 else math.inf


v_son_2x = v_star(f(S, "cost_usd_mean"), acc["sonnet-5.5"], 2 * E_wh)
v_hai_2x = v_star(f(H, "cost_usd_mean"), acc["haiku-5.5"], 2 * E_wh)
assert math.isclose(v_star(f(S, "cost_usd_mean"), acc["sonnet-5.5"], E_wh), v_son, rel_tol=1e-9)


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def usd(x: float, sig: int = 2) -> str:
    if x == 0:
        return "$0"
    dec = max(2, sig - 1 - math.floor(math.log10(abs(x))))
    return f"${x:.{dec}f}"


def n0(x: float) -> str:
    return f"{x:,.0f}"


doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1))

normal = doc.styles["Normal"]
normal.font.name = BODY_FONT
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
normal.paragraph_format.space_after = Pt(0)
normal.paragraph_format.space_before = Pt(0)

for name, size, align in (("Heading 1", 13, WD_ALIGN_PARAGRAPH.CENTER), ("Heading 2", 12, WD_ALIGN_PARAGRAPH.LEFT)):
    hs = doc.styles[name]
    hs.font.name, hs.font.size, hs.font.bold, hs.font.italic = HEAD_FONT, Pt(size), True, False
    hs.font.color.rgb = None
    hs.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)
    hs.element.rPr.rFonts.set(qn("w:hAnsi"), HEAD_FONT)
    hs.element.rPr.rFonts.set(qn("w:ascii"), HEAD_FONT)
    pf = hs.paragraph_format
    pf.alignment, pf.space_before, pf.space_after = align, Pt(12), Pt(6)
    pf.line_spacing_rule, pf.keep_with_next = WD_LINE_SPACING.SINGLE, True

hdr = sec.header.paragraphs[0]
hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
run = hdr.add_run()
for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
    if tag:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), tag)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = text
    run._r.append(el)

STAR = "⁣"
TOKEN = re.compile(r"(\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\}|\[\[[^\]]+\]\]|<<[^>]+>>)")


def add_runs(p, text: str, base_italic: bool = False, bold: bool = False, size: float | None = None) -> None:
    text = text.replace("\\*", STAR)
    for part in TOKEN.split(text):
        if not part:
            continue
        r = p.add_run()
        if part.startswith("<<"):
            url = part[2:-2]
            r.text, r.italic = url, base_italic
            link = OxmlElement("w:hyperlink")
            link.set(qn("r:id"), p.part.relate_to(url, RT.HYPERLINK, is_external=True))
            r._r.addprevious(link)
            link.append(r._r)
        elif part.startswith("[["):
            r.text, r.italic = part[2:-2], base_italic
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif part.startswith("*"):
            r.text, r.italic = part[1:-1], not base_italic
        elif part.startswith("_{"):
            r.text, r.font.subscript, r.italic = part[2:-1], True, base_italic
        elif part.startswith("^{"):
            r.text, r.font.superscript, r.italic = part[2:-1], True, base_italic
        else:
            r.text, r.italic = part, base_italic
        r.text = r.text.replace(STAR, "*")
        r.bold = bold or None
        if size:
            r.font.size = Pt(size)


def para(text: str, indent: bool = True, align=None, italic: bool = False, bold: bool = False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.5)
    if align is not None:
        p.alignment = align
    add_runs(p, text, base_italic=italic, bold=bold)
    return p


def eq(text: str) -> None:
    p = para(text, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def h1(t: str) -> None:
    doc.add_heading(t, level=1)


def h2(t: str) -> None:
    doc.add_heading(t, level=2)


def page_break() -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def figure(n: int, path: Path, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(6.0))
    lab = para(f"Figure {n}", indent=False, bold=True)
    lab.paragraph_format.keep_with_next = True
    lab.paragraph_format.space_before = Pt(6)
    para(caption, indent=False, italic=True)


def _border(el, edges: dict[str, int]) -> None:
    b = OxmlElement("w:tblBorders" if el.tag.endswith("tblPr") else "w:tcBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        sz = edges.get(edge, 0)
        e.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            e.set(qn("w:sz"), str(sz))
            e.set(qn("w:color"), "000000")
        b.append(e)
    el.append(b)


def table(n: int, title: str, header: list[str], body: list[list[str]], widths: list[float],
          text_table: bool = False) -> None:
    lab = para(f"Table {n}", indent=False, bold=True)
    lab.paragraph_format.keep_with_next = True
    lab.paragraph_format.space_before = Pt(12)
    cap = para(title, indent=False, italic=True)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(body), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _border(t._tbl.tblPr, {"top": 8, "bottom": 8})
    for i, row in enumerate([header] + body):
        for j, txt in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            cp = c.paragraphs[0]
            cp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            cp.paragraph_format.keep_with_next = i < len(body)
            cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 or text_table else WD_ALIGN_PARAGRAPH.RIGHT
            add_runs(cp, txt, size=10)
            if i == 0:
                _border(c._tc.get_or_add_tcPr(), {"bottom": 4})
    doc.add_paragraph().paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE


for _ in range(6):
    doc.add_paragraph()
para(TITLE, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
doc.add_paragraph()
for line in (AUTHOR, f"{STUDENT_ID}, Vistula University, Warsaw", "Architecture of Computers (CII3SE003CI), Group 1",
             "Professor Tariq Eldakruri", "Winter 2026/2027"):
    para(line, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

para(TITLE, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
para("Abstract", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
para(
    "*Background:* A student or a small team can run an open language model on their own gaming computer or pay "
    "a commercial API per token. Price lists compare tokens, not correct answers, and published break-even "
    "studies look at organisations with servers. "
    "*Materials and Methods:* The first 200 problems of the GSM8K test set were given, with one prompt and one "
    "automatic checker, to qwen3:8b (4-bit) on a home PC with an NVIDIA RTX 4070 and to Claude Haiku 5.5 and "
    "Claude Sonnet 5.5 through the Anthropic API, with reasoning modes switched off. Tokens, time and graphics card "
    "power were recorded, and a simple cost model gives the daily number of tasks at which buying the PC pays off. "
    f"*Results:* Accuracy was {pct(acc['qwen3-8b'], 0)} for qwen3:8b, {pct(acc['haiku-5.5'], 0)} for Haiku and "
    f"{pct(acc['sonnet-5.5'], 0)} for Sonnet. On a PC that is already owned, a correct answer costs "
    f"{usd(owned[0.20])} in electricity, against {usd(per_correct['haiku'])} for Haiku and "
    f"{usd(per_correct['sonnet'])} for Sonnet. A $2,000 PC bought for the job pays off against Sonnet above about "
    f"{n0(round(v_son, -1))} tasks per day, but against Haiku only above {n0(round(v_hai, -2))}, more than the "
    f"{n0(v_max)} tasks the PC can process in a day. Text generation ran at {pct(tps_med / ceiling, 0)} of the "
    "limit set by memory bandwidth. "
    "*Conclusion:* A local model is the cheapest option when the computer is already there, and buying one pays "
    "off only against an expensive API model. Against a cheap API model, and with batch prices, it does not.",
    indent=False,
)
para("*Keywords:* large language models; inference cost; local deployment; energy consumption; memory bandwidth; "
     "break-even analysis", indent=False)
page_break()

h1("1. Introduction")
h2("1.1 Global/Regional Context")
para("Large language models are mostly used through commercial APIs that charge per million tokens. At the same "
     "time, open-weight models such as Qwen3 (Yang et al. 2025) can be downloaded and run on a normal computer with "
     "free tools such as Ollama (Ollama 2026). Four-bit quantisation makes an eight-billion-parameter model small "
     "enough for the memory of a consumer graphics card (Lin et al. 2023). API prices are far from uniform: they "
     "differ by about two orders of magnitude between models (Chen, Zaharia, and Zou 2023).")
h2("1.2 Local Context / Case Study")
para("The case in this paper is a typical student setup: a home PC with a gaming graphics card, the NVIDIA "
     "GeForce RTX 4070, compared with two models of the Anthropic Claude API. The question such a user faces is "
     "simple: is it cheaper to answer questions on the computer they already have, or to pay the API, and does it "
     "make sense to buy a computer for this?")
h2("1.3 Problem Statement")
para("Two recent studies compare local and cloud models, but both for organisations. Pan et al. (2025) build a "
     "break-even model for running open models on company servers instead of paying cloud providers. Peng, Lin, "
     "and Lee (2026) follow one developer's coding agent and find that prompt caching cut the cloud cost by 88.6%, "
     "to $0.57 per million tokens against $2.83 on their own server. Neither looks at one consumer computer. Price "
     "per token also hides quality: a cheaper model that answers wrongly more often is not really cheaper. Energy "
     "use of inference has been measured mainly on data-centre hardware (Samsi et al. 2023; Luccioni, Jernite, and "
     "Strubell 2024).")
h2("1.4 Research Objectives")
para("The research question is: at what daily number of tasks does a local model on a home computer become "
     "cheaper than the Claude API per correctly solved task? To answer it, the paper measures accuracy, time and "
     "energy of a local model and two API models on the same tasks, and checks how close the local generation "
     "speed comes to the limit set by memory bandwidth, the computer-architecture side of the question.")
h2("1.5 Paper Structure")
para("Section 2 describes the setup and the cost model, Section 3 gives the measurements and the break-even "
     "volumes, Section 4 discusses them, and Section 5 concludes. The tables are in Section 7.")

h1("2. Methodology")
h2("2.1 Study Area / System Description")
para("The local machine is a home PC with an AMD Ryzen 7 5700X3D processor, 32 GB of RAM and an NVIDIA GeForce "
     "RTX 4070 with 12 GB of GDDR6X memory on a 192-bit bus (NVIDIA, n.d.), running Windows 11 Pro. The local "
     f"model is qwen3:8b (8.2 billion parameters, Q4_K_M quantisation, {weights_gb:.2f} GB of weights) served by "
     "Ollama 0.40.16. The API models are Claude Haiku 5.5 and Claude Sonnet 5.5. Table 1 lists the setup.")
h2("2.2 Technology / Intervention / Framework")
para("A small Python program sends every task to every model and stores the answer, token counts, time and "
     "power readings. All models get the same prompt, which asks them to end with a line “ANSWER: <number>”, "
     "and the same checker: it takes the number after “ANSWER:” (or the last number if the line is missing) "
     "and marks the task correct if it equals the reference answer. The output limit is 4,096 tokens for all "
     "models, and each task is answered once.")
para("Reasoning (“thinking”) modes were switched off for all three models, because they make answers "
     "longer and more expensive and harder to compare. The local model runs with temperature 0 and a fixed seed. "
     "Claude 5.5 models do not accept a non-default temperature, so they run with the default, which is a "
     "difference worth keeping in mind (Section 2.8).")
h2("2.3 Resource / Data Assessment")
para("Energy is measured with nvidia-smi, which reports the power of the graphics card. It is read every 100 ms "
     "during each answer, the idle power measured for 30 seconds before the run is subtracted, and the rest is "
     "added up into watt-hours per task. This covers the graphics card only, not the whole computer. API cost is "
     "computed from the token counts that the API returns with each answer and the official prices, checked on "
     "23 November 2026 (Anthropic 2026): $0.10 per million input tokens and $0.50 per million output tokens for "
     "Haiku 5.5, $2 and $10 for Sonnet 5.5, and half of that for batch requests.")
h2("2.4 Scenario Definition / Experimental Setup")
para("The tasks are the first 200 of the 1,319 problems in the GSM8K test set, short school maths word problems "
     "with a numeric answer (Cobbe et al. 2021). They were run on 24 November 2026, first 20 tasks as a pilot and "
     "then the other 180 with the same settings. Two scenarios are compared: the PC is already owned, so only "
     "electricity is paid, and the PC is bought for this work. For the second, the price is assumed to be $2,000 "
     "with a lifetime of four years, and electricity costs $0.20 per kWh. Both values are varied in Section 3.4.")
h2("2.5 Quantitative Assessment")
para("The main measure is the cost of one correctly solved task. For the API it is the average cost of a "
     "task divided by accuracy:")
eq("cost per correct (API) = *c*_{api} / *acc*_{api}")
para("For the local model, with *V* tasks per day, a PC price *C*_{hw}, a lifetime of *L* years, energy *E* per "
     "task and electricity price *p*, the daily cost and the cost per correct task are:", indent=False)
eq("*C*_{local}(*V*) = *C*_{hw} / (365 *L*) + *V E p*,    cost per correct (local) = *C*_{local}(*V*) / (*V acc*_{local})")
para("Setting the two equal gives the break-even volume *V*\\*, above which the local model is cheaper. If the "
     "denominator is zero or negative, the local model is never cheaper:", indent=False)
eq("*V*\\* = (*C*_{hw} / (365 *L*)) / (*acc*_{local} *c*_{api} / *acc*_{api} − *E p*)")
para("A computer also has a limit: one task takes *t* seconds, so it can do at most *V*_{max} = 86,400 / *t* "
     "tasks per day. If *V*\\* is above *V*_{max}, buying the computer cannot pay off.", indent=False)
h2("2.6 Environmental / Social / Risk")
para("Electricity is the only running cost of the local model, so it is also the environmental side of the "
     "comparison. The energy used by the API provider is not visible to the customer and is part of the API "
     "price, so only local energy is measured.")
h2("2.7 Data Sources")
para("Tasks come from GSM8K (Cobbe et al. 2021), prices from the Anthropic pricing page (Anthropic 2026), and "
     "the graphics card data from NVIDIA (n.d.) and from nvidia-smi on the test machine. All measurements are "
     "from the authors' own runs.")
h2("2.8 Uncertainty Analysis")
para("Accuracy is given with a 95% Wilson confidence interval (Wilson 1927), which works well for proportions "
     "close to one. Each task was answered once, so the run-to-run variation of the Claude models, which do not "
     f"run at temperature 0, is not measured. Energy per task varies with answer length (standard deviation "
     f"{float(Q['energy_wh_net_sd']):.3f} Wh), and the average over 200 tasks is used.")

h1("3. Results")
h2("3.1 Resource / System Potential")
para("When a model generates text, every new token needs all the weights to be read from memory once, so speed "
     "is limited by memory bandwidth rather than by computing power (Williams, Waterman, and Patterson 2009; Pope "
     "et al. 2022; Yuan et al. 2024). The upper limit is roughly bandwidth divided by the size of the weights. "
     "The RTX 4070 has a 192-bit bus, and nvidia-smi reports a memory clock of 10,501 MHz, which is about 21 Gbit/s "
     f"per pin, so the bandwidth is 24 bytes × 21 GT/s = {bandwidth:.0f} GB/s. With {weights_gb:.2f} GB of "
     f"weights the limit is {ceiling:.1f} tokens per second.")
figure(1, FIG / "fig3_speed_vs_ceiling.png",
       "Generation Speed of qwen3:8b on the RTX 4070 for Each of the 200 Tasks, Against the Memory-Bandwidth Limit.")
para(f"The measured median speed was {tps_med:.1f} tokens per second (range {tps_min:.1f} to {tps_max:.1f}), "
     f"which is {pct(tps_med / ceiling, 0)} of the limit (Figure 1). So the simple rule explains most of the "
     "speed, and the rest is lost to work the formula ignores, such as reading the growing context. Reading the "
     f"prompt is a different case: it ran at a median of {n0(prompt_tps)} tokens per second, because all prompt "
     "tokens are processed together and the weights are read once for all of them.")
h2("3.2 Quantitative / Techno-Economic Results")
para(f"qwen3:8b solved {Q['n_correct']} of 200 tasks ({pct(acc['qwen3-8b'], 0)}), Haiku {H['n_correct']} "
     f"({pct(acc['haiku-5.5'], 0)}) and Sonnet {S['n_correct']} ({pct(acc['sonnet-5.5'], 0)}) (Table 2). The "
     "confidence intervals of Haiku and Sonnet overlap, so on 200 tasks the two cannot be told apart. The local "
     f"model took {t_q:.1f} seconds per task, against {f(H, 't_task_s'):.1f} for Haiku and {f(S, 't_task_s'):.1f} "
     f"for Sonnet, mostly because its answers were about twice as long ({f(Q, 'out_tokens_mean'):.0f} output "
     f"tokens against {f(H, 'out_tokens_mean'):.0f} and {f(S, 'out_tokens_mean'):.0f}). The whole main run cost "
     f"${api_spend:.2f} in API fees.")
para(f"If the PC is already owned, a correct answer costs {usd(owned[0.20])} in electricity. That is "
     f"{per_correct['haiku'] / owned[0.20]:.1f} times less than Haiku ({usd(per_correct['haiku'])}) and about "
     f"{per_correct['sonnet'] / owned[0.20]:.0f} times less than Sonnet ({usd(per_correct['sonnet'])}). If the PC "
     "is bought for $2,000, its price has to be spread over the tasks, and Figure 2 shows how the cost per correct "
     "answer falls as the daily volume grows.")
figure(2, FIG / "fig1_cost_vs_volume.png",
       "Cost per Correctly Solved Task Against the Number of Tasks per Day. PC price $2,000, lifetime four years, "
       "electricity $0.20 per kWh, standard API prices. Both axes are logarithmic.")
para(f"Against Sonnet the bought PC becomes cheaper at {n0(v_son)} tasks per day, well below its limit of "
     f"{n0(v_max)} tasks per day. Against Haiku the break-even volume is {n0(v_hai)} tasks per day, more than the "
     "computer can process, so buying a PC to replace Haiku does not pay off in this scenario.")
h2("3.3 Environmental / Social Results")
para(f"During generation the graphics card drew {avg_w:.0f} W on average, against {idle_w:.0f} W at idle, which "
     f"gives {E_wh:.3f} Wh per task above idle, or about {E_wh * 200:.0f} Wh for all 200 tasks. At $0.20 per kWh "
     f"this is {usd(E_wh / 1000 * 0.20)} per task. Electricity is therefore a very small part of the local cost "
     f"once a computer has to be bought: at the Sonnet break-even volume the PC costs ${hw_day:.2f} per day and "
     f"the electricity ${v_son * E_wh / 1000 * 0.20:.2f}.")
h2("3.4 Sensitivity Analysis")
para(f"Table 3 repeats the calculation for lifetimes of three to five years, electricity at $0.10, $0.20 and "
     f"$0.35 per kWh, and standard and batch API prices. Against Sonnet the break-even volume stays between "
     f"{n0(min(son_std))} and {n0(max(son_std))} tasks per day with standard prices and between {n0(min(son_bat))} "
     f"and {n0(max(son_bat))} with batch prices, always within what the PC can do. Against Haiku it is reachable "
     f"only with cheap electricity and a long lifetime ({n0(v_hai_cheap[5])} to {n0(v_hai_cheap[4])} tasks per "
     f"day, that is, the PC busy {pct(v_hai_cheap[5] / v_max, 0)} to {pct(v_hai_cheap[4] / v_max, 0)} of the day). "
     f"With batch prices Haiku costs {usd(per_correct_batch['haiku'])} per correct answer, which is less than "
     f"the local electricity alone at $0.35 per kWh ({usd(owned[0.35])}), so there even an owned PC is more "
     "expensive. Figure 3 shows how the break-even volume depends on the price of the computer.")
figure(3, FIG / "fig2_breakeven_vs_price.png",
       "Break-Even Volume Against the Price of the Computer. Lines are for $0.20 per kWh, bands cover $0.10 to "
       "$0.35 per kWh, lifetime four years, standard API prices.")

h1("4. Discussion")
h2("4.1 Interpretation of Results")
para("The result depends mostly on the price of the API model, not on electricity. Sonnet costs "
     f"{per_correct['sonnet'] / per_correct['haiku']:.1f} times more than Haiku per correct answer, and this "
     "difference decides whether buying a computer pays off. Electricity matters only when the computer is "
     "already owned, and then the local model is cheapest at normal electricity prices. This agrees with Peng, "
     "Lin, and Lee (2026), where a discount on the cloud side, prompt caching in their case, made the cloud "
     "cheaper than local hardware, and with the general idea of Pan et al. (2025) that the answer depends on the "
     "volume of work. The local model is cheaper but also less accurate: 94% against 98–99%.")
h2("4.2 Policy / Practical Implications")
para("For a student with a gaming PC, running a small model locally is the cheapest way to solve many simple "
     "tasks, as long as 94% accuracy is enough. Buying a computer only for this makes sense against an expensive "
     "model like Sonnet and with more than about a thousand tasks a day, which is a lot for one person. Against a "
     "cheap model like Haiku it is better to use the API, especially with batch requests.")
h2("4.3 Barriers & Risk Mitigation")
para("The cost model counts only correct answers, but wrong answers also cost time if someone has to find them. "
     "The local setup also needs time to install and maintain, which is not counted, and API prices can change. "
     "Because all the numbers come from the result files, the analysis can be repeated quickly with new prices.")
h2("4.4 Limitations & Future Research")
para("The study uses one computer, one local model, one set of school maths problems and 200 tasks answered "
     "once. GSM8K is public, so the models may have seen it during training. Energy covers the graphics card "
     "only, so the real energy of the whole PC is higher. Even if it were twice as high, the break-even volume "
     f"against Sonnet would only move from {n0(v_son)} to {n0(v_son_2x)} tasks per day, and against Haiku it "
     f"would stay out of reach ({n0(round(v_hai_2x, -2))}). Two tasks (119 and 184) were failed by all three "
     "models, and their GSM8K reference answers are doubtful: in 119 the reference reads “30 percent higher” "
     "as 30% less than Adrien's salary, and in 184 it gives a difference of 25 percentage points where all models "
     "answered a relative difference of 100%. The scoring was not changed. Future work could measure the whole "
     "PC with a wall meter, add programming tasks and larger models, and include prompt caching.")

h1("5. Conclusion")
para(f"On a home PC with an RTX 4070, qwen3:8b solved {pct(acc['qwen3-8b'], 0)} of 200 GSM8K tasks at "
     f"{usd(owned[0.20])} of electricity per correct answer, cheaper than Claude Haiku 5.5 and much cheaper than "
     f"Claude Sonnet 5.5, but less accurate. Buying a $2,000 PC for this pays off against Sonnet above about "
     f"{n0(round(v_son, -1))} tasks per day and practically never against Haiku. Generation speed reached "
     f"{pct(tps_med / ceiling, 0)} of the memory-bandwidth limit, so memory bandwidth is what sets the speed of "
     "a local model.")

h1("Data and Code Availability")
para(f"The measurement and analysis code, the configuration and prices, the raw model answers, the per-task "
     f"measurements (results.csv), the summary tables and the scripts that produce the figures and this paper are "
     f"available at <<{REPO_URL}>>. The GSM8K tasks are downloaded from the original public source and checked with "
     "a fixed SHA-256 hash. Repeating the run needs Ollama, an NVIDIA graphics card and an Anthropic API key.")
h1("Declaration of Generative AI Use")
para("Generative AI was used in this work. Claude (Anthropic), through the Claude Code tool, was used to help "
     "write the measurement and analysis code, to make the figures, to check the bibliographic data of the "
     "references against arXiv and Crossref, and to draft and edit the English text. The topic, the research "
     "question and the scope were chosen by the authors, and all experiments were run on the authors' own computer; "
     "every number in the paper comes from these runs and is computed from the result files by the published "
     "scripts. Claude Haiku 5.5 and Claude Sonnet 5.5 are also the objects of the study; their use as measured "
     "models is described in Section 2. The authors have reviewed the paper and take full responsibility for its "
     "content.")

page_break()
h1("6. References")
REFS = [
    "Anthropic. 2026. “Pricing.” Claude API Documentation. Accessed November 23, 2026. "
    "https://platform.claude.com/docs/en/about-claude/pricing.",
    "Chen, Lingjiao, Matei Zaharia, and James Zou. 2023. “FrugalGPT: How to Use Large Language Models While "
    "Reducing Cost and Improving Performance.” arXiv:2305.05176. https://doi.org/10.48550/arXiv.2305.05176.",
    "Cobbe, Karl, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, "
    "et al. 2021. “Training Verifiers to Solve Math Word Problems.” arXiv:2110.14168. "
    "https://doi.org/10.48550/arXiv.2110.14168.",
    "Lin, Ji, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, "
    "Chuang Gan, and Song Han. 2023. “AWQ: Activation-aware Weight Quantization for LLM Compression and "
    "Acceleration.” arXiv:2306.00978. https://doi.org/10.48550/arXiv.2306.00978.",
    "Luccioni, Sasha, Yacine Jernite, and Emma Strubell. 2024. “Power Hungry Processing: Watts Driving the Cost "
    "of AI Deployment?” In *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency "
    "(FAccT ’24)*, 85–99. https://doi.org/10.1145/3630106.3658542.",
    "NVIDIA. n.d. “GeForce RTX 4070 Family.” Accessed November 24, 2026. "
    "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/.",
    "Ollama. 2026. *Ollama*. Version 0.40.16. https://github.com/ollama/ollama.",
    "Pan, Guanzhong, Vishal Chodnekar, Abinas Roy, and Haibo Wang. 2025. “A Cost-Benefit Analysis of On-Premise "
    "Large Language Model Deployment: Breaking Even with Commercial LLM Services.” arXiv:2509.18101. "
    "https://doi.org/10.48550/arXiv.2509.18101.",
    "Peng, Sheng-Wei, Yi-Hsun Lin, and Yi-Pei Lee. 2026. “Inference Economics of Enterprise Coding Agents: A Case "
    "Study of Cloud vs. On-Premise LLMs.” arXiv:2607.13080. https://doi.org/10.48550/arXiv.2607.13080.",
    "Pope, Reiner, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan "
    "Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2022. “Efficiently Scaling Transformer Inference.” "
    "arXiv:2211.05102. https://doi.org/10.48550/arXiv.2211.05102.",
    "Samsi, Siddharth, Dan Zhao, Joseph McDonald, Baolin Li, Adam Michaleas, Michael Jones, William Bergeron, "
    "Jeremy Kepner, Devesh Tiwari, and Vijay Gadepally. 2023. “From Words to Watts: Benchmarking the Energy "
    "Costs of Large Language Model Inference.” arXiv:2310.03003. https://doi.org/10.48550/arXiv.2310.03003.",
    "Williams, Samuel, Andrew Waterman, and David Patterson. 2009. “Roofline: An Insightful Visual Performance "
    "Model for Multicore Architectures.” *Communications of the ACM* 52 (4): 65–76. "
    "https://doi.org/10.1145/1498765.1498785.",
    "Wilson, Edwin B. 1927. “Probable Inference, the Law of Succession, and Statistical Inference.” *Journal of "
    "the American Statistical Association* 22 (158): 209–12. https://doi.org/10.1080/01621459.1927.10502953.",
    "Yang, An, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. 2025. “Qwen3 "
    "Technical Report.” arXiv:2505.09388. https://doi.org/10.48550/arXiv.2505.09388.",
    "Yuan, Zhihang, Yuzhang Shang, Yang Zhou, Zhen Dong, Zhe Zhou, Chenhao Xue, Bingzhe Wu, et al. 2024. “LLM "
    "Inference Unveiled: Survey and Roofline Model Insights.” arXiv:2402.16363. "
    "https://doi.org/10.48550/arXiv.2402.16363.",
]
assert REFS == sorted(REFS), "reference list must be alphabetical"
for ref in REFS:
    p = para(ref, indent=False)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.5)

page_break()
h1("7. Appendices")
table(1, "Test Setup. The same tasks, prompt, checker and output limit were used for all three models.",
      ["Item", "Value"],
      [["Computer", "AMD Ryzen 7 5700X3D, 32 GB RAM, Windows 11 Pro"],
       ["Graphics card", f"NVIDIA GeForce RTX 4070, 12 GB GDDR6X, 192-bit, {bandwidth:.0f} GB/s"],
       ["Local model", f"qwen3:8b, Q4_K_M, {weights_gb:.2f} GB, Ollama 0.40.16, temperature 0, seed 42"],
       ["API models", "Claude Haiku 5.5, Claude Sonnet 5.5, default temperature"],
       ["Reasoning mode", "off for all models"],
       ["Tasks", "GSM8K test, problems 1–200, one answer per task"],
       ["Output limit", "4,096 tokens"],
       ["Energy", "nvidia-smi, every 100 ms, minus idle power (graphics card only)"],
       ["API prices", "Haiku $0.10 / $0.50, Sonnet $2 / $10 per million input / output tokens"]],
      [1.5, 4.7], text_table=True)


def ci(r: dict[str, str]) -> str:
    return f"{f(r, 'acc_ci_lo') * 100:.1f}–{f(r, 'acc_ci_hi') * 100:.1f}"


table(2, "Results per Model on 200 GSM8K Tasks. Cost is per task at standard prices; energy is for the graphics "
         "card above idle.",
      ["Model", "Correct", "95% CI, %", "Time, s", "Output tokens", "Cost, $", "Energy, Wh"],
      [["Claude Haiku 5.5", f"{H['n_correct']}/200", ci(H), f"{f(H, 't_task_s'):.2f}",
        f"{f(H, 'out_tokens_mean'):.0f}", f"{f(H, 'cost_usd_mean'):.7f}", "—"],
       ["Claude Sonnet 5.5", f"{S['n_correct']}/200", ci(S), f"{f(S, 't_task_s'):.2f}",
        f"{f(S, 'out_tokens_mean'):.0f}", f"{f(S, 'cost_usd_mean'):.7f}", "—"],
       ["qwen3:8b (local)", f"{Q['n_correct']}/200", ci(Q), f"{t_q:.2f}",
        f"{f(Q, 'out_tokens_mean'):.0f}", "—", f"{E_wh:.3f}"]],
      [1.5, 0.75, 0.9, 0.7, 0.9, 0.85, 0.75])


def vfmt(api: str, pricing: str, life: int, p: float) -> str:
    r = be(api, pricing, life, p)
    v = float(r["v_star_per_day"])
    if math.isinf(v):
        return "never"
    return n0(v) + ("" if r["feasible"] == "True" else "*")


body = []
for api, label in (("sonnet-5.5", "Sonnet"), ("haiku-5.5", "Haiku")):
    for pricing in ("standard", "batch"):
        for life in (3, 4, 5):
            body.append([f"{label}, {pricing}, {life} years"] + [vfmt(api, pricing, life, p) for p in (0.10, 0.20, 0.35)])
table(3, f"Break-Even Volume in Tasks per Day for a $2,000 PC. * means above the PC's limit of {n0(v_max)} tasks "
         "per day, so the PC cannot pay off; “never” means the local electricity alone costs more than the API.",
      ["API model, prices, PC lifetime", "$0.10/kWh", "$0.20/kWh", "$0.35/kWh"], body, [2.8, 1.1, 1.1, 1.1])

now = dt.datetime(2026, 11, 24, 12, 0, 0)
cp = doc.core_properties
cp.author = cp.last_modified_by = AUTHOR
cp.title = TITLE
cp.subject = "Architecture of Computers (CII3SE003CI)"
cp.keywords = "large language models; inference cost; local deployment"
cp.comments = cp.category = cp.content_status = cp.identifier = cp.language = cp.version = ""
cp.created = cp.modified = now
cp.revision = 1

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)


def clean_package(path: Path) -> None:
    import zipfile

    app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
           'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
           '<Template>Normal.dotm</Template><TotalTime>0</TotalTime><Application>Microsoft Office Word</Application>'
           '<DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company>'
           '<LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged>'
           '<AppVersion>16.0000</AppVersion></Properties>')
    drop = re.compile(r"^(customXml/|docProps/thumbnail)")
    rel = re.compile(r'<Relationship [^>]*Target="[^"]*(?:thumbnail|customXml)[^"]*"[^>]*/>')
    ovr = re.compile(r'<Override PartName="/(?:customXml|docProps/thumbnail)[^"]*"[^>]*/>')
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if drop.match(item.filename) or item.filename.endswith("/"):
                continue
            data = zin.read(item.filename)
            if item.filename == "docProps/app.xml":
                data = app.encode()
            elif item.filename.endswith(".rels"):
                data = rel.sub("", data.decode("utf-8")).encode()
            elif item.filename == "[Content_Types].xml":
                data = ovr.sub("", data.decode("utf-8")).encode()
            info = zipfile.ZipInfo(item.filename, date_time=now.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(info, data)
    tmp.replace(path)


clean_package(OUT)


def words() -> tuple[int, int]:
    texts = [p.text for p in doc.paragraphs]
    i0 = texts.index("Abstract")
    i1 = texts.index("1. Introduction")
    i2 = texts.index("Data and Code Availability")
    count = lambda ts: sum(len(t.split()) for t in ts)
    return count(texts[i0 + 1:i1]), count(t for t in texts[i1:i2] if not t.startswith("Figure "))


ab, main = words()
print(f"saved {OUT}\nabstract {ab} words, main text {main} words")
sys.exit(0)
