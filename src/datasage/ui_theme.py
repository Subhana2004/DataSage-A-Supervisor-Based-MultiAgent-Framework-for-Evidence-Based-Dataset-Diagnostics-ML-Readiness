"""Visual system for DataSage Studio.

Static markup and CSS only; never interpolate uploaded dataset content here.
"""
from __future__ import annotations

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root { --ds-bg:#080e1a; --ds-surface:#111b2c; --ds-line:#25334b; --ds-text:#f3f7ff; --ds-muted:#a2b3ce; --ds-accent:#59e2cd; --ds-blue:#77a8ff; }
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: radial-gradient(ellipse 820px 450px at 74% -8%,rgba(65,117,192,.14),transparent 69%),
              radial-gradient(ellipse 700px 420px at -14% 38%,rgba(45,194,171,.07),transparent 72%),
              var(--ds-bg) !important;
  color: var(--ds-text);
  font-family:'DM Sans',system-ui,sans-serif;
}
[data-testid="stHeader"] {background:transparent !important;}
[data-testid="stToolbar"] {visibility:hidden;}
[data-testid="stDecoration"] {display:none;}
[data-testid="stMainBlockContainer"], .block-container {max-width:1250px; padding-top:1.25rem; padding-bottom:4rem;}
[data-testid="stSidebar"] {background:#0d1524 !important; border-right:1px solid #1c2a42;}
[data-testid="stSidebar"] > div {padding-top:1.35rem;}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p {color:#e8f0ff;}
h1,h2,h3,h4 {font-family:'Space Grotesk',system-ui,sans-serif; letter-spacing:-.025em; color:var(--ds-text);}
h2 {font-size:1.35rem !important;}
p, li, label, [data-testid="stMarkdownContainer"] {color:#d6e2f6;}
small, .stCaption, [data-testid="stCaptionContainer"] {color:var(--ds-muted) !important;}
[data-testid="stWidgetLabel"] p {color:#c9d9ef !important; font-weight:600;}
[data-testid="stFileUploader"] {background:#0e192b;border:1px dashed #49758d;border-radius:20px;padding:.65rem;}
[data-testid="stFileUploaderDropzone"] {background:#101d31;border-radius:14px;border:0;}
[data-testid="stFileUploaderDropzone"] button {border-color:#4c899a;}
[data-testid="stDataFrame"], [data-testid="stTable"] {border-radius:15px;overflow:hidden;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,rgba(21,37,59,.98),rgba(14,25,43,.96));border:1px solid var(--ds-line);padding:1.2rem 1.35rem;border-radius:18px;box-shadow:0 12px 30px rgba(0,0,0,.09)}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {color:var(--ds-muted);}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif;color:#f8fbff;}
div[data-testid="stExpander"] {background:rgba(16,28,46,.92);border:1px solid var(--ds-line);border-radius:16px;overflow:hidden;margin-bottom:.45rem;}
div[data-testid="stExpander"] summary {color:#ebf2ff;font-weight:650;}
div[data-testid="stTabs"] button {font-weight:650;white-space:nowrap;}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:.45rem;border-bottom:1px solid #253550;}
div[data-testid="stTabs"] [data-baseweb="tab"] {background:#111d30;border-radius:11px 11px 0 0;padding:.7rem 1rem;}
div[data-testid="stTabs"] [aria-selected="true"] {background:rgba(54,150,148,.2);color:#8affdd !important;}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background:#59e2cd;}
.stButton button, [data-testid="stDownloadButton"] button {border:1px solid #3b5676;border-radius:12px;padding:.65rem 1.15rem;font-weight:700;transition:transform .16s ease,border-color .16s ease,background .16s ease;}
.stButton button:hover, [data-testid="stDownloadButton"] button:hover {transform:translateY(-1px);border-color:#69d9ce;color:#d9fff6;}
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
 background:linear-gradient(110deg,#61e8ce,#75acff);border:0;color:#081827;box-shadow:0 9px 24px rgba(50,198,180,.2)
}
.stButton button[kind="primary"]:hover {color:#081827;background:linear-gradient(110deg,#89fbe2,#91beff);}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea {background-color:#121e31 !important;border-color:#344762 !important;border-radius:11px !important;}
[data-testid="stAlert"] {border-radius:13px;}
[data-testid="stSelectbox"] {margin-bottom:.3rem;}
[data-testid="stHorizontalBlock"] {align-items:start;}
hr {border-color:var(--ds-line)!important;}
.ds-hero {border:1px solid #29435a;position:relative;overflow:hidden;padding:2rem 2.3rem 1.9rem;border-radius:25px;
  background:radial-gradient(ellipse 510px 400px at 84% 22%,rgba(52,174,183,.18),transparent 66%),linear-gradient(120deg,#15253b,#0f1c30 64%,#14273d);margin-bottom:1.3rem;}
.ds-hero:after {content:'';position:absolute;width:210px;height:210px;border:1px solid rgba(105,230,216,.22);border-radius:50%;right:5%;top:-24%;box-shadow:0 0 0 35px rgba(84,165,207,.06),0 0 0 76px rgba(91,190,203,.035);pointer-events:none;}
.ds-overline {font-size:.75rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:#78f0dd!important;}
.ds-hero h1 {font-size:clamp(2rem,4vw,3.25rem)!important;line-height:1.13;max-width:720px;margin:.55rem 0 .6rem;color:#fff;}
.ds-hero h1 span {color:#73e6d1;}
.ds-hero p {max-width:680px;color:#b9cde3!important;font-size:1.02rem;line-height:1.7;margin:0;}
.ds-tags {display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1.25rem;}
.ds-tag {font-size:.76rem;font-weight:700;color:#b9f6ec;border:1px solid #426c78;background:rgba(48,156,148,.11);border-radius:999px;padding:.35rem .75rem;}
.ds-brand {display:flex;gap:.75rem;align-items:center;padding:.6rem .3rem 1.3rem;border-bottom:1px solid #24334a;margin-bottom:1.25rem;}
.ds-logo {height:39px;width:39px;display:grid;place-items:center;border-radius:12px;background:linear-gradient(150deg,#68e7cb,#6b9fff);color:#081827;font-size:1.35rem;font-family:'Space Grotesk',sans-serif;font-weight:800;box-shadow:0 5px 22px rgba(55,185,179,.16)}
.ds-brand strong {font-family:'Space Grotesk',sans-serif;font-size:1.17rem;display:block;line-height:1.1;}
.ds-brand small {font-size:.73rem;color:#9db1ce;}
.ds-side-label {letter-spacing:.13em;text-transform:uppercase;color:#718cae!important;font-size:.7rem;font-weight:800;margin:1.35rem .3rem .6rem;}
.ds-step {display:flex;align-items:center;gap:.75rem;padding:.62rem .5rem;color:#c6d8ef;font-size:.83rem;border-radius:11px;}
.ds-step span {height:25px;width:25px;display:grid;place-items:center;background:#1b3146;border:1px solid #38556c;border-radius:8px;color:#77eedc;font-weight:700;font-size:.75rem;}
.ds-tip {border:1px solid #2f4a5f;background:#102332;border-radius:14px;padding:1rem;color:#b2d4de;font-size:.82rem;line-height:1.55;}
.ds-section {display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:1.3rem 0 .65rem}
.ds-section h2 {font-size:1.3rem;margin:0;}
.ds-section p {font-size:.85rem;color:#90a7c4;margin:.15rem 0 0;}
.ds-kicker {font-size:.68rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:#72e2d2;}
.ds-note {padding:.9rem 1rem;background:#132338;border:1px solid #2b455e;border-radius:13px;color:#abc5dc;font-size:.86rem;line-height:1.55;margin:.5rem 0 1rem;}
.ds-note strong {color:#f7fcff;}
.ds-result-badge {display:inline-block;font-size:.72rem;padding:.22rem .55rem;border-radius:7px;background:#1c4051;color:#97f0df;margin-right:.45rem;}
.ds-result-badge.warn {background:#493827;color:#ffcd97;}
.ds-result-badge.pending {background:#293b56;color:#bbd4ff;}
.ds-foot {border-top:1px solid #26334a;padding-top:1rem;margin-top:2.2rem;font-size:.76rem;color:#8196b4;text-align:center;}
@media(max-width:780px) {
 .ds-hero {padding:1.45rem 1.3rem;}
 .ds-hero:after {right:-95px;}
 .ds-hero h1 {font-size:2.1rem!important;}
 .block-container {padding-left:1rem;padding-right:1rem;}
 [data-testid="stSidebar"] {max-width:290px;}
}
@media(prefers-reduced-motion:reduce) { .stButton button, [data-testid="stDownloadButton"] button {transition:none!important; transform:none!important;} }
</style>
"""


def install(st) -> None:
    st.markdown(STYLE, unsafe_allow_html=True)


def hero(st) -> None:
    st.markdown(
        '<div class="ds-hero"><div class="ds-overline">DATA INTELLIGENCE / STUDIO 01</div>'
        '<h1>Understand your data.<br><span>Trust the evidence.</span></h1>'
        '<p>Five diagnostic agents. One evidence-first workflow. '
        'Discover suspicious patterns, investigate their context, and export auditable findings.</p>'
        '<div class="ds-tags"><span class="ds-tag">◆ Evidence-first</span>'
        '<span class="ds-tag">◇ Read-only analysis</span>'
        '<span class="ds-tag">↗ Adaptive investigations</span></div></div>',
        unsafe_allow_html=True,
    )


def sidebar(st, public_demo: bool) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="ds-brand"><div class="ds-logo">D</div>'
            '<div><strong>DataSage</strong><small>DATASET HEALTH STUDIO</small></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ds-side-label">YOUR WORKFLOW</div>', unsafe_allow_html=True)
        for number, label in [
            ("01", "Connect a dataset"),
            ("02", "Configure analysis"),
            ("03", "Explore evidence"),
            ("04", "Export findings"),
        ]:
            st.markdown(
                f'<div class="ds-step"><span>{number}</span>{label}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="ds-side-label">RESEARCH NOTES</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ds-tip"><strong>Verification ≠ causation.</strong><br>'
            'Numerical verification confirms a calculation, not that a dataset '
            'is faulty, biased, or leaking labels.</div>',
            unsafe_allow_html=True,
        )
        if public_demo:
            st.caption("PUBLIC DEMO · Small CSVs only · No server-side saved reports")
        else:
            st.caption("LOCAL STUDIO · Optional report history and AI explanations")
        st.markdown(
            '<div class="ds-foot">Built for reproducible dataset investigations</div>',
            unsafe_allow_html=True,
        )
