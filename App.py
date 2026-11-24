import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Smart Shopping Analysis", page_icon="🛒", layout="wide")

MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COBALT, INK, YELLOW = "#1B2A6B", "#14213D", "#FFC83D"
GREEN, RED, LINE = "#1FA37A", "#E5383B", "#D5DBE6"

# =====================================================================
# Look and feel: cobalt + price-tag yellow, receipt slip for the results
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Instrument+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Instrument Sans', system-ui, sans-serif; }
.stApp { background: #EEF1F6; color: #14213D; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }

/* sidebar */
section[data-testid="stSidebar"] { background: #1B2A6B; }
section[data-testid="stSidebar"] * { color: #EAF0FF !important; }
section[data-testid="stSidebar"] h1 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.5rem; }
section[data-testid="stSidebar"] [role="radiogroup"] label { padding: 6px 0; }

/* hero band with a price tag */
.hero { background: #1B2A6B; color: #fff; border-radius: 14px; padding: 30px 34px; margin-bottom: 22px;
        display: flex; justify-content: space-between; align-items: center; gap: 24px; }
.hero h1 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.5rem;
           line-height: 1.05; margin: 0 0 8px 0; color: #fff; letter-spacing: -0.02em; }
.hero p { margin: 0; color: #C9D3F5; font-size: 1.02rem; max-width: 520px; }
.tag { background: #FFC83D; color: #14213D; padding: 14px 22px 14px 34px; min-width: 150px; position: relative;
       clip-path: polygon(16px 0, 100% 0, 100% 100%, 16px 100%, 0 50%); flex-shrink: 0; }
.tag::before { content: ""; position: absolute; left: 13px; top: 50%; width: 9px; height: 9px;
               margin-top: -4.5px; border-radius: 50%; background: #1B2A6B; }
.tag small { display: block; font-size: .78rem; font-weight: 600; opacity: .75; }
.tag b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 800; }

/* stat strip: one panel, hairline dividers */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); background: #fff; border-radius: 12px; margin-bottom: 22px; }
.stat { padding: 20px 24px; border-left: 1px solid #D5DBE6; }
.stat:first-child { border-left: 0; }
.stat .n { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2rem; color: #1B2A6B; }
.stat .l { color: #5B6682; font-size: .92rem; }

/* receipt slip */
.receipt { background: #fff; font-family: 'IBM Plex Mono', monospace; font-size: .88rem; padding: 22px 22px 26px;
           border-radius: 4px 4px 0 0; position: relative; margin-bottom: 22px; box-shadow: 0 2px 0 #D5DBE6; }
.receipt::after { content: ""; position: absolute; left: 0; right: 0; bottom: -9px; height: 9px;
                  background: linear-gradient(135deg, #fff 5px, transparent 0), linear-gradient(225deg, #fff 5px, transparent 0);
                  background-size: 14px 9px; }
.receipt .rt { font-weight: 600; text-align: center; padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px dashed #9AA3BA; }
.receipt .ln { display: flex; justify-content: space-between; gap: 12px; padding: 3px 0; }
.receipt .sep { border-top: 1px dashed #9AA3BA; margin: 12px 0; }
.receipt .big { font-family: 'Bricolage Grotesque', sans-serif; font-size: 3rem; font-weight: 800; color: #1B2A6B; line-height: 1; margin: 6px 0 10px; }
.receipt .meter { height: 10px; background: #E3E8F2; border-radius: 2px; margin-bottom: 10px; }
.receipt .meter i { display: block; height: 100%; background: #1B2A6B; }
.receipt .pos { color: #1FA37A; font-weight: 600; }
.receipt .neg { color: #E5383B; font-weight: 600; }
.stamp { display: inline-block; margin-top: 14px; padding: 4px 14px; border: 3px solid; font-weight: 600;
         letter-spacing: .06em; transform: rotate(-4deg); }
.stamp.buy { color: #1FA37A; } .stamp.skip { color: #E5383B; }

/* insights rows */
.find { display: flex; gap: 22px; align-items: baseline; padding: 16px 0; border-bottom: 1px solid #D5DBE6; }
.find .big { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2rem; color: #1B2A6B; min-width: 120px; }
.act { padding: 8px 0 8px 22px; position: relative; }
.act::before { content: ""; position: absolute; left: 0; top: 16px; width: 9px; height: 9px; background: #FFC83D; }
h2, h3 { font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.01em; }

/* buttons */
.stButton > button { background: #1B2A6B; color: #fff; border: 0; border-radius: 8px; padding: .6rem 1.6rem; font-weight: 600; }
.stButton > button:hover { background: #FFC83D; color: #14213D; }

@media (max-width: 760px) {
  .hero { flex-direction: column; align-items: flex-start; } .hero h1 { font-size: 1.9rem; }
  .stats { grid-template-columns: 1fr 1fr; } .stat:nth-child(3) { border-left: 0; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
# Small UI helpers
# =====================================================================
def hero(title, sub, tag_label, tag_value):
    st.markdown(f'<div class="hero"><div><h1>{title}</h1><p>{sub}</p></div>'
                f'<div class="tag"><small>{tag_label}</small><b>{tag_value}</b></div></div>',
                unsafe_allow_html=True)


def stats(items):
    cells = "".join(f'<div class="stat"><span class="n">{n}</span><span class="l">{l}</span></div>'
                    for n, l in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Instrument Sans, sans-serif", color=INK),
                      title=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=17), x=0),
                      legend_title_text="")
    fig.update_xaxes(showgrid=False, linecolor=LINE, title_text="")
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


def ln(a, b, cls=""):
    return f'<div class="ln"><span>{a}</span><span class="{cls}">{b}</span></div>'


# =====================================================================
# Loading (model + data)
# =====================================================================
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.joblib")
    info = joblib.load("model_info.joblib")
    return model, info["columns"], info["defaults"]


@st.cache_data
def load_data():
    d = pd.read_csv("shoppers_clean.csv")
    d["Revenue"] = d["Revenue"].astype(int)
    return d


model, columns, defaults = load_model()
df = load_data()


def build_row(v):
    """Turn the form inputs into the exact columns the model was trained on."""
    row = {c: defaults[c] for c in columns}
    row.update({
        "PageValues": v["page_values"], "HasPageValue": int(v["page_values"] > 0),
        "ExitRates": v["exit_rates"], "BounceRates": v["bounce_rates"],
        "ProductRelated": v["product_pages"], "ProductRelated_Duration": v["product_duration"],
        "Administrative": v["admin_pages"], "Administrative_Duration": v["admin_duration"],
        "Weekend": int(v["weekend"]),
    })
    for c in columns:
        if c.startswith("Month_") or c.startswith("VisitorType_"):
            row[c] = 0
    if f"Month_{v['month']}" in columns:
        row[f"Month_{v['month']}"] = 1
    if f"VisitorType_{v['visitor']}" in columns:
        row[f"VisitorType_{v['visitor']}"] = 1
    return pd.DataFrame([row])[columns]


# =====================================================================
# Navigation
# =====================================================================
st.sidebar.title("Smart Shopping")
page = st.sidebar.radio("Pages", ["Dashboard overview", "Purchase prediction",
                                  "Why this prediction?", "Customer insights"],
                        label_visibility="collapsed")
st.sidebar.caption("Model: tuned XGBoost")


# =====================================================================
# Page 1: Dashboard overview
# =====================================================================
if page == "Dashboard overview":
    hero("Who buys, and when?", "Every session in the shop, filtered by visitor type and month.",
         "conversion", f"{df['Revenue'].mean():.1%}")

    avail = [m for m in MONTH_ORDER if m in df["Month"].unique()]
    f1, f2 = st.columns(2)
    visitors = f1.multiselect("Visitor type", sorted(df["VisitorType"].unique()),
                              default=sorted(df["VisitorType"].unique()))
    months = f2.multiselect("Month", avail, default=avail)

    d = df[df["VisitorType"].isin(visitors) & df["Month"].isin(months)]
    if d.empty:
        st.warning("No sessions match these filters. Add a visitor type or a month.")
        st.stop()

    stats([(f"{len(d):,}", "Total sessions"), (f"{int(d['Revenue'].sum()):,}", "Purchases"),
           (f"{d['Revenue'].mean():.1%}", "Purchase conversion rate"),
           (f"{(d['VisitorType'] == 'Returning_Visitor').mean():.1%}", "Returning visitors")])

    monthly = d.groupby("Month")["Revenue"].agg(["mean", "count"]).reset_index()
    monthly["Month"] = pd.Categorical(monthly["Month"], MONTH_ORDER, ordered=True)
    monthly = monthly.sort_values("Month")
    monthly["Conversion %"] = monthly["mean"] * 100

    c1, c2 = st.columns(2)
    fig = px.line(monthly, x="Month", y="Conversion %", markers=True, title="Conversion rate by month (%)")
    fig.update_traces(line_color=COBALT, marker=dict(color=YELLOW, size=10, line=dict(color=COBALT, width=2)))
    c1.plotly_chart(style(fig))
    fig = px.bar(monthly, x="Month", y="count", title="Sessions per month")
    fig.update_traces(marker_color=COBALT)
    fig.update_yaxes(title_text="")
    c2.plotly_chart(style(fig))

    c3, c4 = st.columns(2)
    bv = d.groupby("VisitorType")["Revenue"].mean().mul(100).reset_index()
    bv.columns = ["VisitorType", "Conversion %"]
    fig = px.bar(bv, x="VisitorType", y="Conversion %", title="Conversion rate by visitor type (%)")
    fig.update_traces(marker_color=COBALT)
    c3.plotly_chart(style(fig))

    g = d.assign(Group=np.where(d["PageValues"] > 0, "PageValues above zero", "PageValues zero"))
    bp = g.groupby("Group")["Revenue"].mean().mul(100).reset_index()
    bp.columns = ["Group", "Conversion %"]
    fig = px.bar(bp, x="Group", y="Conversion %", color="Group", title="PageValues zero vs above zero (%)",
                 color_discrete_map={"PageValues zero": "#9AA3BA", "PageValues above zero": YELLOW})
    fig.update_layout(showlegend=False)
    c4.plotly_chart(style(fig))


# =====================================================================
# Page 2: Purchase prediction
# =====================================================================
elif page == "Purchase prediction":
    hero("Will this visitor buy?", "Describe the session, press Predict, and read the slip.",
         "model", "XGBoost")

    left, right = st.columns([3, 2], gap="large")
    with left:
        a, b = st.columns(2)
        with a:
            page_values = st.number_input("PageValues", 0.0, 500.0, 0.0, step=1.0,
                                          help="Average value of the pages the visitor viewed")
            exit_rates = st.number_input("ExitRates", 0.0, 0.2, 0.02, step=0.005, format="%.3f")
            bounce_rates = st.number_input("BounceRates", 0.0, 0.2, 0.01, step=0.005, format="%.3f")
            product_pages = st.number_input("Product pages viewed", 0, 800, 20)
            product_duration = st.number_input("Seconds on product pages", 0.0, 70000.0, 600.0, step=50.0)
        with b:
            admin_pages = st.number_input("Administrative pages viewed", 0, 30, 2)
            admin_duration = st.number_input("Seconds on administrative pages", 0.0, 4000.0, 50.0, step=10.0)
            month = st.selectbox("Month", MONTH_ORDER, index=8)
            visitor = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
            weekend = st.checkbox("Weekend visit")
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Lower catches more buyers but raises false alarms")
        go = st.button("Predict", type="primary")

    if go:
        X_new = build_row(dict(page_values=page_values, exit_rates=exit_rates, bounce_rates=bounce_rates,
                               product_pages=product_pages, product_duration=product_duration,
                               admin_pages=admin_pages, admin_duration=admin_duration,
                               month=month, visitor=visitor, weekend=weekend))
        st.session_state["last_X"] = X_new
        st.session_state["last_proba"] = float(model.predict_proba(X_new)[0, 1])
        st.session_state["last_lines"] = [("PageValues", f"{page_values:g}"), ("Exit rate", f"{exit_rates:.3f}"),
                                          ("Bounce rate", f"{bounce_rates:.3f}"), ("Product pages", product_pages),
                                          ("Month", month), ("Visitor", visitor.replace("_", " ")),
                                          ("Weekend", "yes" if weekend else "no")]

    with right:
        if "last_proba" not in st.session_state:
            st.info("No prediction yet. Fill in the session and press Predict.")
        else:
            p = st.session_state["last_proba"]
            buy = p >= threshold
            items = "".join(ln(a, b) for a, b in st.session_state["last_lines"])
            st.markdown(
                '<div class="receipt"><div class="rt">Session slip</div>' + items +
                '<div class="sep"></div><div>Purchase probability</div>'
                f'<div class="big">{p:.1%}</div><div class="meter"><i style="width:{p*100:.0f}%"></i></div>'
                + ln("Decision threshold", f"{threshold:.0%}") +
                f'<div class="stamp {"buy" if buy else "skip"}">{"WILL BUY" if buy else "WON\'T BUY"}</div></div>',
                unsafe_allow_html=True)
            st.caption("A model estimate, not a guaranteed probability. Page 3 shows the reasons behind it.")


# =====================================================================
# Page 3: Why this prediction?
# =====================================================================
elif page == "Why this prediction?":
    hero("Why did the model say that?", "The factors that moved the last prediction up or down.",
         "method", "Per visitor")

    if "last_X" not in st.session_state:
        st.info("Run a prediction on the Purchase prediction page first, then come back here.")
        st.stop()

    X_new, p = st.session_state["last_X"], st.session_state["last_proba"]
    contrib = pd.Series(model.get_booster().predict(xgb.DMatrix(X_new), pred_contribs=True)[0][:-1], index=columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)

    left, right = st.columns([2, 3], gap="large")
    with left:
        rows = "".join(ln(f"{f} = {X_new.iloc[0][f]:g}", f"{'+' if v >= 0 else '−'}{abs(v):.2f}",
                          "pos" if v >= 0 else "neg") for f, v in top.items())
        st.markdown('<div class="receipt"><div class="rt">What moved the score</div>' + rows +
                    '<div class="sep"></div>' + ln("Final chance to buy", f"{p:.1%}") + '</div>',
                    unsafe_allow_html=True)
    with right:
        plot_df = pd.DataFrame({"Feature": top.index, "Effect": top.values,
                                "Direction": np.where(top.values >= 0, "Raises the chance", "Lowers the chance")}).iloc[::-1]
        fig = px.bar(plot_df, x="Effect", y="Feature", color="Direction", orientation="h",
                     title="Strongest factors for this visitor",
                     color_discrete_map={"Raises the chance": GREEN, "Lowers the chance": RED})
        st.plotly_chart(style(fig, 420))
    st.caption("Bar length shows how strongly a factor pushed this visitor's score. Units are model log-odds, "
               "so compare sizes and directions rather than reading exact numbers.")


# =====================================================================
# Page 4: Customer insights
# =====================================================================
else:
    hero("What the data says", "Findings from every session, and what the shop can do about them.",
         "sessions", f"{len(df):,}")

    overall = df["Revenue"].mean()
    pv_yes = df.loc[df["PageValues"] > 0, "Revenue"].mean()
    pv_no = df.loc[df["PageValues"] == 0, "Revenue"].mean()
    bvis = df.groupby("VisitorType")["Revenue"].mean()
    bmon = df.groupby("Month")["Revenue"].mean().sort_values(ascending=False)
    bounce = df.groupby("Revenue")["BounceRates"].mean()
    exitr = df.groupby("Revenue")["ExitRates"].mean()

    finds = [(f"{overall:.1%}", "of sessions end in a purchase, so most visitors leave without buying."),
             (f"{pv_yes:.1%}", f"is the conversion rate when PageValues is above zero, against {pv_no:.1%} when it is zero.")]
    if {"Returning_Visitor", "New_Visitor"} <= set(bvis.index):
        finds.append((f"{bvis['New_Visitor']:.1%}", f"is the conversion rate of new visitors; returning visitors convert at {bvis['Returning_Visitor']:.1%}."))
    finds += [(str(bmon.index[0]), f"is the best month at {bmon.iloc[0]:.1%}; {bmon.index[-1]} is the weakest at {bmon.iloc[-1]:.1%}."),
              (f"{bounce[1]:.3f}", f"is the average bounce rate of buyers, against {bounce[0]:.3f} for non-buyers. Exit rates follow the same pattern ({exitr[1]:.3f} vs {exitr[0]:.3f}).")]

    st.subheader("Key findings")
    st.markdown("".join(f'<div class="find"><span class="big">{b}</span><span>{t}</span></div>' for b, t in finds),
                unsafe_allow_html=True)

    st.subheader("What the shop can do")
    acts = ["Send visitors toward pages that carry value, for example by promoting them on the home and category pages.",
            "Cut early exits by making landing and product pages clearer and faster.",
            "Plan campaigns and stock around the high-converting months.",
            "Use the model to spot likely buyers, and offer a discount to hesitant ones. Lower the threshold when missing a buyer costs more than a wasted offer."]
    st.markdown("".join(f'<div class="act">{a}</div>' for a in acts), unsafe_allow_html=True)

    st.subheader("About the model")
    st.write("The final model is a tuned XGBoost classifier. On the held-out test set it reached an F1 of about 0.68 "
             "for buyers and a ROC-AUC of about 0.94. PageValues and HasPageValue were the most important features.")