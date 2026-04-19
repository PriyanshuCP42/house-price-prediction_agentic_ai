


import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from io import BytesIO

st.set_page_config(page_title="🏠 House Price Predictor", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card { background:white; border-radius:14px; padding:22px 20px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08); text-align:center;
    border-top:4px solid #3b82f6; margin-bottom:10px; }
.metric-card-green  { border-top-color:#10b981 !important; }
.metric-card-orange { border-top-color:#f59e0b !important; }
.metric-card-red    { border-top-color:#ef4444 !important; }
.metric-label { font-size:11px; font-weight:700; text-transform:uppercase;
    color:#6b7280; letter-spacing:1px; }
.metric-value { font-size:30px; font-weight:800; color:#1e293b; margin:8px 0 4px; }
.metric-sub { font-size:12px; color:#94a3b8; }
.badge { display:inline-block; padding:5px 16px; border-radius:20px; font-size:14px; font-weight:700; }
.badge-fair  { background:#d1fae5; color:#065f46; }
.badge-over  { background:#fee2e2; color:#991b1b; }
.badge-under { background:#dbeafe; color:#1e40af; }
.section-title { font-size:18px; font-weight:700; color:#1e293b;
    padding:10px 0 6px; border-bottom:2px solid #e2e8f0; margin-bottom:14px; }
.stButton>button { background:linear-gradient(135deg,#3b82f6,#1d4ed8); color:white;
    border-radius:10px; font-weight:600; padding:12px 24px; border:none;
    font-size:15px; width:100%; }
#MainMenu { visibility:hidden; } footer { visibility:hidden; }
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    df = pd.read_csv('kc_house_data.csv')
    df = df[df['bedrooms'] <= 10]
    df['house_age']      = datetime.datetime.now().year - df['yr_built']
    df['renovated']      = df['yr_renovated'].apply(lambda x: 0 if x==0 else 1)
    df['amenity_score']  = df['waterfront'] + df['view'] + df['condition'] + df['grade']
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    return df

try:
    artifacts     = load_model()
    model         = artifacts['model']
    scaler        = artifacts['scaler']
    feature_names = artifacts['feature_names']
    zipcode_mean  = artifacts['zipcode_mean']
    current_year  = artifacts['current_year']
    df_full       = load_dataset()
    MODEL_OK      = True
except FileNotFoundError:
    MODEL_OK = False

def predict_price(input_dict):
    feature_frame = pd.DataFrame([[input_dict.get(f, 0) for f in feature_names]], columns=feature_names)
    scaled = scaler.transform(feature_frame)
    tree_preds = np.array([t.predict(scaled)[0] for t in model.estimators_])
    price = tree_preds.mean()
    std   = tree_preds.std()
    conf  = max(0, min(100, 100-(std/price*100)))
    return price, max(0, price-std), price+std, conf, std

def market_status(price, sqft):
    ppsf = price/sqft
    if ppsf < 180:   return "Underpriced 🔵","badge-under","#3b82f6",ppsf
    elif ppsf < 290: return "Fair Price 🟢", "badge-fair", "#10b981",ppsf
    else:            return "Overpriced 🔴", "badge-over", "#ef4444",ppsf

def investment_score(price, sqft, grade, condition, house_age, waterfront, view, renovated):
    score=50; ppsf=price/sqft
    if ppsf<150: score+=25
    elif ppsf<200: score+=15
    elif ppsf<250: score+=5
    elif ppsf>320: score-=20
    if grade>=10: score+=12
    elif grade>=8: score+=6
    elif grade<=5: score-=10
    if condition>=4: score+=8
    elif condition<=2: score-=8
    if house_age<=15: score+=10
    elif house_age>60: score-=12
    if waterfront: score+=15
    if view>=3: score+=8
    elif view>=1: score+=4
    if renovated: score+=6
    return max(0, min(100, score))

def inv_label(score):
    if score>=75: return "Strong Buy ✅","#10b981"
    elif score>=60: return "Good Buy 🟢","#22c55e"
    elif score>=45: return "Moderate 🟡","#f59e0b"
    elif score>=30: return "Caution 🟠","#f97316"
    else: return "Avoid 🔴","#ef4444"

def get_comparables(df, zipcode, bedrooms, sqft_living, price, n=6):
    mask = ((df['zipcode']==zipcode) &
            (df['bedrooms'].between(bedrooms-1,bedrooms+1)) &
            (df['sqft_living'].between(sqft_living*0.75,sqft_living*1.25)))
    sub = df[mask].copy()
    if len(sub)<3: sub = df[df['zipcode']==zipcode].copy()
    if len(sub)==0: return pd.DataFrame()
    sub['diff'] = abs(sub['price']-price)
    return sub.nsmallest(min(n,len(sub)),'diff')[
        ['price','bedrooms','bathrooms','sqft_living','grade','condition','house_age','price_per_sqft']
    ].reset_index(drop=True)

def chart_feature_importance(model, feature_names, top_n=12):
    fd = pd.DataFrame({'Feature':feature_names,'Importance':model.feature_importances_})
    fd = fd.sort_values('Importance',ascending=True).tail(top_n)
    fig = px.bar(fd,x='Importance',y='Feature',orientation='h',
                 color='Importance',color_continuous_scale='Blues',
                 title=f'🔍 Top {top_n} Price Drivers (Explainable AI)',
                 labels={'Importance':'Importance Score','Feature':''})
    fig.update_layout(height=420,coloraxis_showscale=False,plot_bgcolor='white',
                      paper_bgcolor='white',font=dict(family='Arial',size=12),
                      title_font_size=15,margin=dict(l=10,r=10,t=50,b=30))
    fig.update_xaxes(showgrid=True,gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=False)
    return fig

def chart_price_dist(df, zipcode, price):
    zdf = df[df['zipcode']==zipcode]
    if len(zdf)<5: zdf=df; title='Price Distribution – All King County'
    else: title=f'Price Distribution – Zipcode {zipcode}'
    fig = px.histogram(zdf,x='price',nbins=40,title=title,
                       color_discrete_sequence=['#93c5fd'],opacity=0.8,
                       labels={'price':'House Price ($)'})
    fig.add_vline(x=price,line_dash='dash',line_color='#1d4ed8',line_width=3,
                  annotation_text=f'Your: ${price:,.0f}',annotation_position='top right',
                  annotation_font=dict(color='#1d4ed8',size=13))
    fig.add_vline(x=zdf['price'].mean(),line_dash='dot',line_color='#ef4444',line_width=2,
                  annotation_text=f'Avg: ${zdf["price"].mean():,.0f}',annotation_position='top left',
                  annotation_font=dict(color='#ef4444',size=11))
    fig.update_layout(height=360,plot_bgcolor='white',paper_bgcolor='white',
                      font=dict(family='Arial',size=12),title_font_size=14,
                      margin=dict(l=10,r=10,t=50,b=30),showlegend=False)
    return fig

def chart_confidence_gauge(conf):
    """Confidence gauge with needle and labeled zones"""
    conf_label = "High Confidence" if conf>=70 else "Medium Confidence" if conf>=50 else "Low Confidence"
    conf_color = "#10b981" if conf>=70 else "#f59e0b" if conf>=50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=conf,
        number={'suffix':'%', 'font':{'size':32,'color':'#1e293b','family':'Arial'}},
        title={'text': f"<b>Model Confidence</b><br><span style='font-size:13px;color:{conf_color}'>{conf_label}</span>",
               'font':{'size':14,'color':'#1e293b','family':'Arial'}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#94a3b8',
                    'tickvals':[0,25,50,75,100],'ticktext':['0%','25%','50%','75%','100%']},
            'bar': {'color': conf_color, 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range':[0,40],  'color':'#fee2e2'},
                {'range':[40,70], 'color':'#fef3c7'},
                {'range':[70,100],'color':'#d1fae5'},
            ],
            'threshold':{'line':{'color':conf_color,'width':4},'thickness':0.85,'value':conf}
        }
    ))
    fig.add_annotation(x=0.18, y=0.12, text="Low", showarrow=False,
                       font=dict(size=10, color='#ef4444', family='Arial'))
    fig.add_annotation(x=0.5, y=-0.02, text="Medium", showarrow=False,
                       font=dict(size=10, color='#f59e0b', family='Arial'))
    fig.add_annotation(x=0.82, y=0.12, text="High", showarrow=False,
                       font=dict(size=10, color='#10b981', family='Arial'))
    fig.update_layout(height=260, margin=dict(l=30,r=30,t=60,b=20),
                      paper_bgcolor='white', font=dict(family='Arial'))
    return fig

def chart_investment_gauge(score):
    """Investment score gauge with needle and clear zone labels"""
    lbl_text, lbl_color = inv_label(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix':'/100','font':{'size':32,'color':'#1e293b','family':'Arial'}},
        title={'text': f"<b>Investment Score</b><br><span style='font-size:13px;color:{lbl_color}'>{lbl_text}</span>",
               'font':{'size':14,'color':'#1e293b','family':'Arial'}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#94a3b8',
                    'tickvals':[0,30,45,60,75,100],
                    'ticktext':['0','30','45','60','75','100']},
            'bar': {'color': lbl_color, 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range':[0,30],  'color':'#fee2e2'},
                {'range':[30,45], 'color':'#ffedd5'},
                {'range':[45,60], 'color':'#fef3c7'},
                {'range':[60,75], 'color':'#dcfce7'},
                {'range':[75,100],'color':'#d1fae5'},
            ],
            'threshold':{'line':{'color':lbl_color,'width':4},'thickness':0.85,'value':score}
        }
    ))
    # Zone labels
    fig.add_annotation(x=0.12, y=0.08, text="Avoid", showarrow=False,
                       font=dict(size=9, color='#ef4444', family='Arial'))
    fig.add_annotation(x=0.3, y=0.02, text="Caution", showarrow=False,
                       font=dict(size=9, color='#f97316', family='Arial'))
    fig.add_annotation(x=0.5, y=-0.04, text="Moderate", showarrow=False,
                       font=dict(size=9, color='#f59e0b', family='Arial'))
    fig.add_annotation(x=0.7, y=0.02, text="Good", showarrow=False,
                       font=dict(size=9, color='#22c55e', family='Arial'))
    fig.add_annotation(x=0.88, y=0.08, text="Strong Buy", showarrow=False,
                       font=dict(size=9, color='#10b981', family='Arial'))
    fig.update_layout(height=260, margin=dict(l=30,r=30,t=60,b=30),
                      paper_bgcolor='white', font=dict(family='Arial'))
    return fig

def chart_forest_dist(model, scaler, feature_names, input_dict, price):
    feature_frame = pd.DataFrame([[input_dict.get(f, 0) for f in feature_names]], columns=feature_names)
    scaled = scaler.transform(feature_frame)
    preds = np.array([t.predict(scaled)[0] for t in model.estimators_])
    fig = px.histogram(x=preds,nbins=40,title='Forest Prediction Distribution (200 Trees)',
                       labels={'x':'Predicted Price ($)'},
                       color_discrete_sequence=['#86efac'],opacity=0.85)
    fig.add_vline(x=price,line_dash='dash',line_color='#15803d',line_width=3,
                  annotation_text=f'Avg: ${price:,.0f}',annotation_position='top right',
                  annotation_font=dict(color='#15803d',size=13))
    fig.update_layout(height=360,plot_bgcolor='white',paper_bgcolor='white',
                      font=dict(family='Arial',size=12),title_font_size=14,
                      margin=dict(l=10,r=10,t=50,b=30),showlegend=False)
    return fig

def chart_price_sqft(df, sqft_living, price):
    samp = df.sample(min(800,len(df)),random_state=42)
    fig = px.scatter(samp,x='sqft_living',y='price',color='grade',opacity=0.5,
                     color_continuous_scale='Viridis',
                     title='Price vs Living Area (Your Property = Red Star)',
                     labels={'sqft_living':'Living Area (sqft)','price':'Price ($)','grade':'Grade'},
                     hover_data=['bedrooms','bathrooms'])
    fig.add_scatter(x=[sqft_living],y=[price],mode='markers',
                    marker=dict(size=18,color='red',symbol='star',
                                line=dict(color='white',width=2)),
                    name='Your Property',showlegend=True)
    fig.update_layout(height=380,plot_bgcolor='white',paper_bgcolor='white',
                      font=dict(family='Arial',size=12),title_font_size=14,
                      margin=dict(l=10,r=10,t=50,b=30))
    return fig

def generate_pdf(inputs, price, low, high, conf, inv_score, status_text, comp_df):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable, KeepTogether)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

        # ── Palette ────────────────────────────────────────────────────────
        NAVY   = colors.HexColor('#0f2744')
        STEEL  = colors.HexColor('#1e3a5f')
        BLUE   = colors.HexColor('#2563eb')
        LBLUE  = colors.HexColor('#eff6ff')
        GOLD   = colors.HexColor('#b45309')
        LGOLD  = colors.HexColor('#fef3c7')
        GREEN  = colors.HexColor('#15803d')
        LGREEN = colors.HexColor('#f0fdf4')
        RED    = colors.HexColor('#b91c1c')
        LRED   = colors.HexColor('#fef2f2')
        AMBER  = colors.HexColor('#d97706')
        LAMBER = colors.HexColor('#fffbeb')
        GRAY   = colors.HexColor('#4b5563')
        MGRAY  = colors.HexColor('#9ca3af')
        LGRAY  = colors.HexColor('#f9fafb')
        RULE   = colors.HexColor('#e5e7eb')
        WHITE  = colors.white
        STRIPE = colors.HexColor('#f8faff')

        lbl, _ = inv_label(inv_score)
        ppsf   = price / max(inputs.get('sqft_living', 1), 1)
        today  = datetime.datetime.now().strftime('%B %d, %Y')

        # Status colours
        if 'Fair'  in status_text: s_bg, s_fg, s_word = LGREEN, GREEN, "FAIR MARKET VALUE"
        elif 'Over' in status_text: s_bg, s_fg, s_word = LRED,   RED,   "OVERPRICED"
        else:                        s_bg, s_fg, s_word = LBLUE,  BLUE,  "UNDERPRICED"

        if inv_score >= 70:   r_bg, r_fg, r_head = LGREEN, GREEN, "STRONG BUY RECOMMENDATION"
        elif inv_score >= 45: r_bg, r_fg, r_head = LAMBER, AMBER, "MODERATE — REVIEW ADVISED"
        else:                 r_bg, r_fg, r_head = LRED,   RED,   "CAUTION — NOT RECOMMENDED"

        # ── Document ───────────────────────────────────────────────────────
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=letter,
            leftMargin=0.75*inch, rightMargin=0.75*inch,
            topMargin=0.6*inch,   bottomMargin=0.65*inch
        )
        W = 7.0 * inch  # usable width

        # ── Style factory ──────────────────────────────────────────────────
        def S(name, **kw):
            d = dict(fontName='Helvetica', fontSize=10, textColor=GRAY,
                     spaceAfter=3, spaceBefore=0, leading=14)
            d.update(kw)
            return ParagraphStyle(name, **d)

        def P(text, style): return Paragraph(text, style)

        def rule(thickness=0.5, color=RULE):
            return HRFlowable(width='100%', thickness=thickness, color=color,
                              spaceAfter=0, spaceBefore=0)

        def sp(n=8): return Spacer(1, n)

        # ── Cell helpers ───────────────────────────────────────────────────
        def cp(text, bold=False, size=9, color=None, align=TA_LEFT, italic=False):
            fn = 'Helvetica-BoldOblique' if bold and italic else \
                 'Helvetica-Bold' if bold else \
                 'Helvetica-Oblique' if italic else 'Helvetica'
            return Paragraph(text, S('_cp', fontName=fn, fontSize=size,
                                     textColor=color or GRAY, alignment=align,
                                     spaceAfter=0, spaceBefore=0, leading=size+3))

        def kv_table(rows, w1=2.6*inch, w2=4.4*inch):
            """Two-column key-value table with subtle stripes"""
            data = []
            for k, v in rows:
                data.append([
                    cp(k, bold=True, size=9, color=STEEL),
                    cp(v, size=9, color=GRAY)
                ])
            t = Table(data, colWidths=[w1, w2])
            cmds = [
                ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE, LGRAY]),
                ('LEFTPADDING',  (0,0), (-1,-1), 10),
                ('RIGHTPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING',   (0,0), (-1,-1), 6),
                ('BOTTOMPADDING',(0,0), (-1,-1), 6),
                ('LINEBELOW', (0,0), (-1,-2), 0.3, RULE),
                ('BOX', (0,0), (-1,-1), 0.5, RULE),
            ]
            t.setStyle(TableStyle(cmds))
            return t

        def section_heading(text):
            return [
                sp(14),
                P(text.upper(), S('sh', fontName='Helvetica-Bold', fontSize=8,
                                  textColor=STEEL, spaceBefore=0, spaceAfter=4,
                                  leading=10, letterSpacing=1.2)),
                rule(1.0, STEEL),
                sp(6),
            ]

        def colored_badge(text, fg, bg, width=W):
            t = Table([[cp(text, bold=True, size=10, color=fg, align=TA_CENTER)]],
                      colWidths=[width])
            t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,-1), bg),
                ('TOPPADDING',    (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING',   (0,0), (-1,-1), 12),
                ('RIGHTPADDING',  (0,0), (-1,-1), 12),
                ('BOX',           (0,0), (-1,-1), 0.5, fg),
            ]))
            return t

        # ══════════════════════════════════════════════════════════════════
        story = []

        # ── PAGE HEADER (letterhead style) ─────────────────────────────
        hdr = Table(
            [[cp("PROPERTY PRICE ADVISORY REPORT", bold=True, size=18,
                  color=WHITE, align=TA_CENTER),
              cp(today, size=9, color=colors.HexColor('#93c5fd'), align=TA_RIGHT)],
             [cp("Intelligent Real Estate Intelligence System  ·  King County, Washington",
                  size=9, color=colors.HexColor('#93c5fd'), align=TA_LEFT),
              cp("CONFIDENTIAL", bold=True, size=8,
                  color=colors.HexColor('#bfdbfe'), align=TA_RIGHT)]],
            colWidths=[5.2*inch, 1.8*inch]
        )
        hdr.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), NAVY),
            ('TOPPADDING',    (0,0), (-1,-1), 20),
            ('BOTTOMPADDING', (0,0), (-1,-1), 18),
            ('LEFTPADDING',   (0,0), (-1,-1), 24),
            ('RIGHTPADDING',  (0,0), (-1,-1), 24),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ('SPAN',          (0,0), (0,0), ),
        ]))
        story.append(hdr)
        story.append(sp(16))

        # ── EXECUTIVE SUMMARY METRICS (4 boxes) ────────────────────────
        def metric_box(label, value, sub, bg, fg, border_color):
            inner = Table(
                [[cp(label, bold=True, size=7, color=MGRAY, align=TA_CENTER)],
                 [cp(value, bold=True, size=15, color=fg,   align=TA_CENTER)],
                 [cp(sub,   size=7,    color=MGRAY,          align=TA_CENTER)]],
                colWidths=[1.65*inch]
            )
            inner.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,-1), bg),
                ('TOPPADDING',    (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                ('LEFTPADDING',   (0,0), (-1,-1), 4),
                ('RIGHTPADDING',  (0,0), (-1,-1), 4),
                ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                ('BOX',           (0,0), (-1,-1), 1.5, border_color),
            ]))
            return inner

        cf_col = GREEN if conf>=70 else AMBER if conf>=50 else RED
        cf_bg  = LGREEN if conf>=70 else LAMBER if conf>=50 else LRED
        iv_col = GREEN if inv_score>=70 else AMBER if inv_score>=45 else RED
        iv_bg  = LGREEN if inv_score>=70 else LAMBER if inv_score>=45 else LRED

        metrics = Table([[
            metric_box("PREDICTED PRICE",      f"${price:,.0f}", "AI Best Estimate",      LBLUE,  BLUE,  BLUE),
            metric_box("PRICE RANGE",          f"${low/1000:.0f}K – ${high/1000:.0f}K", "Low – High Band", LGRAY, STEEL, RULE),
            metric_box("MODEL CONFIDENCE",     f"{conf:.0f}%",   "Certainty Level",       cf_bg,  cf_col, cf_col),
            metric_box("INVESTMENT SCORE",     f"{inv_score}/100", lbl,                   iv_bg,  iv_col, iv_col),
        ]], colWidths=[1.75*inch]*4)
        metrics.setStyle(TableStyle([
            ('LEFTPADDING',  (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING',   (0,0), (-1,-1), 0),
            ('BOTTOMPADDING',(0,0), (-1,-1), 0),
        ]))
        story.append(metrics)
        story.append(sp(12))

        # ── MARKET STATUS BAR ──────────────────────────────────────────
        mkt_line = (f"Market Valuation: {s_word}     |     "
                    f"Price per Sq.Ft.: ${ppsf:.0f}     |     "
                    f"Property Age: {inputs.get('house_age')} years     |     "
                    f"Zipcode: {inputs.get('zipcode_encoded') and 'King County'}")
        story.append(colored_badge(
            f"Market Valuation:  {s_word}   |   "
            f"Price / Sq.Ft.: ${ppsf:.0f}   |   "
            f"House Age: {inputs.get('house_age')} yrs",
            s_fg, s_bg
        ))
        story.append(sp(14))

        # ── SECTION A: PROPERTY DETAILS + RECOMMENDATION (side-by-side) ─
        story.extend(section_heading("A.  Property Profile & Investment Recommendation"))

        # Left: property details table
        prop_rows = [
            ("Bedrooms",         str(inputs.get('bedrooms'))),
            ("Bathrooms",        str(inputs.get('bathrooms'))),
            ("Living Area",      f"{inputs.get('sqft_living'):,} sq.ft."),
            ("Lot Size",         f"{inputs.get('sqft_lot'):,} sq.ft."),
            ("Number of Floors", str(inputs.get('floors'))),
            ("Building Grade",   f"{inputs.get('grade')} / 13"),
            ("Condition Rating", f"{inputs.get('condition')} / 5"),
            ("Year Built",       f"{inputs.get('current_year', datetime.datetime.now().year) - inputs.get('house_age', 0)} (Age: {inputs.get('house_age')} yrs)"),
            ("Waterfront",       "Yes" if inputs.get('waterfront') else "No"),
            ("Renovation",       "Previously Renovated" if inputs.get('renovated') else "No Renovation on Record"),
        ]
        prop_tbl = kv_table(prop_rows, w1=1.55*inch, w2=1.65*inch)

        # Right: recommendation panel
        if inv_score >= 70:
            rec_body = (
                "This property presents a strong investment opportunity. "
                "The AI model identifies competitive pricing relative to market "
                "benchmarks, above-average build quality, and favorable location "
                "metrics. Buyers should act promptly given the positive value indicators."
            )
        elif inv_score >= 45:
            rec_body = (
                "This property represents a reasonable market transaction at current "
                "rates. The valuation is consistent with comparable sales, though "
                "some quality or age factors slightly temper the investment appeal. "
                "Negotiating on asking price or factoring in renovation costs is advised "
                "before finalising the purchase."
            )
        else:
            rec_body = (
                "Caution is advised for this property. One or more key indicators "
                "— including price-per-sq.ft., condition rating, or building grade — "
                "fall below the market average for this area. Conduct a thorough "
                "structural inspection, obtain independent valuations, and negotiate "
                "aggressively before proceeding."
            )

        rec_panel = Table(
            [[cp(r_head, bold=True, size=10, color=r_fg, align=TA_CENTER)],
             [sp(4)],
             [cp(rec_body, size=8.5, color=GRAY, align=TA_LEFT)]],
            colWidths=[3.5*inch]
        )
        rec_panel.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), r_bg),
            ('BOX',           (0,0), (-1,-1), 0.8, r_fg),
            ('TOPPADDING',    (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING',   (0,0), (-1,-1), 14),
            ('RIGHTPADDING',  (0,0), (-1,-1), 14),
        ]))

        two_col = Table(
            [[prop_tbl, sp(16), rec_panel]],
            colWidths=[3.2*inch, 0.3*inch, 3.5*inch]
        )
        two_col.setStyle(TableStyle([
            ('VALIGN',       (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING',   (0,0), (-1,-1), 0),
            ('BOTTOMPADDING',(0,0), (-1,-1), 0),
            ('LEFTPADDING',  (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(two_col)

        # ── SECTION B: MARKET ANALYSIS ────────────────────────────────
        story.extend(section_heading("B.  Market Analysis & Price Confidence"))

        analysis_rows = [
            ("Predicted Market Value",     f"${price:,.0f}"),
            ("Confidence Interval (Low)",  f"${low:,.0f}  (−${(price-low)/1000:.0f}K from estimate)"),
            ("Confidence Interval (High)", f"${high:,.0f}  (+${(high-price)/1000:.0f}K from estimate)"),
            ("Model Confidence Level",     f"{conf:.1f}%  ({'High — narrow price band' if conf>=70 else 'Medium — some uncertainty' if conf>=50 else 'Low — wide price band'})"),
            ("Price per Sq. Ft.",          f"${ppsf:.0f}  ({'Below market avg — potential value buy' if ppsf<200 else 'At market rate' if ppsf<290 else 'Above market avg — price at premium'})"),
            ("Market Valuation Status",    s_word),
            ("Investment Score",           f"{inv_score} / 100  ({lbl})"),
            ("Amenity Score",              f"{inputs.get('amenity_score', 'N/A')} / 22  (Waterfront + View + Condition + Grade)"),
        ]
        story.append(kv_table(analysis_rows, w1=2.8*inch, w2=4.2*inch))

        # ── SECTION C: COMPARABLE PROPERTIES ─────────────────────────
        if comp_df is not None and len(comp_df) > 0:
            story.extend(section_heading("C.  Comparable Property Sales in Same Area"))
            story.append(P(
                f"The following {len(comp_df)} properties were sold in the same zipcode with "
                f"similar bedroom count and living area. They serve as market benchmarks "
                f"to validate the predicted price for this property.",
                S('intro', fontSize=8.5, textColor=GRAY, spaceAfter=8, leading=13)
            ))

            # Table header
            comp_rows_pdf = [[
                cp("Property Description",   bold=True, size=8.5, color=WHITE, align=TA_LEFT),
                cp("Sale Price",             bold=True, size=8.5, color=WHITE, align=TA_CENTER),
                cp("Sq.Ft.",                 bold=True, size=8.5, color=WHITE, align=TA_CENTER),
                cp("Grade / Cond.",          bold=True, size=8.5, color=WHITE, align=TA_CENTER),
                cp("Age",                    bold=True, size=8.5, color=WHITE, align=TA_CENTER),
                cp("$/Sq.Ft.",               bold=True, size=8.5, color=WHITE, align=TA_CENTER),
                cp("vs Your Price",          bold=True, size=8.5, color=WHITE, align=TA_CENTER),
            ]]

            for idx, (_, r) in enumerate(comp_df.iterrows()):
                diff = r['price'] - price
                diff_str = f"+${diff/1000:.0f}K" if diff > 0 else f"−${abs(diff)/1000:.0f}K"
                diff_col = RED if diff > 0 else GREEN
                desc = f"{int(r['bedrooms'])} Bed  |  {r['bathrooms']} Bath  |  {int(r['sqft_living']):,} sq.ft."
                bg = WHITE if idx % 2 == 0 else STRIPE
                comp_rows_pdf.append([
                    cp(desc,                            size=8.5, color=STEEL),
                    cp(f"${r['price']:,.0f}",           size=8.5, color=NAVY, align=TA_CENTER),
                    cp(f"{int(r['sqft_living']):,}",    size=8.5, align=TA_CENTER),
                    cp(f"{int(r['grade'])} / {int(r['condition'])}",  size=8.5, align=TA_CENTER),
                    cp(f"{int(r['house_age'])} yrs",    size=8.5, align=TA_CENTER),
                    cp(f"${r['price_per_sqft']:.0f}",   size=8.5, align=TA_CENTER),
                    cp(diff_str, bold=True, size=8.5, color=diff_col, align=TA_CENTER),
                ])

            # Your property (highlighted footer row)
            comp_rows_pdf.append([
                cp(f"YOUR PROPERTY  |  {inputs.get('bedrooms')} Bed  |  {inputs.get('bathrooms')} Bath  |  {inputs.get('sqft_living'):,} sq.ft.",
                   bold=True, size=8.5, color=BLUE),
                cp(f"${price:,.0f}",                   bold=True, size=8.5, color=BLUE, align=TA_CENTER),
                cp(f"{inputs.get('sqft_living'):,}",   bold=True, size=8.5, align=TA_CENTER),
                cp(f"{inputs.get('grade')} / {inputs.get('condition')}",
                   bold=True, size=8.5, align=TA_CENTER),
                cp(f"{inputs.get('house_age')} yrs",   bold=True, size=8.5, align=TA_CENTER),
                cp(f"${ppsf:.0f}",                     bold=True, size=8.5, align=TA_CENTER),
                cp("SUBJECT PROPERTY",                 bold=True, size=7.5, color=BLUE, align=TA_CENTER),
            ])

            comp_tbl = Table(
                comp_rows_pdf,
                colWidths=[2.4*inch, 0.9*inch, 0.65*inch, 0.8*inch, 0.6*inch, 0.65*inch, 0.9*inch]
            )
            n = len(comp_rows_pdf)
            comp_tbl.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,0), NAVY),
                ('TEXTCOLOR',     (0,0), (-1,0), WHITE),
                ('ROWBACKGROUNDS',(0,1), (-1,n-2), [WHITE, STRIPE]),
                ('BACKGROUND',    (0,n-1), (-1,n-1), LBLUE),
                ('BOX',           (0,0), (-1,-1), 0.6, RULE),
                ('LINEBELOW',     (0,0), (-1,-2), 0.3, RULE),
                ('LINEBELOW',     (0,0), (-1,0),  1.0, NAVY),
                ('LINEABOVE',     (0,n-1), (-1,n-1), 1.0, BLUE),
                ('TOPPADDING',    (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING',   (0,0), (-1,-1), 8),
                ('RIGHTPADDING',  (0,0), (-1,-1), 6),
                ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(comp_tbl)
            story.append(sp(4))
            story.append(P(
                "Note: 'vs Your Price' column shows how each comparable sold home's price compares "
                "to your AI-predicted price. Negative values indicate the comparable sold for less; "
                "positive values indicate it sold for more.",
                S('note', fontSize=7.5, textColor=MGRAY, leading=11, spaceAfter=0,
                  fontName='Helvetica-Oblique')
            ))

        # ── FOOTER ─────────────────────────────────────────────────────
        story.append(sp(16))
        story.append(rule(0.5, RULE))
        story.append(sp(5))
        story.append(P(
            f"This report was generated by the Intelligent Property Price Prediction System on {today}.  "
            f"Predictions are based on a Random Forest Regressor trained on King County house sales "
            f"data (2014–2015) and are intended for informational purposes only.  "
            f"This document does not constitute financial or legal advice.",
            S('disc', fontSize=7, textColor=MGRAY, alignment=TA_CENTER,
              leading=10, fontName='Helvetica-Oblique')
        ))

        doc.build(story)
        buf.seek(0)
        return buf, None

    except ImportError:
        return None, "reportlab not installed. Run: pip install reportlab"
    except Exception as e:
        return None, str(e)

# ── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown("# 🏠 House Price Prediction System")
st.markdown("**AI-powered Real Estate Intelligence** · King County, Washington")
st.markdown("---")

if not MODEL_OK:
    st.error("⚠️ model.pkl not found! Run the Colab notebook first, then place model.pkl here.")
    st.stop()

with st.sidebar:
    st.markdown("## 🏡 Enter Property Details")
    st.markdown("---")
    st.markdown("### 🛏 Basic Info")
    bedrooms  = st.slider("Bedrooms",  1, 10, 3)
    bathrooms = st.slider("Bathrooms", 0, 8,  2,
                          help="Whole bathrooms only (0–8)")
    floors    = st.slider("Floors",    1, 3,  1,
                          help="Number of floors (1, 2, or 3)")
    st.markdown("### 📐 Size")
    sqft_living   = st.number_input("Living Area (sqft)",   300,13000,1800,step=50)
    sqft_lot      = st.number_input("Lot Size (sqft)",      500,1600000,7500,step=100)
    sqft_above    = st.number_input("Sqft Above Ground",    300,9000,1800,step=50)
    sqft_basement = max(0, sqft_living - sqft_above)
    sqft_living15 = st.number_input("Neighbor Avg Living Sqft",300,6000,1800,step=50)
    sqft_lot15    = st.number_input("Neighbor Avg Lot Sqft",500,500000,7500,step=100)
    st.markdown("### ⭐ Quality")
    grade     = st.slider("Grade (1-13)",    1,13,7,help="7=Average, 10=Very Good, 13=Mansion")
    condition = st.slider("Condition (1-5)", 1,5, 3,help="1=Poor, 3=Average, 5=Excellent")
    st.markdown("### 📍 Location")
    zipcode = st.number_input("Zipcode",   98001,98199,98103)
    lat     = st.number_input("Latitude",  47.10,47.80,47.50,step=0.001,format="%.4f")
    long_   = st.number_input("Longitude",-122.60,-121.30,-122.20,step=0.001,format="%.4f")
    st.markdown("### 🗓 Age & History")
    yr_built  = st.number_input("Year Built", 1900,2024,1990)
    renovated = st.checkbox("Has Been Renovated")
    st.markdown("### 🌟 Special Features")
    waterfront = st.checkbox("Waterfront Property 🌊")
    view       = st.slider("View Quality (0-4)",0,4,0)
    st.markdown("---")
    st.button("🔍 Predict Price", width="stretch")

# Compute derived values
house_age     = current_year - yr_built
amenity_score = int(waterfront) + view + condition + grade
sqft_basement = max(0, sqft_living - sqft_above)
zipcode_enc   = zipcode_mean.get(zipcode, float(zipcode_mean.mean()))

inputs = {
    'bedrooms':bedrooms,'bathrooms':bathrooms,'sqft_living':sqft_living,
    'sqft_lot':sqft_lot,'floors':floors,'waterfront':int(waterfront),
    'view':view,'condition':condition,'grade':grade,'sqft_above':sqft_above,
    'sqft_basement':sqft_basement,'lat':lat,'long':long_,'sqft_living15':sqft_living15,
    'sqft_lot15':sqft_lot15,'renovated':int(renovated),'house_age':house_age,
    'amenity_score':amenity_score,'zipcode_encoded':zipcode_enc,
}

price,low,high,conf,std_dev = predict_price(inputs)
inv_sc = investment_score(price,sqft_living,grade,condition,house_age,int(waterfront),view,int(renovated))
status_text,badge_class,status_color,ppsf = market_status(price,sqft_living)
lbl,inv_color = inv_label(inv_sc)
comp_df = get_comparables(df_full,zipcode,bedrooms,sqft_living,price)

# ROW 1: Metrics
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Price</div>'
                f'<div class="metric-value">${price:,.0f}</div>'
                f'<div class="metric-sub">Best Estimate</div></div>',unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card metric-card-green"><div class="metric-label">Price Range</div>'
                f'<div class="metric-value" style="font-size:15px;line-height:1.4;">'
                f'${low:,.0f} &ndash; ${high:,.0f}</div>'
                f'<div class="metric-sub">±1 Std Deviation</div></div>',unsafe_allow_html=True)
with c3:
    cc = "metric-card-green" if conf>=70 else "metric-card-orange" if conf>=50 else "metric-card-red"
    st.markdown(f'<div class="metric-card {cc}"><div class="metric-label">Model Confidence</div>'
                f'<div class="metric-value">{conf:.1f}%</div>'
                f'<div class="metric-sub">{"High ✅" if conf>=70 else "Medium ⚠️" if conf>=50 else "Low ❌"}</div></div>',
                unsafe_allow_html=True)
with c4:
    ic = "metric-card-green" if inv_sc>=70 else "metric-card-orange" if inv_sc>=45 else "metric-card-red"
    st.markdown(f'<div class="metric-card {ic}"><div class="metric-label">Investment Score</div>'
                f'<div class="metric-value">{inv_sc}<span style="font-size:16px;">/100</span></div>'
                f'<div class="metric-sub">{lbl}</div></div>',unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ROW 2: Market Status + Gauges
st.markdown('<div class="section-title">📊 Market Intelligence</div>',unsafe_allow_html=True)
ca,cb,cc2 = st.columns([1,1,2])
with ca:
    st.markdown("**Market Status**")
    st.markdown(f'<span class="badge {badge_class}" style="font-size:16px;">{status_text}</span>',unsafe_allow_html=True)
    st.markdown(f"""<br><table style="font-size:13px;width:100%;">
    <tr><td style="color:#6b7280;">Price per sqft</td><td><b>${ppsf:.0f}</b></td></tr>
    <tr><td style="color:#6b7280;">House Age</td><td><b>{house_age} yrs</b></td></tr>
    <tr><td style="color:#6b7280;">Amenity Score</td><td><b>{amenity_score}/22</b></td></tr>
    <tr><td style="color:#6b7280;">Waterfront</td><td><b>{"Yes ✅" if waterfront else "No"}</b></td></tr>
    <tr><td style="color:#6b7280;">Renovated</td><td><b>{"Yes ✅" if renovated else "No"}</b></td></tr>
    </table>""", unsafe_allow_html=True)
with cb:
    st.plotly_chart(chart_confidence_gauge(conf), width="stretch")
with cc2:
    st.plotly_chart(chart_investment_gauge(inv_sc), width="stretch")

# ROW 3: Analytics
st.markdown("---")
st.markdown('<div class="section-title">📈 Analytics Dashboard</div>',unsafe_allow_html=True)
cl,cr = st.columns(2)
with cl: st.plotly_chart(chart_feature_importance(model,feature_names), width="stretch")
with cr: st.plotly_chart(chart_price_dist(df_full,zipcode,price), width="stretch")
cl2,cr2 = st.columns(2)
with cl2: st.plotly_chart(chart_forest_dist(model,scaler,feature_names,inputs,price), width="stretch")
with cr2: st.plotly_chart(chart_price_sqft(df_full,sqft_living,price), width="stretch")

# ROW 4: Comparables
st.markdown("---")
st.markdown('<div class="section-title">🏘 Comparable Properties</div>', unsafe_allow_html=True)

if len(comp_df) > 0:
    cd = comp_df.copy()

    # ── Display table: all integers, clean columns ────────────────────
    disp = cd.copy()
    disp['price']          = disp['price'].apply(lambda x: f"${x:,.0f}")
    disp['bathrooms']      = disp['bathrooms'].apply(lambda x: int(round(x)))   # integer baths
    disp['sqft_living']    = disp['sqft_living'].apply(lambda x: f"{int(x):,}")
    disp['price_per_sqft'] = disp['price_per_sqft'].apply(lambda x: f"${x:.0f}/sqft")
    disp['grade']          = disp['grade'].apply(lambda x: int(x))
    disp['condition']      = disp['condition'].apply(lambda x: int(x))
    disp['house_age']      = disp['house_age'].apply(lambda x: f"{int(x)} yrs")
    disp.columns = ['Sale Price', 'Beds', 'Baths', 'Living Area', 'Grade', 'Condition', 'Age', '$/sqft']

    # ── Your property summary row ─────────────────────────────────────
    your_row = {
        'Sale Price': f"${price:,.0f} ★",
        'Beds':        bedrooms,
        'Baths':       bathrooms,
        'Living Area': f"{sqft_living:,}",
        'Grade':       grade,
        'Condition':   condition,
        'Age':         f"{house_age} yrs",
        '$/sqft':      f"${ppsf:.0f}/sqft"
    }

    st.markdown(
        '<p style="font-size:13px;color:#6b7280;margin-bottom:6px;">'
        f'Showing <b>{len(cd)}</b> similar sold properties in zipcode <b>{zipcode}</b> '
        f'— matched by bedrooms (±1) and living area (±25%)</p>',
        unsafe_allow_html=True
    )

    ct1, ct2 = st.columns([1, 1.1])

    with ct1:
        st.markdown("**📋 Sold Properties**")
        st.dataframe(disp, width="stretch", hide_index=True, height=260)

        # Your property card below table
        avg_comp      = cd['price'].mean()
        diff_from_avg = price - avg_comp
        diff_color    = "#10b981" if diff_from_avg < 0 else "#ef4444"
        diff_label    = "below" if diff_from_avg < 0 else "above"
        st.markdown(
            f'<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;'
            f'padding:10px 14px;margin-top:8px;font-size:13px;">'
            f'<b>★ Your Property (Predicted):</b> <b style="color:#1d4ed8;">${price:,.0f}</b> &nbsp;|&nbsp; '
            f'{bedrooms} bed &nbsp;{bathrooms} bath &nbsp;{sqft_living:,} sqft &nbsp;Grade {grade}<br>'
            f'<span style="color:{diff_color};font-weight:600;">'
            f'${abs(diff_from_avg):,.0f} {diff_label} comparable average (${avg_comp:,.0f})'
            f'</span></div>',
            unsafe_allow_html=True
        )

    with ct2:
        st.markdown("**📊 Price Comparison Chart**")

        # Build labels — short and clean, no decimals in baths
        def make_label(row):
            diff     = row['price'] - price
            diff_str = f"+${diff/1000:.0f}K" if diff > 0 else f"-${abs(diff)/1000:.0f}K" if diff != 0 else "≈ same"
            baths_i  = int(round(row['bathrooms']))
            return f"  {int(row['bedrooms'])} bed · {baths_i} bath · {int(row['sqft_living']):,} sqft · Grade {int(row['grade'])}   [{diff_str}]"

        your_label  = f"  ★ YOUR PROPERTY  —  {bedrooms} bed · {bathrooms} bath · {sqft_living:,} sqft · Grade {grade}"
        comp_labels = [make_label(cd.iloc[i]) for i in range(len(cd))]

        all_labels = [your_label] + comp_labels
        all_prices = [price]      + list(cd['price'].values)
        bar_colors = ['#1e40af']  + ['#60a5fa'] * len(cd)

        fc = go.Figure()
        fc.add_trace(go.Bar(
            y=all_labels,
            x=all_prices,
            orientation='h',
            marker=dict(
                color=bar_colors,
                line=dict(color='rgba(255,255,255,0.5)', width=1)
            ),
            text=[f"  ${p:,.0f}" for p in all_prices],
            textposition='inside',
            insidetextanchor='start',
            textfont=dict(size=11, color='white', family='Arial'),
            hovertemplate='<b>%{y}</b><br>Price: <b>$%{x:,.0f}</b><extra></extra>',
            width=0.65,   # thicker bars
        ))

        # Your price reference line
        fc.add_vline(
            x=price,
            line_dash='dash',
            line_color='#dc2626',
            line_width=2,
            annotation_text=f'Your price: ${price:,.0f}',
            annotation_position='top right',
            annotation_font=dict(color='#dc2626', size=10, family='Arial'),
            annotation_bgcolor='rgba(255,255,255,0.85)',
        )

        # Avg comparable line
        fc.add_vline(
            x=avg_comp,
            line_dash='dot',
            line_color='#9ca3af',
            line_width=1.5,
            annotation_text=f'Avg: ${avg_comp:,.0f}',
            annotation_position='bottom right',
            annotation_font=dict(color='#9ca3af', size=9, family='Arial'),
        )

        x_pad = (max(all_prices) - min(all_prices)) * 0.12
        x_min = min(all_prices) - x_pad
        x_max = max(all_prices) + x_pad * 1.5

        fc.update_layout(
            height=max(380, 60 * len(all_labels) + 80),
            plot_bgcolor='#f8fafc',
            paper_bgcolor='white',
            margin=dict(l=10, r=20, t=20, b=50),
            showlegend=False,
            xaxis=dict(
                showgrid=True, gridcolor='#e2e8f0', gridwidth=1,
                range=[x_min, x_max],
                title=dict(text='Sale Price (USD)', font=dict(size=11, color='#475569')),
                tickformat='$,.0f',
                tickfont=dict(size=9, family='Arial', color='#475569'),
                zeroline=False,
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=9.5, family='Arial', color='#1e293b'),
                automargin=True,
            ),
            bargap=0.35,
        )

        st.plotly_chart(fc, width="stretch")

        # Clean legend (no markdown rendering issues)
        st.markdown(
            '<div style="font-size:12px;color:#6b7280;line-height:1.7;margin-top:-10px;">'
            '<span style="display:inline-block;width:14px;height:14px;background:#1e40af;'
            'border-radius:3px;vertical-align:middle;margin-right:5px;"></span>'
            'Your predicted price&emsp;'
            '<span style="display:inline-block;width:14px;height:14px;background:#60a5fa;'
            'border-radius:3px;vertical-align:middle;margin-right:5px;"></span>'
            'Comparable sold homes&emsp;'
            '<span style="color:#dc2626;font-weight:600;">— — —</span> Your prediction line'
            '</div>',
            unsafe_allow_html=True
        )

else:
    st.info("No comparable properties found for this zipcode.")

# ROW 5: PDF
st.markdown("---")
st.markdown('<div class="section-title">📄 Advisory Report</div>',unsafe_allow_html=True)
r1,r2 = st.columns([2,1])
with r1:
    st.markdown(f"""Report includes: Predicted Price: ${price:,.0f} · 
    Range: ${low:,.0f}–${high:,.0f} · Status: {status_text} · 
    Investment Score **{inv_sc}/100** · Comparables · AI Recommendation""")
with r2:
    if st.button("📥 Generate PDF Report"):
        with st.spinner("Generating..."):
            pdf_buf, err = generate_pdf(inputs,price,low,high,conf,inv_sc,status_text,
                                        comp_df if len(comp_df)>0 else None)
        if pdf_buf:
            st.download_button("⬇ Download PDF", data=pdf_buf,
                               file_name=f"property_report_{zipcode}.pdf",
                               mime="application/pdf")
            st.success("✅ Ready!")
        else:
            st.error(f"Error: {err}")

st.markdown("---")
st.markdown("<center style='color:#94a3b8;font-size:13px;'>🏠 House Price Prediction · King County WA · Random Forest ML + Streamlit + Plotly</center>",
            unsafe_allow_html=True)
