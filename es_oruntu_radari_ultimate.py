
# -*- coding: utf-8 -*-
"""
E.Ş. ÖRÜNTÜ RADARI — ULTIMATE EDITION
İstatistik • Kalite • Prototip • Ekonomi • Raporlama
"""

from __future__ import annotations

import io
import json
import math
import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# AYARLAR
# ============================================================

st.set_page_config(
    page_title="E.Ş. Örüntü Radarı",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORANGE = "#F28C28"
DARK = "#111318"
LIGHT = "#F4F5F7"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {DARK};
    }}
    [data-testid="stSidebar"] {{
        background: #191C23;
        border-right: 1px solid #343944;
    }}
    h1, h2, h3 {{
        letter-spacing: -0.5px;
    }}
    .es-hero {{
        padding: 25px;
        border: 1px solid #393D46;
        border-left: 5px solid {ORANGE};
        border-radius: 16px;
        background: linear-gradient(115deg,#191C23,#22252D);
        margin-bottom: 22px;
    }}
    .es-kicker {{
        color: {ORANGE};
        font-weight: 800;
        letter-spacing: 3px;
        font-size: 12px;
    }}
    .es-sub {{
        color: #B9BEC8;
        font-size: 14px;
    }}
    div.stButton > button[kind="primary"] {{
        background: {ORANGE};
        color: #111318;
        border: 0;
        font-weight: 700;
    }}
    [data-testid="stMetric"] {{
        background: #1B1F27;
        padding: 14px;
        border: 1px solid #343944;
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VERİ YARDIMCILARI
# ============================================================

def demo_data():
    rng = np.random.default_rng(42)
    n = 240
    production = rng.normal(100, 8, n)
    conversion = rng.normal(72, 6, n)
    cost = rng.normal(48, 5, n)
    revenue = conversion * 1.15 + rng.normal(0, 8, n)

    return pd.DataFrame({
        "Tarih": pd.date_range("2026-01-01", periods=n, freq="D"),
        "Üretim": production.round(2),
        "Dönüşüm": conversion.round(2),
        "Maliyet": cost.round(2),
        "Gelir": revenue.round(2),
        "Hata_Adedi": rng.poisson(3, n),
        "Memnuniyet": rng.uniform(2.5, 5, n).round(2),
    })


def read_upload(upload):
    name = upload.name.lower()

    if name.endswith(".csv"):
        raw = upload.getvalue()
        for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin1"):
            try:
                return pd.read_csv(
                    io.BytesIO(raw), encoding=encoding
                )
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV kodlaması okunamadı.")

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)

    if name.endswith(".json"):
        return pd.read_json(upload)

    raise ValueError("Desteklenmeyen dosya türü.")


def numeric_columns(df):
    return list(df.select_dtypes(include=np.number).columns)


def safe_number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def clean_text(value):
    """Raporlarda HTML/XML karakterlerinin güvenli gösterimi."""
    return escape(str(value))


def data_profile(df):
    nums = numeric_columns(df)

    return {
        "Satır sayısı": int(len(df)),
        "Sütun sayısı": int(len(df.columns)),
        "Sayısal sütun": int(len(nums)),
        "Eksik hücre": int(df.isna().sum().sum()),
        "Yinelenen satır": int(df.duplicated().sum()),
        "Bellek (KB)": round(
            df.memory_usage(deep=True).sum() / 1024, 2
        ),
    }


def automatic_findings(df):
    findings = []
    profile = data_profile(df)

    findings.append(
        f"Veri kümesi {profile['Satır sayısı']} satır ve "
        f"{profile['Sütun sayısı']} sütun içeriyor."
    )
    findings.append(
        f"Eksik hücre sayısı: {profile['Eksik hücre']}."
    )
    findings.append(
        f"Yinelenen satır sayısı: {profile['Yinelenen satır']}."
    )

    nums = numeric_columns(df)

    if nums and len(df) >= 4:
        q1 = df[nums].quantile(0.25)
        q3 = df[nums].quantile(0.75)
        iqr = q3 - q1
        mask = (
            (df[nums] < q1 - 1.5 * iqr)
            | (df[nums] > q3 + 1.5 * iqr)
        ).any(axis=1)

        findings.append(
            f"IQR kuralıyla {int(mask.sum())} satırda "
            "olası aykırı değer işaretlendi."
        )

    if len(nums) >= 2 and len(df) >= 3:
        corr = df[nums].corr()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        ).stack()

        if not upper.empty:
            pair = upper.abs().idxmax()
            value = float(upper.loc[pair])
            findings.append(
                f"En yüksek mutlak Pearson korelasyonu: "
                f"{pair[0]} ile {pair[1]}, r={value:.3f}. "
                "Korelasyon nedensellik kanıtı değildir."
            )

    return findings


def local_report(df, findings, question=""):
    profile = data_profile(df)
    lines = [
        "# E.Ş. Örüntü Radarı — Analiz Raporu",
        "",
        f"Rapor tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        "",
        "## Yönetici Özeti",
        "",
        f"Analiz edilen veri kümesi {len(df)} satır ve "
        f"{len(df.columns)} sütundan oluşmaktadır.",
        "",
        "## Veri Profili",
        "",
    ]

    for key, value in profile.items():
        lines.append(f"- {key}: {value}")

    lines += ["", "## Temel Bulgular", ""]
    lines += [f"- {item}" for item in findings]

    lines += [
        "",
        "## Sınırlılıklar",
        "",
        "Aykırı değer işaretlenmesi tek başına ölçüm hatası olduğunu "
        "göstermez. Korelasyon nedenselliği kanıtlamaz. "
        "Sonuçlar veri kalitesi ve kullanılan yöntemlerle sınırlıdır.",
        "",
        "## Sonuç ve Öneriler",
        "",
        "İş açısından önemli bulgular ham kayıtlar, ölçüm yöntemi ve "
        "alan bilgisiyle doğrulanmalıdır. Kritik üretim kararları "
        "bağımsız kalite kontrolüyle desteklenmelidir.",
    ]

    if question.strip():
        lines += [
            "",
            "## Kullanıcı Sorusu",
            "",
            question.strip(),
            "",
            "Bu bölüm yerel raporda otomatik olarak yorumlanmamıştır.",
        ]

    return "\n".join(lines)


def create_report(df, findings, api_key="", model="gpt-4.1-mini",
                  question=""):
    report = local_report(df, findings, question)

    if not api_key:
        return report, "Yerel analiz"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=(
                "Türkçe veri analisti olarak yaz. Doğal ve dil bilgisi "
                "doğru Türkçe kullan; UTF-8 Türkçe karakterleri koru. "
                "Yalnızca verilen verileri kullan; sayı uydurma. "
                "Korelasyonu nedensellik olarak sunma. Belirsizlikleri "
                "açıkça belirt. Yönetici Özeti, Veri Profili, Bulgular, "
                "Sınırlılıklar, Öneriler ve Sonuç başlıklarını kullan."
            ),
            input=json.dumps({
                "profil": data_profile(df),
                "bulgular": findings,
                "soru": question,
                "yerel_rapor": report,
            }, ensure_ascii=False, default=str),
        )

        text = (response.output_text or "").strip()
        if text:
            return text, f"Yapay zekâ · {model}"

    except Exception as exc:
        report += (
            "\n\n## Yapay Zekâ Durumu\n\n"
            f"Yapay zekâ yanıtı alınamadı: {type(exc).__name__}. "
            "Yerel rapor kullanılmaktadır."
        )

    return report, "Yerel yedek analiz"


# ============================================================
# DIŞA AKTARMA
# ============================================================

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def excel_bytes(df, report, profile):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Veri", index=False)
        pd.DataFrame(
            list(profile.items()),
            columns=["Gösterge", "Değer"]
        ).to_excel(writer, sheet_name="Veri Profili", index=False)
        pd.DataFrame({"Rapor": report.splitlines()}).to_excel(
            writer, sheet_name="Rapor", index=False
        )
    return buf.getvalue()


def pdf_bytes(title, report, df):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.8 * cm, leftMargin=1.8 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm,
        title=title, author="E.Ş. Örüntü Radarı"
    )

    font_name = "Helvetica"
    bold_name = "Helvetica-Bold"

    font_candidates = [
        (
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
    ]

    for regular, bold in font_candidates:
        if os.path.exists(regular):
            try:
                pdfmetrics.registerFont(TTFont("ESFont", regular))
                font_name = "ESFont"
                if os.path.exists(bold):
                    pdfmetrics.registerFont(TTFont("ESFontBold", bold))
                    bold_name = "ESFontBold"
                break
            except Exception:
                pass

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ESBody", parent=styles["BodyText"],
        fontName=font_name, fontSize=9, leading=13,
        spaceAfter=5, wordWrap="LTR"
    ))
    styles.add(ParagraphStyle(
        name="ESHeading", parent=styles["Heading2"],
        fontName=bold_name, fontSize=13, leading=17,
        textColor=colors.HexColor("#F28C28"),
        spaceBefore=12, spaceAfter=7
    ))
    styles.add(ParagraphStyle(
        name="ESTitle", parent=styles["Title"],
        fontName=bold_name, fontSize=21, leading=26,
        textColor=colors.HexColor("#F28C28"),
        alignment=TA_LEFT, spaceAfter=8
    ))

    story = [
        Paragraph(clean_text(title), styles["ESTitle"]),
        Paragraph(
            "E.Ş. Örüntü Radarı | "
            + datetime.now().strftime("%d.%m.%Y %H:%M"),
            styles["ESBody"]
        ),
        Spacer(1, 12),
    ]

    for line in report.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.startswith("# "):
            story.append(Paragraph(
                clean_text(line[2:]), styles["ESTitle"]
            ))
        elif line.startswith("## "):
            story.append(Paragraph(
                clean_text(line[3:]), styles["ESHeading"]
            ))
        elif line.startswith("- "):
            story.append(Paragraph(
                "• " + clean_text(line[2:]), styles["ESBody"]
            ))
        else:
            story.append(Paragraph(clean_text(line), styles["ESBody"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Veri Profili", styles["ESHeading"]))

    profile = data_profile(df)
    rows = [["Gösterge", "Değer"]]
    rows += [[str(k), str(v)] for k, v in profile.items()]
    table = Table(rows, colWidths=[8 * cm, 7 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#22252D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#F28C28")),
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#F28C28"))
        canvas.line(1.8 * cm, 1.25 * cm, 19.2 * cm, 1.25 * cm)
        canvas.setFont(font_name, 8)
        canvas.drawString(1.8 * cm, 0.8 * cm, "E.Ş. Örüntü Radarı")
        canvas.drawRightString(
            19.2 * cm, 0.8 * cm, f"Sayfa {document.page}"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ============================================================
# UYGULAMA BAŞLIĞI
# ============================================================

st.markdown(
    """
    <div class="es-hero">
      <div class="es-kicker">E.Ş. • DATA INTELLIGENCE</div>
      <h1 style="margin:8px 0 5px 0;">ÖRÜNTÜ RADARI</h1>
      <div class="es-sub">
        İstatistik • Endüstriyel Kalite • Prototip Testleri • Ekonomi
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## E.Ş. Kontrol Merkezi")
    st.caption("Veri yükle, analiz et, raporla.")

    upload = st.file_uploader(
        "Veri dosyası",
        type=["csv", "xlsx", "xls", "json"],
        help="CSV, Excel veya JSON dosyası seç.",
    )

    st.divider()
    st.markdown("### Yapay zekâ raporu")
    api_key = st.text_input(
        "OpenAI API anahtarı",
        type="password",
        help="İsteğe bağlıdır. API kullanımı ücret doğurabilir.",
    )
    ai_model = st.text_input("Model adı", value="gpt-4.1-mini")

    st.divider()
    st.caption("Dosya seçmezsen örnek veri kullanılır.")


# ============================================================
# VERİ YÜKLEME VE HAZIRLAMA
# ============================================================

if upload is not None:
    try:
        source_df = read_upload(upload)
        st.success(f"Dosya okundu: {upload.name}")
    except Exception as exc:
        st.error(f"Dosya okunamadı: {type(exc).__name__}: {exc}")
        st.stop()
else:
    source_df = demo_data()

if source_df.empty:
    st.warning("Veri kümesi boş. Lütfen başka bir dosya yükle.")
    st.stop()

df = source_df.copy()
df.columns = [str(c).strip() for c in df.columns]

# Sayısal gibi duran metinleri kullanıcı kontrolünde dönüştürme.
with st.sidebar:
    convert_numeric = st.checkbox(
        "Sayısal görünümlü metinleri dönüştür",
        value=False,
    )

if convert_numeric:
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            if converted.notna().sum() >= max(3, int(len(df) * 0.7)):
                df[col] = converted

with st.sidebar:
    st.markdown("### Veri temizleme")
    missing_method = st.selectbox(
        "Eksik değer işlemi",
        ["İşlem yapma", "Sayısal medyan / metin mod", "Eksik satırları sil"],
    )
    remove_duplicates = st.checkbox("Yinelenen satırları kaldır", False)

if missing_method == "Sayısal medyan / metin mod":
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].notna().any():
                    df[col] = df[col].fillna(df[col].median())
            else:
                modes = df[col].mode(dropna=True)
                if not modes.empty:
                    df[col] = df[col].fillna(modes.iloc[0])

elif missing_method == "Eksik satırları sil":
    df = df.dropna().copy()

if remove_duplicates:
    df = df.drop_duplicates().copy()

profile = data_profile(df)
findings = automatic_findings(df)
nums = numeric_columns(df)
cats = list(df.select_dtypes(exclude=np.number).columns)


# ============================================================
# GENEL BAKIŞ
# ============================================================

tabs = st.tabs([
    "Genel Bakış",
    "Grafik Stüdyosu",
    "Örüntü Laboratuvarı",
    "Anomali Merkezi",
    "Kalite & Fabrika",
    "Prototip Testleri",
    "Ekonomik Analiz",
    "Zaman Serisi",
    "Veri Kalitesi",
    "Rapor Merkezi",
])

with tabs[0]:
    st.subheader("Genel Bakış")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satır", f"{len(df):,}")
    c2.metric("Sütun", f"{len(df.columns):,}")
    c3.metric("Eksik hücre", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Yinelenen satır", f"{int(df.duplicated().sum()):,}")

    st.markdown("### Veri ön izlemesi")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("### Otomatik bulgular")
    for finding in findings:
        st.write("• " + finding)

    if nums:
        st.markdown("### Sayısal özet")
        st.dataframe(df[nums].describe().T, use_container_width=True)


# ============================================================
# GRAFİK STÜDYOSU
# ============================================================

with tabs[1]:
    st.subheader("Grafik Stüdyosu")

    if not nums:
        st.info("Grafik üretmek için sayısal sütun gerekli.")
    else:
        chart_type = st.selectbox(
            "Grafik türü",
            [
                "Histogram",
                "Kutu grafiği",
                "Yoğunluk",
                "Dağılım",
                "Çizgi",
                "Çubuk",
                "Korelasyon ısı haritası",
                "Kümülatif dağılım",
            ],
        )

        x = st.selectbox("X sütunu", df.columns, key="chart_x")
        y_choices = [c for c in nums if c != x] or nums
        y = st.selectbox("Y / değer sütunu", y_choices, key="chart_y")

        color_col = st.selectbox(
            "Renk grubu (isteğe bağlı)",
            ["Kullanma"] + cats,
        )
        color_arg = None if color_col == "Kullanma" else color_col

        if chart_type == "Histogram":
            fig = px.histogram(
                df, x=x, color=color_arg, marginal="box",
                title=f"{x} dağılımı"
            )
        elif chart_type == "Kutu grafiği":
            fig = px.box(
                df, x=x if x in cats else None,
                y=y if y in nums else x,
                color=color_arg, points="outliers",
            )
        elif chart_type == "Yoğunluk":
            fig = px.density_contour(df, x=x, y=y)
        elif chart_type == "Dağılım":
            if x not in nums:
                st.warning("Dağılım grafiği için X sayısal olmalı.")
                fig = None
            else:
                fig = px.scatter(df, x=x, y=y, color=color_arg,
                                 trendline="ols")
        elif chart_type == "Çizgi":
            fig = px.line(df, x=x, y=y, color=color_arg,
                          markers=True)
        elif chart_type == "Çubuk":
            if x in cats:
                grouped = df.groupby(x, dropna=False)[y].mean().reset_index()
                fig = px.bar(grouped, x=x, y=y)
            else:
                st.warning("Çubuk grafiği için kategorik X sütunu seç.")
                fig = None
        elif chart_type == "Korelasyon ısı haritası":
            corr = df[nums].corr()
            fig = px.imshow(
                corr, text_auto=".2f", zmin=-1, zmax=1,
                color_continuous_scale="RdBu_r",
                title="Pearson korelasyon matrisi",
            )
        else:
            if x not in nums:
                st.warning("Kümülatif dağılım için sayısal sütun seç.")
                fig = None
            else:
                values = np.sort(df[x].dropna().to_numpy())
                if len(values):
                    fig = go.Figure(
                        go.Scatter(
                            x=values,
                            y=np.arange(1, len(values) + 1) / len(values),
                            mode="lines",
                        )
                    )
                    fig.update_layout(
                        title=f"{x} — ampirik kümülatif dağılım",
                        xaxis_title=x,
                        yaxis_title="Kümülatif oran",
                    )
                else:
                    fig = None

        if fig is not None:
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ÖRÜNTÜ LABORATUVARI
# ============================================================

with tabs[2]:
    st.subheader("Örüntü Laboratuvarı")

    if len(nums) < 2:
        st.info("Korelasyon veya kümeleme için en az iki sayısal sütun gerekli.")
    else:
        corr = df[nums].corr()
        st.markdown("### Korelasyon matrisi")
        st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

        st.markdown("### K-Means kümeleme")
        selected = st.multiselect(
            "Kümeleme değişkenleri", nums,
            default=nums[:min(3, len(nums))],
        )
        k = st.slider("Küme sayısı", 2, 8, 3)

        if len(selected) >= 2 and len(df.dropna(subset=selected)) > k:
            cluster_df = df.dropna(subset=selected).copy()
            X = StandardScaler().fit_transform(cluster_df[selected])
            labels = KMeans(
                n_clusters=k, random_state=42, n_init=10
            ).fit_predict(X)
            cluster_df["Küme"] = labels.astype(str)

            if len(selected) >= 2:
                fig = px.scatter(
                    cluster_df, x=selected[0], y=selected[1],
                    color="Küme", title="K-Means kümeleri",
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                cluster_df.groupby("Küme")[selected].mean(),
                use_container_width=True,
            )

        st.markdown("### Doğrusal regresyon")
        x_reg = st.selectbox("Girdi değişkeni", nums, key="reg_x")
        y_reg = st.selectbox(
            "Tahmin edilecek değişken",
            [c for c in nums if c != x_reg],
            key="reg_y",
        )

        reg_data = df[[x_reg, y_reg]].dropna()
        if len(reg_data) >= 3 and reg_data[x_reg].nunique() >= 2:
            model = LinearRegression()
            model.fit(reg_data[[x_reg]], reg_data[y_reg])
            pred = model.predict(reg_data[[x_reg]])
            r2 = model.score(reg_data[[x_reg]], reg_data[y_reg])

            st.metric("R² (aynı veri üzerinde)", f"{r2:.3f}")
            st.caption(
                "Bu değer eğitim verisi üzerindedir; bağımsız test "
                "performansı olarak yorumlanmamalıdır."
            )

            fig = px.scatter(reg_data, x=x_reg, y=y_reg)
            order = np.argsort(reg_data[x_reg].to_numpy())
            fig.add_scatter(
                x=reg_data[x_reg].to_numpy()[order],
                y=pred[order], mode="lines", name="Doğrusal model",
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ANOMALİ MERKEZİ
# ============================================================

with tabs[3]:
    st.subheader("Anomali Merkezi")

    if not nums:
        st.info("Aykırı değer analizi için sayısal sütun gerekli.")
    else:
        col = st.selectbox("Analiz sütunu", nums, key="anomaly_col")
        method = st.radio(
            "Yöntem",
            ["IQR", "Z-skoru", "Isolation Forest"],
            horizontal=True,
        )

        series = df[col]
        valid = series.dropna()
        mask = pd.Series(False, index=df.index)

        if len(valid) >= 4:
            if method == "IQR":
                q1, q3 = valid.quantile([0.25, 0.75])
                spread = q3 - q1
                mask = (
                    (series < q1 - 1.5 * spread)
                    | (series > q3 + 1.5 * spread)
                ).fillna(False)

            elif method == "Z-skoru":
                std = valid.std()
                if std and np.isfinite(std):
                    z = (series - valid.mean()) / std
                    mask = (z.abs() > 3).fillna(False)

            elif len(valid) >= 10:
                iso = IsolationForest(
                    contamination=0.05, random_state=42
                )
                labels = iso.fit_predict(valid.to_numpy().reshape(-1, 1))
                mask.loc[valid.index] = labels == -1

        result = df.copy()
        result["Olası_Aykırı"] = mask
        st.metric("İşaretlenen satır", int(mask.sum()))
        st.dataframe(result[mask], use_container_width=True)

        fig = px.scatter(
            df.reset_index(), x="index", y=col,
            color=mask.reset_index(drop=True).map(
                {True: "Olası aykırı", False: "Diğer"}
            ),
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Aykırı değer sonuçlarını indir",
            csv_bytes(result),
            file_name="ES_aykiri_degerler.csv",
            mime="text/csv",
        )


# ============================================================
# KALİTE VE FABRİKA
# ============================================================

with tabs[4]:
    st.subheader("Kalite Kontrol ve Fabrika Analizi")
    st.caption(
        "Kontrol sınırları süreç verisinden hesaplanan yaklaşık "
        "istatistiksel sınırlardır; teknik spesifikasyon değildir."
    )

    if not nums:
        st.info("Kalite analizi için sayısal ölçüm sütunu gerekli.")
    else:
        measurement = st.selectbox(
            "Ölçüm değişkeni", nums, key="qc_measurement"
        )
        s = df[measurement].dropna()

        if len(s) >= 2:
            mean = s.mean()
            std = s.std(ddof=1)
            lcl = mean - 3 * std
            ucl = mean + 3 * std
            out = (s < lcl) | (s > ucl)

            a, b, c = st.columns(3)
            a.metric("Ortalama", f"{mean:.4f}")
            b.metric("Alt istatistik sınırı", f"{lcl:.4f}")
            c.metric("Üst istatistik sınırı", f"{ucl:.4f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(s))), y=s.to_numpy(),
                mode="lines+markers", name="Ölçüm",
            ))
            fig.add_hline(y=mean, line_dash="dash", annotation_text="Ortalama")
            fig.add_hline(y=ucl, line_dash="dot", line_color="red",
                          annotation_text="Üst sınır")
            fig.add_hline(y=lcl, line_dash="dot", line_color="red",
                          annotation_text="Alt sınır")
            fig.update_layout(
                template="plotly_dark",
                title="Bireysel ölçümler kontrol görünümü",
                xaxis_title="Gözlem sırası",
                yaxis_title=measurement,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Sınır dışı gözlem", int(out.sum()))

            st.markdown("### Teknik tolerans ve süreç yeterliliği")
            low = st.number_input("Alt spesifikasyon sınırı (LSL)", value=float(mean - 3 * std))
            high = st.number_input("Üst spesifikasyon sınırı (USL)", value=float(mean + 3 * std))

            if high > low and std > 0:
                cp = (high - low) / (6 * std)
                cpk = min((high - mean) / (3 * std), (mean - low) / (3 * std))
                c1, c2 = st.columns(2)
                c1.metric("Cp", f"{cp:.3f}")
                c2.metric("Cpk", f"{cpk:.3f}")
                st.caption(
                    "Cp/Cpk yorumları süreç kararlılığı, dağılım varsayımları "
                    "ve geçerli teknik spesifikasyon gerektirir. Varsayılan "
                    "sınırlar yalnızca örnek başlangıç değeridir."
                )


# ============================================================
# PROTOTİP TESTLERİ
# ============================================================

with tabs[5]:
    st.subheader("Prototip ve Yarışma Test Laboratuvarı")
    st.write(
        "Test sonuçlarını girerek performans özeti ve kriter bazlı "
        "değerlendirme oluştur."
    )

    prototype_name = st.text_input("Prototip adı", "Prototip A")
    test_name = st.text_input("Test adı", "Fonksiyon testi")
    target = st.number_input("Hedef değer", value=100.0)
    tolerance = st.number_input(
        "İzin verilen mutlak sapma", min_value=0.0, value=5.0
    )
    measurements = st.text_area(
        "Ölçüm değerleri (virgül veya satır ile ayır)",
        "98, 101, 99, 102, 100",
    )

    parsed = []
    for token in re.split(r"[,\n;]+", measurements):
        token = token.strip().replace(",", ".")
        if token:
            try:
                parsed.append(float(token))
            except ValueError:
                pass

    if parsed:
        arr = np.array(parsed, dtype=float)
        arr = arr[np.isfinite(arr)]

        if len(arr):
            errors = arr - target
            passed = np.abs(errors) <= tolerance

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test adedi", len(arr))
            c2.metric("Ortalama", f"{arr.mean():.3f}")
            c3.metric("Standart sapma", f"{arr.std(ddof=1) if len(arr)>1 else 0:.3f}")
            c4.metric("Hedef toleransı içinde", f"{passed.mean()*100:.1f}%")

            test_df = pd.DataFrame({
                "Prototip": prototype_name,
                "Test": test_name,
                "Ölçüm": arr,
                "Hedef": target,
                "Sapma": errors,
                "Tolerans içinde": passed,
            })

            st.dataframe(test_df, use_container_width=True)
            st.download_button(
                "Test sonuçlarını CSV indir",
                csv_bytes(test_df),
                file_name="ES_prototip_testi.csv",
                mime="text/csv",
            )

            fig = px.scatter(
                test_df.reset_index(), x="index", y="Ölçüm",
                color="Tolerans içinde",
                title=f"{prototype_name} — {test_name}",
            )
            fig.add_hline(y=target, line_dash="dash",
                          annotation_text="Hedef")
            fig.add_hline(y=target + tolerance, line_dash="dot",
                          annotation_text="Üst tolerans")
            fig.add_hline(y=target - tolerance, line_dash="dot",
                          annotation_text="Alt tolerans")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geçerli sayısal ölçüm bulunamadı.")
    else:
        st.info("Test ölçümlerini gir.")


# ============================================================
# EKONOMİK ANALİZ
# ============================================================

with tabs[6]:
    st.subheader("Ekonomik ve Maliyet Analizi")

    c1, c2 = st.columns(2)
    unit_cost = c1.number_input(
        "Birim üretim maliyeti (TL)", min_value=0.0, value=50.0
    )
    sale_price = c2.number_input(
        "Birim satış fiyatı (TL)", min_value=0.0, value=80.0
    )

    c3, c4 = st.columns(2)
    fixed_cost = c3.number_input(
        "Toplam sabit maliyet (TL)", min_value=0.0, value=5000.0
    )
    expected_units = c4.number_input(
        "Planlanan satış adedi", min_value=0, value=250, step=10
    )

    contribution = sale_price - unit_cost
    break_even = (
        math.ceil(fixed_cost / contribution)
        if contribution > 0 else None
    )
    revenue = sale_price * expected_units
    variable = unit_cost * expected_units
    profit = revenue - variable - fixed_cost

    a, b, c = st.columns(3)
    a.metric("Birim katkı payı", f"{contribution:,.2f} TL")
    b.metric(
        "Başabaş satış adedi",
        f"{break_even:,}" if break_even is not None else "Hesaplanamaz",
    )
    c.metric("Tahmini kâr / zarar", f"{profit:,.2f} TL")

    if contribution <= 0:
        st.warning(
            "Birim satış fiyatı maliyeti aşmıyor. Bu varsayımlarla "
            "başabaş noktası hesaplanamaz."
        )

    economic_df = pd.DataFrame({
        "Gösterge": [
            "Satış geliri", "Değişken maliyet",
            "Sabit maliyet", "Tahmini kâr/zarar",
        ],
        "TL": [revenue, variable, fixed_cost, profit],
    })
    st.dataframe(economic_df, use_container_width=True)

    st.markdown("### Fiyat ve maliyet senaryosu")
    scenario_margin = st.slider(
        "Birim maliyet değişimi (%)", -30, 50, 0
    )
    scenario_price = st.slider(
        "Satış fiyatı değişimi (%)", -30, 50, 0
    )

    scenario_cost = unit_cost * (1 + scenario_margin / 100)
    scenario_sale = sale_price * (1 + scenario_price / 100)
    scenario_profit = (
        scenario_sale * expected_units
        - scenario_cost * expected_units
        - fixed_cost
    )
    st.metric("Senaryo kârı / zararı", f"{scenario_profit:,.2f} TL")


# ============================================================
# ZAMAN SERİSİ
# ============================================================

with tabs[7]:
    st.subheader("Zaman Serisi Analizi")

    date_cols = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if cats:
        date_cols += [c for c in cats if c not in date_cols]

    if not date_cols or not nums:
        st.info("Zaman serisi için tarih ve sayısal değer sütunu gerekli.")
    else:
        date_col = st.selectbox("Tarih sütunu", date_cols)
        value_col = st.selectbox("Ölçüm sütunu", nums)
        window = st.slider("Hareketli ortalama penceresi", 2, 30, 7)

        ts = df[[date_col, value_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
        ts = ts.dropna().sort_values(date_col)

        if len(ts) >= 2:
            ts["Hareketli Ortalama"] = ts[value_col].rolling(
                window=window, min_periods=1
            ).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts[date_col], y=ts[value_col],
                mode="lines", name=value_col,
            ))
            fig.add_trace(go.Scatter(
                x=ts[date_col], y=ts["Hareketli Ortalama"],
                mode="lines", name="Hareketli ortalama",
            ))
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ts, use_container_width=True)


# ============================================================
# VERİ KALİTESİ
# ============================================================

with tabs[8]:
    st.subheader("Veri Kalitesi Merkezi")

    quality = pd.DataFrame({
        "Sütun": df.columns,
        "Veri tipi": [str(df[c].dtype) for c in df.columns],
        "Eksik": [int(df[c].isna().sum()) for c in df.columns],
        "Eksik %": [
            round(float(df[c].isna().mean() * 100), 2)
            for c in df.columns
        ],
        "Benzersiz değer": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })

    st.dataframe(quality, use_container_width=True)

    st.markdown("### Temizlenmiş veriyi indir")
    st.download_button(
        "CSV indir",
        csv_bytes(df),
        file_name="ES_temizlenmis_veri.csv",
        mime="text/csv",
    )

    st.download_button(
        "Excel indir",
        excel_bytes(df, local_report(df, findings), profile),
        file_name="ES_temizlenmis_veri.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ============================================================
# RAPOR MERKEZİ
# ============================================================

with tabs[9]:
    st.subheader("Rapor Merkezi")

    question = st.text_area(
        "Rapor için özel soru veya odak",
        placeholder=(
            "Örnek: Hata oranındaki değişimleri değerlendir ve "
            "üretim kalitesini iyileştirmek için öneriler sun."
        ),
    )

    if st.button("Raporu oluştur", type="primary"):
        with st.spinner("Rapor hazırlanıyor..."):
            report_text, report_source = create_report(
                df, findings, api_key, ai_model, question
            )
        st.session_state["es_report_text"] = report_text
        st.session_state["es_report_source"] = report_source

    report_text = st.session_state.get("es_report_text")
    report_source = st.session_state.get("es_report_source")

    if report_text:
        st.caption(f"Rapor kaynağı: {report_source}")
        st.text_area(
            "Oluşturulan yazılı rapor",
            value=report_text,
            height=380,
        )

        st.download_button(
            "Metin raporunu indir",
            data=report_text.encode("utf-8-sig"),
            file_name="ES_analiz_raporu.md",
            mime="text/markdown; charset=utf-8",
        )

        st.download_button(
            "PDF raporunu indir",
            data=pdf_bytes("E.Ş. Örüntü Radarı", report_text, df),
            file_name="ES_analiz_raporu.pdf",
            mime="application/pdf",
        )

        st.download_button(
            "Rapor ve veriyi Excel indir",
            data=excel_bytes(df, report_text, data_profile(df)),
            file_name="ES_analiz_paketi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Rapor oluşturmak için yukarıdaki düğmeye bas.")

st.divider()
st.caption(
    "E.Ş. Örüntü Radarı • İstatistiksel karar desteği. "
    "Sonuçları kritik uygulamalarda bağımsız olarak doğrulayın."
)
