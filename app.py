"""
EduPulse AI - Student Analytics & Success Platform
Main Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_student_data, get_filtered_students, get_cohort_kpis
from src.model_utils import load_model, predict_student_risk, explain_student_indicators, predict_placement_risk

from src.recommendations import generate_recommendations

# --- Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="EduPulse AI | Student Analytics & Success Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark theme accents, glassmorphic cards, gradient headers)
st.markdown("""
<style>
    /* Main Background & Fonts */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Header Gradient Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .header-title {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #c7d2fe;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* KPI Card Style */
    .kpi-card {
        background: #1e293b;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        text-align: center;
    }

    .kpi-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    .kpi-subtext {
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* Risk Badges */
    .risk-high {
        background-color: #ef4444;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .risk-medium {
        background-color: #f59e0b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .risk-low {
        background-color: #10b981;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    /* Recommendation Cards */
    .rec-card {
        background-color: #1e293b;
        border-left: 5px solid #6366f1;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.9rem;
        border: 1px solid #334155;
    }

    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .rec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .rec-details {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Data & Model Artifact ---
@st.cache_data
def get_cached_data():
    return load_student_data()

@st.cache_resource
def get_cached_model():
    return load_model()

df = get_cached_data()
artifact = get_cached_model()

# --- Sidebar Controls & Navigation ---
st.sidebar.image("https://img.icons8.com/isometric/96/000000/graduation-cap.png", width=60)
st.sidebar.title("EduPulse AI Navigation")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Portal View:",
    [
        "📊 Cohort Analytics Overview",
        "🎓 Student Risk Predictor & Deep-Dive",
        "🔮 Interactive What-If Simulator",
        "📁 Student Data Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Cohort Data")
selected_major = st.sidebar.selectbox(
    "Department / Major:",
    ["All Majors"] + sorted(list(df["major"].unique()))
)

selected_grade = st.sidebar.selectbox(
    "Grade Level:",
    ["All Grades"] + list(df["grade_level"].unique())
)

selected_risk_filter = st.sidebar.selectbox(
    "Risk Level Filter:",
    ["All Risk Levels", "High", "Medium", "Low"]
)

# Apply Sidebar Filters to Data
filtered_df = get_filtered_students(
    df,
    major=selected_major,
    grade_level=selected_grade,
    risk_level=selected_risk_filter
)

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **Demo Disclaimer**: EduPulse AI uses synthetic student data (`students_1200.csv`) and a prototype ML model for hackathon demonstration. It is not validated for real-world academic decisions.")


# --- Top Banner Header ---
st.markdown("""
<div class="header-banner">
    <div class="header-title">🎓 EduPulse AI - Student Analytics & Success Platform</div>
    <div class="header-subtitle">Early Academic Risk Detection & Intelligent Support Recommendation Engine</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# VIEW 1: COHORT ANALYTICS OVERVIEW
# ==========================================
if app_mode == "📊 Cohort Analytics Overview":
    kpis = get_cohort_kpis(filtered_df)

    # KPI Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Students</div>
            <div class="kpi-value">{kpis['total']}</div>
            <div class="kpi-subtext" style="color: #94a3b8;">Active Cohort</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High Risk Count</div>
            <div class="kpi-value" style="color: #ef4444;">{kpis['high_risk_cnt']}</div>
            <div class="kpi-subtext" style="color: #ef4444;">{kpis['high_risk_pct']}% of total</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Success Score</div>
            <div class="kpi-value" style="color: #6366f1;">{kpis['avg_success_score']}</div>
            <div class="kpi-subtext" style="color: #94a3b8;">0-100 Cohort Avg</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Cohort Avg GPA</div>
            <div class="kpi-value" style="color: #38bdf8;">{kpis['avg_gpa']}</div>
            <div class="kpi-subtext" style="color: #94a3b8;">4.0 Scale</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Attendance</div>
            <div class="kpi-value" style="color: #10b981;">{kpis['avg_attendance']}%</div>
            <div class="kpi-subtext" style="color: #94a3b8;">Target: 80%+</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Assignment Comp.</div>
            <div class="kpi-value" style="color: #a855f7;">{kpis['avg_assignment']}%</div>
            <div class="kpi-subtext" style="color: #94a3b8;">Target: 85%+</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1: Donut Distribution & Scatter Plot
    c1, c2 = st.columns([1.2, 1.8])

    with c1:
        st.subheader("🎯 Academic Risk Distribution")
        risk_counts = filtered_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}

        fig_donut = px.pie(
            risk_counts,
            values="Count",
            names="Risk Level",
            hole=0.5,
            color="Risk Level",
            color_discrete_map=colors
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_donut, width="stretch")

    with c2:
        st.subheader("📈 GPA vs. Attendance Rate by Risk")
        fig_scatter = px.scatter(
            filtered_df,
            x="attendance_rate",
            y="gpa",
            color="risk_level",
            size="study_hours_per_week",
            hover_data=["student_id", "name", "major"],
            labels={"attendance_rate": "Attendance Rate (%)", "gpa": "GPA", "study_hours_per_week": "Study Hrs/Wk"},
            color_discrete_map=colors
        )
        fig_scatter.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # Charts Row 2: Department & Grade Breakdown
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("🏢 Risk Breakdown by Major")
        dept_risk = pd.crosstab(filtered_df["major"], filtered_df["risk_level"]).reset_index()
        fig_dept = px.bar(
            dept_risk,
            x="major",
            y=["High", "Medium", "Low"],
            color_discrete_map=colors,
            labels={"value": "Student Count", "major": "Major"}
        )
        fig_dept.update_layout(
            barmode="stack",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc"),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_dept, width="stretch")

    with c4:
        st.subheader("🎓 Risk Breakdown by Grade Level")
        grade_risk = pd.crosstab(filtered_df["grade_level"], filtered_df["risk_level"]).reset_index()
        fig_grade = px.bar(
            grade_risk,
            x="grade_level",
            y=["High", "Medium", "Low"],
            color_discrete_map=colors,
            labels={"value": "Student Count", "grade_level": "Grade Level"}
        )
        fig_grade.update_layout(
            barmode="group",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc"),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_grade, width="stretch")


# ==========================================
# VIEW 2: STUDENT RISK PREDICTOR & DEEP-DIVE
# ==========================================
elif app_mode == "🎓 Student Risk Predictor & Deep-Dive":
    st.subheader("🔍 Select Student Record")

    search_col, select_col = st.columns([1, 2])
    with search_col:
        search_query = st.text_input("Filter student by Name or ID:", placeholder="e.g. EDU-2026-1015 or Alex")

    selectable_df = get_filtered_students(df, search_query=search_query) if search_query else df

    student_options = [f"{row['student_id']} - {row['name']} ({row['major']})" for _, row in selectable_df.iterrows()]

    if not student_options:
        st.warning("No student matching your query was found.")
        st.stop()

    with select_col:
        selected_option = st.selectbox("Choose Student:", student_options)

    selected_id = selected_option.split(" - ")[0]
    student = df[df["student_id"] == selected_id].iloc[0]

    # Run Prediction & Feature Explanation
    pred = predict_student_risk(student, artifact)
    placement_pred = predict_placement_risk(student)
    indicators = explain_student_indicators(student, artifact)
    recommendations = generate_recommendations(student, pred, placement_pred)

    st.markdown("---")

    # Student Header Card & Risk Summary
    sc1, sc2 = st.columns([1.5, 2.5])

    with sc1:
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid #334155;">
            <h3 style="color: #38bdf8; margin-top:0;">👤 {student['name']}</h3>
            <p><strong>Student ID:</strong> {student['student_id']}</p>
            <p><strong>Major:</strong> {student['major']}</p>
            <p><strong>Grade Level:</strong> {student['grade_level']}</p>
            <p><strong>Success Score:</strong> <span style="color: #6366f1; font-weight: bold;">{pred['success_score']} / 100</span></p>
            <p><strong>Tuition Status:</strong> <span style="color: {'#10b981' if student['tuition_status']=='Paid' else '#ef4444'}; font-weight: bold;">{student['tuition_status']}</span></p>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        risk_class = pred["predicted_risk"]
        color_hex = "#ef4444" if risk_class == "High" else "#f59e0b" if risk_class == "Medium" else "#10b981"
        badge_symbol = "🔴" if risk_class == "High" else "🟡" if risk_class == "Medium" else "🟢"

        p_risk = placement_pred["placement_risk"]
        p_color = "#ef4444" if p_risk == "High" else "#f59e0b" if p_risk == "Medium" else "#10b981"
        p_symbol = "🔴" if p_risk == "High" else "🟡" if p_risk == "Medium" else "🟢"

        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 1.2rem; border-radius: 12px; border: 2px solid {color_hex}; text-align: center;">
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Academic Risk Level</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: {color_hex}; margin: 0.2rem 0;">
                        {badge_symbol} {risk_class.upper()} RISK
                    </div>
                </div>
                <div style="border-left: 1px solid #334155; height: 50px;"></div>
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Placement Readiness Risk</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: {p_color}; margin: 0.2rem 0;">
                        {p_symbol} {p_risk.upper()} RISK
                    </div>
                    <div style="font-size: 0.78rem; color: #cbd5e1;">Readiness Score: {placement_pred['placement_score']}/100</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-around; margin-top: 0.8rem; padding-top: 0.6rem; border-top: 1px solid #334155;">
                <div><span style="color:#ef4444; font-weight:bold;">High Risk Prob:</span> {pred['high_risk_prob']}%</div>
                <div><span style="color:#f59e0b; font-weight:bold;">Medium Risk Prob:</span> {pred['medium_risk_prob']}%</div>
                <div><span style="color:#10b981; font-weight:bold;">Low Risk Prob:</span> {pred['low_risk_prob']}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Indicators & Breakdown Section
    st.subheader("💡 Key Academic Indicators & Model Explanations")

    ind_col1, ind_col2 = st.columns([1.8, 1.2])

    with ind_col1:
        st.markdown("#### Feature Metrics vs. Cohort Average")
        ind_df = pd.DataFrame(indicators)

        # Horizontal impact chart
        fig_ind = px.bar(
            ind_df,
            x="impact_score",
            y="feature_name",
            orientation="h",
            color="status",
            color_discrete_map={
                "Risk Driver": "#ef4444",
                "Average": "#64748b",
                "Positive Strength": "#10b981"
            },
            labels={"impact_score": "Relative Indicator Impact Score", "feature_name": "Indicator"},
            hover_data=["student_value", "cohort_avg", "badge"]
        )
        fig_ind.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc"),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_ind, width="stretch")

    with ind_col2:
        st.markdown("#### Indicator Summary Table")
        display_ind = ind_df[["badge", "feature_name", "student_value", "cohort_avg"]].copy()
        display_ind.columns = ["Status", "Metric", "Student Value", "Cohort Avg"]
        st.dataframe(display_ind, width="stretch", hide_index=True)

    st.markdown("---")

    # Actionable Academic Support Recommendations Section
    st.subheader("🛠️ Recommended Academic Support Interventions")

    for rec in recommendations:
        priority_color = "#ef4444" if "High" in rec["priority"] or "Critical" in rec["priority"] else "#f59e0b" if "Medium" in rec["priority"] else "#38bdf8"

        st.markdown(f"""
        <div class="rec-card" style="border-left-color: {priority_color};">
            <div class="rec-header">
                <span class="rec-title">{rec['action']}</span>
                <span style="background-color: {priority_color}; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
                    {rec['priority']}
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.3rem;">
                CATEGORY: {rec['category'].upper()}
            </div>
            <div class="rec-details">
                {rec['details']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# VIEW 3: INTERACTIVE WHAT-IF SIMULATOR
# ==========================================
elif app_mode == "🔮 Interactive What-If Simulator":
    st.subheader("🔮 Interventions Simulator (What-If Scenario)")
    st.markdown("Test how targeted academic interventions (e.g., increasing attendance or study hours) impact a student's predicted risk level in real-time.")

    sim_student_id = st.selectbox("Select Student to Simulate:", df["student_id"] + " - " + df["name"])
    sid = sim_student_id.split(" - ")[0]
    base_student = df[df["student_id"] == sid].iloc[0]

    original_pred = predict_student_risk(base_student, artifact)

    st.markdown("---")
    sim_col1, sim_col2 = st.columns([1.2, 1.8])

    with sim_col1:
        st.markdown("### 🎛️ Adjust Student Parameters")

        sim_attendance = st.slider(
            "Attendance Rate (%)",
            min_value=40.0, max_value=100.0,
            value=float(base_student["attendance_rate"]), step=1.0
        )

        sim_study_hours = st.slider(
            "Study Hours / Week",
            min_value=2.0, max_value=38.0,
            value=float(base_student["study_hours_per_week"]), step=1.0
        )

        sim_assignment = st.slider(
            "Assignment Completion (%)",
            min_value=40.0, max_value=100.0,
            value=float(base_student["assignment_completion_rate"]), step=1.0
        )

        sim_midterm = st.slider(
            "Midterm Score (%)",
            min_value=35.0, max_value=100.0,
            value=float(base_student["midterm_score"]), step=1.0
        )

        sim_stress = st.slider(
            "Stress Level (1-10)",
            min_value=1, max_value=10,
            value=int(base_student["stress_level"]), step=1
        )

    # Construct simulated student profile
    sim_student = base_student.to_dict()
    sim_student["attendance_rate"] = sim_attendance
    sim_student["study_hours_per_week"] = sim_study_hours
    sim_student["assignment_completion_rate"] = sim_assignment
    sim_student["midterm_score"] = sim_midterm
    sim_student["stress_level"] = sim_stress

    sim_pred = predict_student_risk(sim_student, artifact)

    with sim_col2:
        st.markdown("### 📊 Simulation Results Comparison")

        orig_risk = original_pred["predicted_risk"]
        sim_risk = sim_pred["predicted_risk"]

        orig_color = "#ef4444" if orig_risk == "High" else "#f59e0b" if orig_risk == "Medium" else "#10b981"
        sim_color = "#ef4444" if sim_risk == "High" else "#f59e0b" if sim_risk == "Medium" else "#10b981"

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.markdown(f"""
            <div style="background-color: #1e293b; padding: 1.2rem; border-radius: 10px; border-top: 4px solid {orig_color}; text-align: center;">
                <div style="font-size: 0.85rem; color: #94a3b8;">ORIGINAL RISK LEVEL</div>
                <div style="font-size: 1.6rem; font-weight: bold; color: {orig_color};">{orig_risk.upper()}</div>
                <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">High Risk Prob: {original_pred['high_risk_prob']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown(f"""
            <div style="background-color: #1e293b; padding: 1.2rem; border-radius: 10px; border-top: 4px solid {sim_color}; text-align: center;">
                <div style="font-size: 0.85rem; color: #94a3b8;">SIMULATED RISK LEVEL</div>
                <div style="font-size: 1.6rem; font-weight: bold; color: {sim_color};">{sim_risk.upper()}</div>
                <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">High Risk Prob: {sim_pred['high_risk_prob']}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk probability shift comparison bar chart
        prob_comp_df = pd.DataFrame([
            {"Scenario": "Original Baseline", "High Risk": original_pred['high_risk_prob'], "Medium Risk": original_pred['medium_risk_prob'], "Low Risk": original_pred['low_risk_prob']},
            {"Scenario": "Simulated Intervention", "High Risk": sim_pred['high_risk_prob'], "Medium Risk": sim_pred['medium_risk_prob'], "Low Risk": sim_pred['low_risk_prob']}
        ])

        fig_prob = px.bar(
            prob_comp_df,
            x="Scenario",
            y=["High Risk", "Medium Risk", "Low Risk"],
            color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"},
            title="Risk Probability Distribution Comparison (%)"
        )
        fig_prob.update_layout(
            barmode="group",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_prob, width="stretch")


# ==========================================
# VIEW 4: STUDENT DATA EXPLORER
# ==========================================
elif app_mode == "📁 Student Data Explorer":
    st.subheader("📁 Cohort Student Data Directory")
    st.markdown(f"Displaying **{len(filtered_df)}** records based on active sidebar filters.")

    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True
    )

    # Download CSV button (Sanitized against CSV formula injection)
    export_df = filtered_df.copy()
    for col in export_df.select_dtypes(include=['object', 'string']).columns:
        export_df[col] = export_df[col].astype(str).apply(
            lambda x: f"'{x}" if x and x[0] in ('=', '+', '-', '@') else x
        )
    csv_data = export_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Filtered Student Dataset (CSV)",
        data=csv_data,
        file_name="edupulse_student_records.csv",
        mime="text/csv"
    )
