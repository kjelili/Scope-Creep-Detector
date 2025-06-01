import streamlit as st
import pandas as pd
import os
from extract_text import extract_text_from_file
from scope_rag_checker import check_scope_creep_with_rag
from send_sms import send_sms

st.set_page_config(page_title="Scope Creep Detector (RAG + SMS)", layout="wide")
st.title("📊 Scope Creep Detection with GPT + RAG + SMS Alerts")

api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
stakeholder_phones = st.text_area("📲 Stakeholder Phone Numbers (comma-separated with country codes)", placeholder="+1234567890, +0987654321")

scope_file = st.file_uploader("📄 Upload Scope Document (PDF or DOCX)", type=["pdf", "docx"])
uploaded_file = st.file_uploader("📨 Upload Email CSV (must contain 'email_body' column)", type=["csv"])

if scope_file and uploaded_file and api_key:
    scope_text = extract_text_from_file(scope_file)
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    if "email_body" not in df.columns:
        st.error("CSV must contain 'email_body' column")
    else:
        df["scope_creep"] = ""
        df["relevant_scope"] = ""
        df["justification"] = ""
        df["suggestion"] = ""
        df["risk_level"] = ""
        df["impact_analysis"] = ""

        st.info("⏳ Analyzing emails...")
        for i, row in df.iterrows():
            result = check_scope_creep_with_rag(row["email_body"], scope_text, api_key)
            df.at[i, "scope_creep"] = result.get("scope_creep", "Error")
            df.at[i, "relevant_scope"] = result.get("reference_scope_line", "None")
            df.at[i, "justification"] = result.get("justification", "Error")
            df.at[i, "suggestion"] = result.get("suggestion", "Check logs")
            df.at[i, "risk_level"] = result.get("risk_level", "Unknown")
            df.at[i, "impact_analysis"] = result.get("impact_analysis", "Unknown")

            if result.get("scope_creep", "").lower() == "yes" and result.get("risk_level", "").lower() in ["high", "extreme"]:
                message = f"SCOPE CREEP ALERT [{result.get('risk_level', '').upper()}]:\n{row['email_body'][:200]}...\nSuggested: {result.get('suggestion')}"
                if stakeholder_phones:
                    for number in stakeholder_phones.split(","):
                        send_sms(number.strip(), message)

        st.success("✅ Analysis Complete")
        st.dataframe(df)
        st.download_button("📥 Download Results as CSV", data=df.to_csv(index=False), file_name="scope_creep_results.csv", mime="text/csv")
else:
    st.warning("👆 Please upload both a scope document and email CSV, and enter your OpenAI API key.")
