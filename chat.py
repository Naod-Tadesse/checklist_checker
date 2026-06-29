import streamlit as st
import pandas as pd
from io import BytesIO
from checker import validate_checklist

st.set_page_config(page_title="Checklist Validator", layout="wide")

st.title("📋 Checklist Validator")
st.write("Upload an Excel file and validate checklist standards.")

uploaded_file = st.file_uploader(
    "Upload Excel file",
    type=["xlsx", "xls"]
)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully")
        st.write("### Data Preview")
        st.dataframe(df.head())

        if st.button("🔍 Run Validation"):
            error_df = validate_checklist(df)

            if error_df.empty:
                st.success("✅ No validation errors found!")
            else:
                st.error(f"❌ {len(error_df)} issues found")
                st.write("### Standards with Errors")
                st.dataframe(error_df, width=1200, height=600)

                buffer = BytesIO()
                error_df.to_excel(buffer, index=False)

                st.download_button(
                    "⬇️ Download Error Report",
                    data=buffer.getvalue(),
                    file_name="validation_errors.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error reading file: {e}")
