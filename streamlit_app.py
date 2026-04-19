"""Streamlit Cloud entrypoint.

Keep the production app in advisory_app.py while exposing the conventional
streamlit_app.py file that Streamlit Community Cloud auto-detects.
"""

with open("advisory_app.py", encoding="utf-8") as f:
    exec(f.read())
