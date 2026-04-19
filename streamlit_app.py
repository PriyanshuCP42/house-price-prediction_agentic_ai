"""Streamlit Cloud entrypoint.

Keep the production app in advisory_app.py while exposing the conventional
streamlit_app.py file that Streamlit Community Cloud auto-detects.
"""

# Native entrypoint for Streamlit Cloud.
# We use exec() to ensure the app logic re-runs completely on every interaction
# without duplicating elements in the system sys.modules.

with open("advisory_app.py", encoding="utf-8") as f:
    exec(f.read())
