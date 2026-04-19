"""Streamlit Cloud entrypoint.

Keep the production app in advisory_app.py while exposing the conventional
streamlit_app.py file that Streamlit Community Cloud auto-detects.
"""

import advisory_app  # noqa: F401
