"""Streamlit Cloud entrypoint.

Keep the production app in advisory_app.py while exposing the conventional
streamlit_app.py file that Streamlit Community Cloud auto-detects.
"""

import sys
from importlib import reload
import advisory_app

# Force reload of advisory_app on every rerun to ensure UI updates
reload(advisory_app)
