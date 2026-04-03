# Streamlit Cloud entry point
# This file redirects to the actual frontend application.
# Streamlit Cloud runs this from the repository root.

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

# Execute the frontend app
exec(open(os.path.join(os.path.dirname(__file__), "frontend", "app.py")).read())
