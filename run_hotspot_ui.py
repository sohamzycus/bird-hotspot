import streamlit as st
import os
import sys

# Add the current directory to the path so we can import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the UI module
from bird_hotspot_ui import *

# The UI is defined in bird_hotspot_ui.py and will run automatically when imported 