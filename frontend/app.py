import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
import yaml
from frontend.components.sidebar import render_sidebar
 