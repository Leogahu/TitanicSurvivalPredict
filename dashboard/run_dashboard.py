# dashboard/run_dashboard.py
import streamlit.web.cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "dashboard/app.py"]
    sys.exit(stcli.main())