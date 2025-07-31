# main.py
import streamlit.web.cli as stcli
import sys
import os

def main():
    # Locate the bundled path to graphApp.py
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    app_path = os.path.join(base_path, "graphApp.py")

    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
