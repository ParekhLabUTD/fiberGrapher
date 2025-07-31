import subprocess
import sys
import os

def main():
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, 'graphApp.py')
    subprocess.run(["streamlit", "run", app_path])

if __name__ == '__main__':
    main()
