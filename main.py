import subprocess
import sys
import os

def main():
    # Path to the streamlit app script
    streamlit_script = os.path.join("ui", "streamlit.py")
    
    if not os.path.exists(streamlit_script):
        print(f"Error: Could not find {streamlit_script}")
        sys.exit(1)

    print(f"Starting Streamlit app from {streamlit_script}...")
    
    # Set PYTHONPATH to root directory to allow core/app imports
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")
    
    # Run streamlit via subprocess to avoid internal API issues
    try:
        subprocess.run(["streamlit", "run", streamlit_script], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
