import os
import sys
import platform
import urllib.request
import tarfile
import shutil

SPARK_VERSION = "3.5.3"
SPARK_URL = f"https://archive.apache.org/dist/spark/spark-{SPARK_VERSION}/spark-{SPARK_VERSION}-bin-hadoop3.tgz"
SPARK_DIR_NAME = f"spark-{SPARK_VERSION}-bin-hadoop3"
INSTALL_DIR = os.path.dirname(os.path.abspath(__file__))
SPARK_HOME = os.path.join(INSTALL_DIR, SPARK_DIR_NAME)

def download_spark():
    """Download Spark from the specified URL."""
    print(f"Downloading Spark {SPARK_VERSION} from {SPARK_URL}...")
    try:
        urllib.request.urlretrieve(SPARK_URL, f"spark-{SPARK_VERSION}.tar.gz")
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download Spark: {e}")
        sys.exit(1)

def extract_spark():
    """Extract the downloaded Spark archive."""
    print(f"Extracting Spark to {INSTALL_DIR}...")
    try:
        with tarfile.open(f"spark-{SPARK_VERSION}.tar.gz", "r:gz") as tar:
            tar.extractall(path=INSTALL_DIR)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract Spark: {e}")
        sys.exit(1)

def setup_environment():
    """Set up environment variables for Spark."""
    print("Setting up environment variables...")
    try:
        os.environ["SPARK_HOME"] = SPARK_HOME
        os.environ["PATH"] = f"{os.path.join(SPARK_HOME, 'bin')}:{os.environ['PATH']}"
        shell_profile = os.path.expanduser("~/.bashrc") if platform.system() != "Windows" else os.path.expanduser("~/.bash_profile")
        with open(shell_profile, "a") as f:
            f.write(f"\nexport SPARK_HOME={SPARK_HOME}\n")
            f.write(f'export PATH="$SPARK_HOME/bin:$PATH"\n')
        
        print("Environment setup complete.")
    except Exception as e:
        print(f"Failed to set up environment: {e}")
        sys.exit(1)

def cleanup():
    """Clean up temporary files."""
    print("Cleaning up temporary files...")
    try:
        os.remove(f"spark-{SPARK_VERSION}.tar.gz")
        print("Cleanup complete.")
    except Exception as e:
        print(f"Failed to clean up: {e}")

def main():
    """Main function to set up Spark."""
    if os.path.exists(SPARK_HOME):
        print(f"Spark is already installed at {SPARK_HOME}.")
        return

    download_spark()
    extract_spark()
    setup_environment()
    cleanup()
    
    print(f"Spark {SPARK_VERSION} has been successfully installed at {SPARK_HOME}.")

if __name__ == "__main__":
    main()