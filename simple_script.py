# simple_script.py

import socket

def main():
    print("Hello from the Python script!")
    print(f"Host name: {socket.gethostname()}")
    print("This is a test to confirm that the job submission works correctly.")

if __name__ == "__main__":
    main()
