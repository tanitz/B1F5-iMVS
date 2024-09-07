import tkinter as tk
import subprocess
import os

# Paths to the scripts
path_L = r"/home/code/hikrobot/linux/3cam.py"
path_C = r"/home/code/hikrobot/linux/5Hikrobot.py"
path_R = r"/home/code/hikrobot/linux/2cam.py"

# Dictionary to store the processes
processes = {}

# Function to start the camera script
def run_camera_script(camera_name, script_path):
    if os.path.exists(script_path):
        # Start the script as a subprocess
        process = subprocess.Popen(["python3", script_path])
        processes[camera_name] = process
        print(f"Started {camera_name} script: {script_path}")
    else:
        print(f"{camera_name} script not found!")

# Function to stop the camera script
def stop_camera_script(camera_name):
    process = processes.get(camera_name)
    if process:
        process.terminate()  # Terminate the process
        process.wait()       # Wait for the process to fully terminate
        print(f"Terminated {camera_name} script.")
        processes[camera_name] = None  # Remove the process from the dictionary
    else:
        print(f"No running {camera_name} script to terminate.")

# Create the main window
root = tk.Tk()

# Set the title of the window
root.title("B5 Camera Check")

# Set the size of the window
root.geometry("400x300")  # Width x Height in pixels

title_label = tk.Label(root, text="B5 Camera Check", font=("Helvetica", 16))
title_label.grid(row=0, column=0, columnspan=2, pady=20)

# Buttons for Camera L
start_button1 = tk.Button(root, text="Start Camera L", command=lambda: run_camera_script("Camera L", path_L), font=("Helvetica", 14), width=15)
start_button1.grid(row=1, column=0, padx=10, pady=10)

stop_button1 = tk.Button(root, text="Stop Camera L", command=lambda: stop_camera_script("Camera L"), font=("Helvetica", 14), width=15)
stop_button1.grid(row=1, column=1, padx=10, pady=10)

# Uncomment and modify these lines for Camera C and Camera R if needed
# Buttons for Camera C
start_button2 = tk.Button(root, text="Start Camera C", command=lambda: run_camera_script("Camera C", path_C), font=("Helvetica", 14), width=15)
start_button2.grid(row=2, column=0, padx=10, pady=10)

stop_button2 = tk.Button(root, text="Stop Camera C", command=lambda: stop_camera_script("Camera C"), font=("Helvetica", 14), width=15)
stop_button2.grid(row=2, column=1, padx=10, pady=10)

# Buttons for Camera R
start_button3 = tk.Button(root, text="Start Camera R", command=lambda: run_camera_script("Camera R", path_R), font=("Helvetica", 14), width=15)
start_button3.grid(row=3, column=0, padx=10, pady=10)

stop_button3 = tk.Button(root, text="Stop Camera R", command=lambda: stop_camera_script("Camera R"), font=("Helvetica", 14), width=15)
stop_button3.grid(row=3, column=1, padx=10, pady=10)

# Run the application
root.mainloop()
