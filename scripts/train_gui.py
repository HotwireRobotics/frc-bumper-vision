import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
from threading import Thread
from ultralytics import YOLO
import sys
import io
import torch
import platform
import json
from datetime import datetime
from pathlib import Path
import logging
from email_handler import EmailMonitor
import time

# Logging to file for debug
logging.basicConfig(
    filename='train_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TrainingSubprocess:
    """Manages the subprocess that runs training (train_runner.py)"""
    def __init__(self, config_path: Path, on_finish=None):
        self.config_path = config_path
        self.process = None
        self.on_finish = on_finish

    def start(self):
        if self.process and self.process.poll() is None:
            print("Training already running.")
            return
        self.process = subprocess.Popen(
            [sys.executable, "train_runner.py", str(self.config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        Thread(target=self._read_output, daemon=True).start()
        Thread(target=self._wait_process, daemon=True).start()
        print("🚀 Training subprocess started.")

    def _read_output(self):
        if not self.process or not self.process.stdout:
            print("No subprocess or stdout to read from.")
            return
        for line in self.process.stdout:
            print(line, end="")  # Redirect to GUI console

    def _wait_process(self):
        if not self.process:
            print("No subprocess to wait for.")
            return
        self.process.wait()
        print(f"Training subprocess exited with code {self.process.returncode}")
        if self.on_finish:
            self.on_finish()

    def stop(self):
        if self.process and self.process.poll() is None:
            print("Stopping training subprocess gracefully.")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                print("Training subprocess terminated gracefully.")
            except subprocess.TimeoutExpired:
                print("Subprocess did not stop in time; killing it.")
                self.process.kill()
        else:
            print("No running subprocess to stop.")

class RedirectText(io.StringIO):
    """Redirects stdout to a Tkinter text widget"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, s):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, s)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

def get_device():
    """Detect if CUDA GPU is available"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA is available: {device_name}")
        return 0
    else:
        print("⚠️ CUDA is NOT available. Using CPU instead.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Platform: {platform.system()} {platform.release()}")
        return 'cpu'

def test_gpu():
    """Show a dialog with GPU info or error if no GPU"""
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            messagebox.showinfo("GPU Info", f"✅ GPU Detected:\n{name}\nCompute Capability: {capability}\nMemory: {mem} GB")
        else:
            raise RuntimeError("CUDA not available or PyTorch not installed with GPU support.")
    except Exception as e:
        messagebox.showerror("GPU Check Failed", f"❌ {e}")

def send_test_email(config_path="config/email_config.json"):
    """Send a simple test email to verify email settings"""
    import json
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import smtplib

    with open(config_path) as f:
        cfg = json.load(f)

    sender = cfg["sender"]
    receiver = cfg["receiver"]
    password = cfg["password"]
    smtp_server = cfg.get("smtp", "smtp.gmail.com")

    subject = "Test Email from YOLO Trainer"
    body = "This is a test email sent to verify email handler is working."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(smtp_server, 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("📧 Test email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send test email: {e}")

class YOLOTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Trainer")
        self._stop_training_flag = threading.Event()
        self._temp_monitor_thread = None
        self.compute_capability = self.get_compute_capability()
        default_opt = self.compute_capability == (8, 9)

        self.optimize_var = tk.BooleanVar(value=default_opt)
        tk.Checkbutton(root, text="Enable RTX 4060 Optimizations (CC 8.9)", variable=self.optimize_var).grid(row=3, column=0, columnspan=3, sticky='w', padx=5)

        # Load training config
        with open("config/train_config.json") as f:
            self.config = json.load(f)

        # Dataset YAML selector
        tk.Label(root, text="Dataset YAML:").grid(row=0, column=0, sticky='w')
        self.dataset_path_var = tk.StringVar(value=self.config.get("data", ""))
        tk.Entry(root, textvariable=self.dataset_path_var, width=50).grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_yaml).grid(row=0, column=2)

        # Buttons: Start, Test GPU, Test Email
        self.train_btn = tk.Button(root, text="Start Training", command=self.start_training)
        self.train_btn.grid(row=1, column=0, pady=10, sticky='w')

        self.test_email_btn = tk.Button(root, text="Test Email", command=send_test_email)
        self.test_email_btn.grid(row=1, column=1, pady=10, sticky='e')

        self.test_gpu_btn = tk.Button(root, text="Test GPU", command=test_gpu)
        self.test_gpu_btn.grid(row=1, column=2, pady=10, sticky='e')

        # Output console
        self.output_console = scrolledtext.ScrolledText(root, width=70, height=20, state='disabled')
        self.output_console.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        # Redirect stdout to console
        self.stdout_backup = sys.stdout
        sys.stdout = RedirectText(self.output_console)

    def get_compute_capability(self):
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_capability(0)
            else:
                return None
        except:
            return None

    def browse_yaml(self):
        file = filedialog.askopenfilename(title="Select dataset.yaml", filetypes=[("YAML files", "*.yaml *.yml")])
        if file:
            self.dataset_path_var.set(file)

    def update_training_config_with_optimization(self):
        config_path = Path("config/train_config.json")
        try:
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_data = json.load(f)
            else:
                config_data = {}
        except Exception as e:
            print(f"❌ Failed to load training config: {e}")
            config_data = {}

        use_opt = self.optimize_var.get()
        config_data["use_optimizations"] = use_opt

        # Increase batch size if optimizations enabled
        if use_opt:
            batch = config_data.get("batch", 16)
            config_data["batch"] = max(batch, 24)  # bump to 24 if less
        else:
            # Optionally reset batch size or leave as is
            config_data["batch"] = config_data.get("batch", 16)

        # Ensure dataset YAML path sync
        config_data["data"] = self.dataset_path_var.get()

        try:
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=4)
            print(f"✅ Updated training config with optimizations={use_opt}")
        except Exception as e:
            print(f"❌ Failed to save training config: {e}")

    def start_training(self):
        device = get_device()
        self.train_btn.config(state='disabled')
        self._stop_training_flag.clear()

        # Update training config before starting subprocess
        self.update_training_config_with_optimization()

        # GPU temp monitor (optional)
        if torch.cuda.is_available():
            self._temp_monitor_thread = threading.Thread(target=self.monitor_gpu_temp, daemon=True)
            self._temp_monitor_thread.start()

        use_opt = self.optimize_var.get()
        print(f"Optimization toggle is {'ON' if use_opt else 'OFF'}")

        # Start email monitor with stop event and GUI callback
        def on_email_training_stop():
            def gui_stop_actions():
                print("🛑 Training stop command received via email.")
                self._stop_training_flag.set()
                self.train_btn.config(state='normal')
                messagebox.showinfo("Training Stopped", "Training was stopped by remote command.")
                if hasattr(self, 'training_subprocess'):
                    self.training_subprocess.stop()
            self.root.after(0, gui_stop_actions)

        self.email_monitor = EmailMonitor(
            external_stop_event=self._stop_training_flag,
            on_training_stop=on_email_training_stop
        )
        threading.Thread(target=self.email_monitor.run_monitor_loop, daemon=True).start()

        # Start the actual training subprocess
        self.training_subprocess = TrainingSubprocess(Path("config/train_config.json"), on_finish=self.on_training_finished)
        self.training_subprocess.start()

    def monitor_gpu_temp(self):
        while not self._stop_training_flag.is_set():
            temp = self.get_gpu_temp()
            if temp is not None:
                print(f"GPU Temperature: {temp}°C")
                if temp >= 85:
                    self.alert_overheat(temp)
                    self._stop_training_flag.set()
                    break
            time.sleep(10)

    def get_gpu_temp(self):
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
            else:
                print(f"nvidia-smi error: {result.stderr.strip()}")
                return None
        except Exception as e:
            print(f"Failed to get GPU temp: {e}")
            return None

    def alert_overheat(self, temp):
        def show_alert():
            messagebox.showwarning(
                "GPU Overheat Warning",
                f"⚠️ GPU temperature has reached {temp}°C!\nTraining will stop to prevent damage."
            )
        self.root.after(0, show_alert)

    def on_training_finished(self):
        print("🟢 Training finished or stopped.")
        self._stop_training_flag.set()
        self.train_btn.config(state='normal')

    def on_close(self):
        self._stop_training_flag.set()
        sys.stdout = self.stdout_backup
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTrainerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
