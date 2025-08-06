import imaplib
import smtplib
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
import pandas as pd
import wmi  # Windows-specific hardware info

# --- System Info ---
def get_system_temps():
    try:
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        temps = {}
        for sensor in w.Sensor():
            if sensor.SensorType == u'Temperature':
                temps[sensor.Name] = sensor.Value
        return temps
    except Exception as e:
        return {"error": str(e)}

# --- ETA Estimation ---
def get_eta(start_time, current_epoch, total_epochs, avg_iter_time):
    if current_epoch == 0:
        return "Unknown"
    elapsed = (datetime.now() - start_time).total_seconds()
    remaining_epochs = total_epochs - current_epoch
    est_total_time = avg_iter_time * remaining_epochs
    return str(pd.to_timedelta(elapsed + est_total_time, unit='s'))

# --- Email Monitor Class ---
class EmailMonitor:
    def __init__(self, config_path="config/train_config.json", train_config_path="config/train_config.json", external_stop_event=None, on_training_stop=None):
        with open(config_path) as f:
            cfg = json.load(f)

        self.sender = cfg["sender_email"]
        self.receiver = cfg["receiver_email"]
        self.password = cfg["email_password"]
        self.imap_server = cfg.get("imap", "imap.gmail.com")
        self.smtp_server = cfg.get("smtp_server", "smtp.gmail.com")
        self.check_interval = cfg.get("check_interval", 20)  # seconds

        self.train_config_path = Path(train_config_path)
        self.last_reported_epoch = -1

        self.external_stop_event = external_stop_event  # ✅ Save reference here
        self.on_training_stop = on_training_stop

    def read_train_config(self):
        try:
            if not self.train_config_path.exists():
                return None
            with open(self.train_config_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading training config: {e}")
            return None

    def write_train_config(self, data):
        try:
            with open(self.train_config_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error writing training config: {e}")

    def send_email(self, subject, body):
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender
            msg["To"] = self.receiver
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(self.smtp_server, 587)
            server.starttls()
            server.login(self.sender, self.password)
            server.send_message(msg)
            server.quit()
            print(f"\U0001F4E7 Sent email: {subject}")
        except Exception as e:
            print(f"Failed to send email: {e}")

    def check_email_commands(self):
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.sender, self.password)
            mail.select("inbox")

            status, data = mail.uid("search", f'(UNSEEN FROM "{self.receiver}")')
            if status != "OK":
                mail.logout()
                return

            for uid in data[0].split():
                status, msg_data = mail.uid("fetch", uid, "(RFC822)")
                if status != "OK":
                    continue

                raw_email = msg_data[0][1].decode("utf-8", errors="ignore")
                subject = None
                for line in raw_email.split("\n"):
                    if line.startswith("Subject:"):
                        subject = line[len("Subject:"):].strip()
                        break
                if subject:
                    subject = subject.upper().strip()
                    self.handle_command(subject)
            mail.logout()

        except Exception as e:
            print(f"Email check failed: {e}\n{traceback.format_exc()}")

    def handle_command(self, cmd):
        print(f"\U0001F4E5 Command received: {cmd}")
        config = self.read_train_config()

        if cmd == "SYSTEM_TEMP":
            temps = get_system_temps()
            body = "\n".join(f"{k}: {v} °C" for k, v in temps.items())
            self.send_email("🔥 System Temperatures", body)

        elif cmd == "CURRENT_LOSS":
            if not config:
                self.send_email("📉 Loss Info", "Training config not found.")
                return
            run_dir = Path(config.get("project", "models")) / config.get("name", "frc_bumper_run")
            csv_path = run_dir / "results.csv"
            if not csv_path.exists():
                self.send_email("📉 Loss Info", "results.csv not found.")
                return
            df = pd.read_csv(csv_path)
            last_row = df.tail(1).to_string(index=False)
            self.send_email("📉 Current Losses", last_row)

        elif cmd == "TRAINING_STOP":
            stop_flag = Path("config/stop_training.flag")
            stop_flag.touch()
            if self.external_stop_event:
                self.external_stop_event.set()
            if self.on_training_stop:
                self.on_training_stop()  # <--- call the GUI callback here
            self.send_email("🚩 Training Halt Requested", "Stop flag created. Training should halt soon.")

        else:
            self.send_email("❓ Unknown Command", f"The command '{cmd}' is not recognized.")

    def run_monitor_loop(self):
        print("📱 Email monitor started.")
        while True:
            self.check_email_commands()
            config = self.read_train_config()
            if config:
                epoch = config.get("current_epoch", 0)
                interval = config.get("email_report_interval", 10)
                total_epochs = config.get("epochs", 200)
                avg_iter_time = config.get("avg_iter_time", 5)  # seconds per iteration

                if epoch % interval == 0 and epoch != self.last_reported_epoch:
                    self.last_reported_epoch = epoch

                    temps = get_system_temps()
                    body = f"Epoch {epoch}/{total_epochs}\n\n"
                    body += "System Temps:\n" + "\n".join(f"{k}: {v} °C" for k, v in temps.items()) + "\n"

                    start_time = datetime.fromisoformat(config["start_time"])
                    eta = get_eta(start_time, epoch, total_epochs, avg_iter_time)
                    body += f"\nETA: {eta}"

                    run_dir = Path(config.get("project", "models")) / config.get("name", "frc_bumper_run")
                    csv_path = run_dir / "results.csv"
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        last_loss = df.tail(1).to_string(index=False)
                        body += f"\n\nLatest Loss Info:\n{last_loss}"

                    self.send_email(f"📈 Training Progress Update - Epoch {epoch}", body)

            time.sleep(self.check_interval)

if __name__ == "__main__":
    monitor = EmailMonitor()
    monitor.run_monitor_loop()
