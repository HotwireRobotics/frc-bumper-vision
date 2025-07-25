import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from threading import Thread
from ultralytics import YOLO
import sys
import io
import torch
import platform

# --- Console Redirect ---
class RedirectText(io.StringIO):
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

# --- Device Detection ---
def get_device():
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

# --- GUI ---
class YOLOTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Trainer")

        # Dataset YAML selector
        tk.Label(root, text="Dataset YAML:").grid(row=0, column=0, sticky='w')
        self.dataset_path_var = tk.StringVar(value="config/data.yaml")
        tk.Entry(root, textvariable=self.dataset_path_var, width=50).grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_yaml).grid(row=0, column=2)

        # Epochs selector
        tk.Label(root, text="Epochs:").grid(row=1, column=0, sticky='w')
        self.epochs_var = tk.IntVar(value=50)
        tk.Entry(root, textvariable=self.epochs_var, width=10).grid(row=1, column=1, sticky='w')

        # Patience selector
        tk.Label(root, text="Patience:").grid(row=2, column=0, sticky='w')
        self.patience_var = tk.IntVar(value=50)
        tk.Entry(root, textvariable=self.patience_var, width=10).grid(row=2, column=1, sticky='w')

        # Start training button
        self.train_btn = tk.Button(root, text="Start Training", command=self.start_training)
        self.train_btn.grid(row=3, column=0, columnspan=2, pady=10, sticky='w')

        # Test GPU button
        self.test_gpu_btn = tk.Button(root, text="Test GPU", command=test_gpu)
        self.test_gpu_btn.grid(row=3, column=2, pady=10, sticky='e')

        # Output console
        self.output_console = scrolledtext.ScrolledText(root, width=70, height=20, state='disabled')
        self.output_console.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

        # Redirect print to console
        self.stdout_backup = sys.stdout
        sys.stdout = RedirectText(self.output_console)

    def browse_yaml(self):
        file = filedialog.askopenfilename(
            title="Select dataset.yaml",
            filetypes=[("YAML files", "*.yaml *.yml")]
        )
        if file:
            self.dataset_path_var.set(file)

    def start_training(self):
        dataset = self.dataset_path_var.get()
        epochs = self.epochs_var.get()
        patience = self.patience_var.get()
        if not dataset:
            messagebox.showerror("Error", "Please select your dataset.yaml file.")
            return
        if epochs <= 0 or patience < 0:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and patience.")
            return

        self.train_btn.config(state='disabled')
        print(f"\n🚀 Starting training with dataset: {dataset} for {epochs} epochs, patience {patience}...\n")

        Thread(target=self.train_model, args=(dataset, epochs, patience), daemon=True).start()
        

    def train_model(self, dataset, epochs, patience):
        try:
            device = get_device()
            model = YOLO('yolov8n.pt')
            model.train(
                data=dataset,
                epochs=epochs,
                patience=patience,
                imgsz=640,
                batch=16,
                device=device,
                name="frc_bumper_run",
                save=True,
                save_period=-1,
                project="models",   # Save into 'models' folder instead of 'runs/train'
            )
            print("\n✅ Training complete! 🎉")
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
        finally:
            self.train_btn.config(state='normal')
        
    def on_close(self):
        sys.stdout = self.stdout_backup
        self.root.destroy()

# --- Launch GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTrainerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
