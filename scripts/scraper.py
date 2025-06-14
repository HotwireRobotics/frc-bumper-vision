import os
import yt_dlp
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import re
import urllib.parse
import shutil
from PIL import Image
import numpy as np
import av
from concurrent.futures import ThreadPoolExecutor

# Configuration
SAVE_DIR = "data"  # Points to the existing data folder at project root

# Check for CUDA support
def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            return True, device
        else:
            print("No CUDA devices found. Using CPU.")
            return False, torch.device("cpu")
    except ImportError:
        print("PyTorch not available. Using CPU.")
        return False, None

class YoutubeScraperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame Extractor")
        self.geometry("450x250")  # Made window taller for new controls
        self.resizable(False, False)
        self.video_file = None  # Store video file path for cleanup
        
        # Check for CUDA support
        self.use_gpu, self.device = check_cuda()
        
        # Create save directory if it doesn't exist
        self.data_dir = os.path.abspath(SAVE_DIR)
        print(f"Save directory: {self.data_dir}")  # Debug print
        
        # Ensure we have write permissions
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(self.data_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("Write permissions verified")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot write to {self.data_dir}: {str(e)}")
            raise

        # Input type selection
        self.input_type = tk.StringVar(value="youtube")
        input_frame = tk.Frame(self)
        input_frame.pack(fill="x", padx=10, pady=(10, 0))
        tk.Radiobutton(input_frame, text="YouTube URL", variable=self.input_type, 
                      value="youtube", command=self.toggle_input_type).pack(side="left", padx=5)
        tk.Radiobutton(input_frame, text="Local Video", variable=self.input_type, 
                      value="local", command=self.toggle_input_type).pack(side="left", padx=5)

        # URL input
        self.url_frame = tk.Frame(self)
        self.url_frame.pack(fill="x", padx=10, pady=(5, 0))
        tk.Label(self.url_frame, text="YouTube Video URL:").pack(anchor="w")
        self.url_entry = tk.Entry(self.url_frame, width=60)
        self.url_entry.pack(pady=(0, 5))

        # Local file input
        self.local_frame = tk.Frame(self)
        self.local_frame.pack(fill="x", padx=10, pady=(5, 0))
        tk.Label(self.local_frame, text="Video File:").pack(anchor="w")
        file_frame = tk.Frame(self.local_frame)
        file_frame.pack(fill="x")
        self.file_entry = tk.Entry(file_frame, width=45)
        self.file_entry.pack(side="left", fill="x", expand=True)
        tk.Button(file_frame, text="Browse", command=self.browse_video).pack(side="left", padx=5)

        # Frame interval
        tk.Label(self, text="Frame Interval (seconds):").pack(anchor="w", padx=10, pady=(5, 0))
        self.interval_entry = tk.Entry(self, width=10)
        self.interval_entry.insert(0, "2.0")
        self.interval_entry.pack(padx=10, pady=(0, 5))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 5))

        # Status box
        self.status_text = tk.StringVar()
        self.status_text.set("Ready")
        tk.Label(self, textvariable=self.status_text, fg="blue").pack(pady=(0,10))

        # Download button
        tk.Button(self, text="Extract Frames", command=self.run_scraper).pack(pady=5)

        # Initialize UI state
        self.toggle_input_type()

    def toggle_input_type(self):
        if self.input_type.get() == "youtube":
            self.url_frame.pack(fill="x", padx=10, pady=(5, 0))
            self.local_frame.pack_forget()
        else:
            self.url_frame.pack_forget()
            self.local_frame.pack(fill="x", padx=10, pady=(5, 0))

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)

    def validate_youtube_url(self, url: str) -> bool:
        youtube_regex = r'(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+'
        return bool(re.match(youtube_regex, url))

    def download_youtube_video(self, url: str) -> str:
        self.status_text.set("Downloading video...")
        self.progress_var.set(0)
        self.update()
        
        # Create videos directory inside data folder
        videos_dir = os.path.join(self.data_dir, "videos")
        print(f"Videos directory: {videos_dir}")  # Debug print
        
        # Create videos directory if it doesn't exist
        os.makedirs(videos_dir, exist_ok=True)
        
        ydl_opts = {
            'format': 'bestvideo[height>=1080][ext=mp4]/bestvideo[ext=mp4]/best[ext=mp4]/best',  # Get highest quality video
            'outtmpl': os.path.join(videos_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [self.download_progress_hook],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                video_title = info_dict.get('title', 'video')
                # Print video quality information
                print(f"\nVideo Quality Information:")
                print(f"Resolution: {info_dict.get('resolution', 'unknown')}")
                print(f"Format: {info_dict.get('format', 'unknown')}")
                print(f"Format ID: {info_dict.get('format_id', 'unknown')}")
                print(f"Filesize: {info_dict.get('filesize', 'unknown')} bytes")
                # Sanitize filename
                video_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).strip()
                ext = info_dict.get('ext', 'mp4')
                filename = f"{video_title}.{ext}"
                video_path = os.path.join(videos_dir, filename)
                print(f"Video saved to: {video_path}")  # Debug print
                
                # Verify video was downloaded
                if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                    raise Exception("Video download failed or file is empty")
                    
                return video_path
        except Exception as e:
            raise Exception(f"Failed to download video: {str(e)}")

    def download_progress_hook(self, d):
        if d['status'] == 'downloading':
            try:
                progress = float(d['_percent_str'].replace('%', ''))
                self.progress_var.set(progress)
                self.update()
            except:
                pass

    def extract_frames(self, video_path: str, interval_seconds: float = 2.0):
        self.status_text.set("Extracting frames...")
        self.progress_var.set(0)
        self.update()
        
        # Read video using PyAV
        container = av.open(video_path)
        fps = container.streams.video[0].average_rate

        # First pass: count total frames for accurate progress bar
        total_frames = 0
        print("Starting first pass to count frames...") # Debug print
        for frame in container.decode(video=0):
            total_frames += 1
            if total_frames % 100 == 0:
                print(f"First pass: counted {total_frames} frames...") # Debug print
        container.close() # Close container to reset for second pass

        print(f"FPS: {fps}, Total Frames counted: {total_frames}") # Debug print
        
        if total_frames == 0:
            raise Exception("No frames found in video stream. Video might be empty or corrupted.")

        frame_interval = int(fps * interval_seconds)

        # Re-open container for actual frame extraction
        container = av.open(video_path)

        # Create frames directory inside data folder
        frames_dir = os.path.join(self.data_dir, "frames")
        print(f"Frames directory: {frames_dir}")  # Debug print
        
        # Create frames directory if it doesn't exist
        os.makedirs(frames_dir, exist_ok=True)
        
        # Find the last frame number
        existing_frames = [f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')]
        if existing_frames:
            last_frame_num = max([int(f.split('_')[1].split('.')[0]) for f in existing_frames])
            frame_num_start = last_frame_num + 1
            print(f"Continuing from frame number: {frame_num_start}")
        else:
            frame_num_start = 0
            print("Starting from frame number: 0")

        # Configure video stream for optimal performance
        video_stream = container.streams.video[0]
        video_stream.thread_type = av.codec.context.ThreadType.AUTO
        video_stream.thread_count = os.cpu_count() or 4

        frames_to_process = []
        frame_count = 0
        
        # Extract frames
        print("Starting second pass to extract frames...") # Debug print
        for frame in container.decode(video=0):
            if frame_count % frame_interval == 0:
                frames_to_process.append((frame, frame_num_start + (frame_count // frame_interval)))
            frame_count += 1
            
            # Update progress
            if frame_count % 100 == 0:
                print(f"Second pass: processed {frame_count} frames...") # Debug print
            progress = (frame_count / total_frames) * 100
            self.progress_var.set(progress)
            self.update()

        total_frames_to_save = len(frames_to_process)
        saved_frames_count = 0
        
        # Use ThreadPoolExecutor for parallel saving
        num_workers = os.cpu_count() * 2 if os.cpu_count() else 4
        print(f"Using {num_workers} worker threads for frame saving.")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for frame, current_frame_num in frames_to_process:
                futures.append(executor.submit(self._save_frame, frame, current_frame_num, frames_dir))

            for future in futures:
                try:
                    future.result()
                    saved_frames_count += 1
                except Exception as e:
                    print(f"Error in thread saving frame: {str(e)}")
                    raise

        # Clean up
        container.close()
        
        self.progress_var.set(100)
        print(f"Total frames extracted: {saved_frames_count}")
        
        if saved_frames_count == 0:
            raise Exception("No frames were extracted")
            
        self.status_text.set(f"Done! Extracted {saved_frames_count} frames to '{frames_dir}'.")

    def _save_frame(self, frame, frame_num, frames_dir):
        """Helper function to save a single frame."""
        try:
            # Convert to PIL Image
            frame_np = frame.to_ndarray(format='rgb24')
            pil_image = Image.fromarray(frame_np)
            
            # Save frame
            filename = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
            pil_image.save(filename, format='PNG', optimize=False)
            
            # Verify frame was saved
            if not os.path.exists(filename):
                raise Exception(f"Frame file {filename} was not created")
                
            if os.path.getsize(filename) == 0:
                raise Exception(f"Frame file {filename} is empty")
                
        except Exception as e:
            raise

    def cleanup(self):
        if self.video_file and os.path.exists(self.video_file):
            try:
                os.remove(self.video_file)
                print(f"Cleaned up video file: {self.video_file}")  # Debug print
            except:
                pass

    def run_scraper(self):
        interval_str = self.interval_entry.get().strip()

        try:
            interval = float(interval_str)
            if interval <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive number for frame interval.")
            return

        try:
            if self.input_type.get() == "youtube":
                url = self.url_entry.get().strip()
                if not url:
                    messagebox.showerror("Error", "Please enter a YouTube video URL.")
                    return
                if not self.validate_youtube_url(url):
                    messagebox.showerror("Error", "Please enter a valid YouTube URL.")
                    return
                self.video_file = self.download_youtube_video(url)
            else:
                video_path = self.file_entry.get().strip()
                if not video_path:
                    messagebox.showerror("Error", "Please select a video file.")
                    return
                if not os.path.exists(video_path):
                    messagebox.showerror("Error", "Selected video file does not exist.")
                    return
                self.video_file = video_path

            self.extract_frames(self.video_file, interval)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")
            self.status_text.set("Failed.")
        finally:
            if self.input_type.get() == "youtube":
                self.cleanup()

if __name__ == "__main__":
    app = YoutubeScraperApp()
    app.mainloop()
