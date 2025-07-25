import os
import json
import cv2
import torch
import csv
import shutil
import random
import time
from ultralytics import YOLO
from pathlib import Path
from yt_dlp import YoutubeDL
from ddgs import DDGS
import threading
from tqdm import tqdm
import subprocess
import sys
import logging
import psutil

# --- CONFIG ---
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 10
SAVE_NEGATIVE_EVERY_N = 30
MAX_NEGATIVE_PER_VIDEO = 5  # Limit negative frames per video to save space
MAX_FRAMES = 99_999
RESTART_AFTER = 10  # Restart after this many videos
MIN_FREE_SPACE_GB = 2  # Minimum free disk space in GB to continue saving frames

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_INDEX_FILE = REPO_ROOT / "config" / "frame_counter.txt"
stop_requested = False

URL_LOG = REPO_ROOT / "config" / "seen_urls.json"
LOGS_DIR = REPO_ROOT / "logs" / "eval_reports"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FRAME_LOG_CSV = LOGS_DIR / "frame_log.csv"
QUERIES_FILE = REPO_ROOT / "config" / "search_terms.txt"

# Setup error logging
ERROR_LOG_FILE = REPO_ROOT / "logs" / "errors.log"
logging.basicConfig(filename=ERROR_LOG_FILE, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Frame Counter ---
def load_frame_counter():
    if FRAME_INDEX_FILE.exists():
        with open(FRAME_INDEX_FILE, "r") as f:
            return int(f.read().strip())
    else:
        existing_frames = sorted(OUTPUT_DIR.glob("frame_*.png"))
        if existing_frames:
            return max(int(f.stem.split("_")[1]) for f in existing_frames) + 1
        return 0

def save_frame_counter(counter):
    with open(FRAME_INDEX_FILE, "w") as f:
        f.write(str(counter))

frame_counter = load_frame_counter()
print(f"📸 Starting from frame {frame_counter} (cached).")

# Load seen URLs
if URL_LOG.exists():
    with open(URL_LOG, 'r') as f:
        seen_urls = set(json.load(f))
else:
    seen_urls = set()

model = YOLO(REPO_ROOT / "models" / "frc_bumper_run" / "weights" / "best.pt")

# --- Load Queries ---
def get_random_query():
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        return random.choice(queries)
    return "FRC bumper match"

# --- YouTube Scraper ---
def search_youtube_videos(query, max_results=10):
    print(f"\n🔍 Searching YouTube videos for: '{query}'")
    found = []
    with DDGS() as ddgs:
        for r in ddgs.text(query + " site:youtube.com", max_results=max_results):
            url = r['href']
            print(f"🔗 Found URL: {url}")
            if "youtube.com/watch" in url and url not in seen_urls:
                found.append(url)
    print(f"🔷 Total new URLs found: {len(found)}")
    return found

# --- Keyboard Stop Shortcut ---
def check_for_stop():
    global stop_requested
    try:
        while True:
            key = input()
            if key.strip().lower() == 'q':
                print("\n⏸️ Stop requested. Finishing current video then exiting...")
                stop_requested = True
                break
    except EOFError:
        pass

# --- Disk space check ---
def has_enough_space(min_gb=MIN_FREE_SPACE_GB):
    usage = shutil.disk_usage(str(OUTPUT_DIR))
    free_gb = usage.free / (1024**3)
    return free_gb >= min_gb

# --- Restart in new terminal and kill old ---
def restart_in_new_terminal():
    python = sys.executable
    script = sys.argv[0]
    args = sys.argv[1:]

    ps_command = f'python "{script}" {" ".join(args)}'

    subprocess.Popen([
        "powershell",
        "-NoExit",
        "-Command",
        ps_command
    ])

    time.sleep(2)  # Give new terminal time to open

    parent_pid = os.getppid()
    try:
        parent = psutil.Process(parent_pid)
        parent.terminate()
        print("Old terminal window closed.")
    except Exception as e:
        print(f"Failed to close old terminal: {e}")

    sys.exit()

# --- Download Video ---
def download_video_clip(url, video_id, temp_dir):
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_template = temp_dir / f"{video_id}.%(ext)s"

    ydl_opts = {
        'format': 'bv*[ext=mp4][height<=720]',
        'outtmpl': str(out_template),
        'quiet': True,
        'noplaylist': True,
        'no_warnings': True,
    }

    for attempt in range(3):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(result)
                file_path = Path(filename)
                if not file_path.exists():
                    file_path = file_path.with_suffix('.mp4')
                return file_path if file_path.exists() else None
        except Exception as e:
            logging.error(f"Download failed (attempt {attempt+1}/3) for {url}: {e}")
            print(f"❌ Download failed (attempt {attempt+1}/3) for {url}: {e}")
            time.sleep(5)
    return None

# --- Frame Extractor & Filter ---
def extract_and_filter_frames(video_path):
    global frame_counter
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        logging.error(f"Failed to open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎮 Total frames in video: {total_frames}")

    frame_idx = 0
    saved_negatives = 0
    saved_this_video = 0

    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            # process frame
            frame_idx += FRAME_SKIP


            try:
                results = model.predict(frame, verbose=False)
                detections = results[0].boxes
            except Exception as e:
                print(f"⚠️ AI prediction failed on frame {frame_idx}: {e}")
                logging.error(f"AI prediction failed on frame {frame_idx}: {e}")
                continue

            has_bumper = False
            if detections is not None and len(detections) > 0:
                has_bumper = any(det.conf >= CONFIDENCE_THRESHOLD for det in detections)

            save = False
            if has_bumper:
                save = True
            elif saved_negatives % SAVE_NEGATIVE_EVERY_N == 0 and saved_negatives < MAX_NEGATIVE_PER_VIDEO:
                save = True
                saved_negatives += 1

            if save:
                if frame_counter >= MAX_FRAMES:
                    print(f"⚠️ Frame counter limit reached ({MAX_FRAMES}). Stopping save.")
                    break

                if not has_enough_space():
                    print(f"⚠️ Low disk space. Stopping frame saving.")
                    break

                filename = f"frame_{frame_counter:05}.png"
                filepath = OUTPUT_DIR / filename
                cv2.imwrite(str(filepath), frame)
                frame_counter += 1
                save_frame_counter(frame_counter)
                saved_this_video += 1

    cap.release()
    try:
        video_path.unlink()
    except Exception as e:
        print(f"⚠️ Could not delete video {video_path}: {e}")
        logging.error(f"Could not delete video {video_path}: {e}")

    print(f"✔️ Done processing video. Saved {saved_this_video} new frames.")
    return saved_this_video

# --- Main Loop ---
def main():
    threading.Thread(target=check_for_stop, daemon=True).start()

    temp_dir = REPO_ROOT / "temp"
    runs_done = 0

    while True:
        query = get_random_query()
        new_urls = search_youtube_videos(query=query, max_results=30)

        frame_log_rows = []
        videos_processed = 0
        for url in new_urls:
            if stop_requested:
                print("Stopping as requested. Exiting main loop.")
                break

            print(f"\n🎥 Processing: {url}")
            seen_urls.add(url)

            video_id = url.split("v=")[-1].split("&")[0]
            downloaded_path = download_video_clip(url, video_id, temp_dir)

            if not downloaded_path or not downloaded_path.exists():
                print(f"⚠️ Skipping video due to download failure.")
                continue

            saved_count = extract_and_filter_frames(downloaded_path)
            frame_log_rows.append([video_id, url, saved_count])
            videos_processed += 1
            runs_done += 1

            # Random delay 2-5 seconds between videos to reduce throttling risk
            time.sleep(random.uniform(2, 5))

            if runs_done >= RESTART_AFTER:
                print("🔄 Restart limit reached. Restarting script in new terminal...")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                restart_in_new_terminal()

        with open(URL_LOG, 'w') as f:
            json.dump(sorted(list(seen_urls)), f, indent=2)

        with open(FRAME_LOG_CSV, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(['video_id', 'url', 'frames_saved'])
            writer.writerows(frame_log_rows)

        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print("🧹 Cleaned up temp folder.")
            except Exception as e:
                print(f"⚠️ Failed to delete temp folder: {e}")
                logging.error(f"Failed to delete temp folder: {e}")

        if stop_requested:
            print("\n✅ Stopped by user request.")
            break

    print("\n✅ Done scraping and extracting!")

if __name__ == "__main__":
    main()
