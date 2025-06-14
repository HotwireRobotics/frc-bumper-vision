import os
import shutil
import subprocess

def cleanup_opencv():
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Directories to remove
    dirs_to_remove = [
        os.path.join(project_root, 'opencv'),
        os.path.join(project_root, 'opencv_contrib'),
        os.path.join(project_root, 'build'),
    ]
    
    # Files to remove
    files_to_remove = [
        os.path.join(project_root, 'CMakeCache.txt'),
        os.path.join(project_root, 'CMakeFiles'),
    ]
    
    print("Starting cleanup...")
    
    # Remove directories using rmdir /s /q command
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            print(f"Removing directory: {dir_path}")
            try:
                # Use Windows command to force remove directory
                subprocess.run(['cmd', '/c', f'rmdir /s /q "{dir_path}"'], 
                             shell=True, 
                             check=True)
                print(f"Successfully removed {dir_path}")
            except Exception as e:
                print(f"Error removing {dir_path}: {str(e)}")
    
    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            print(f"Removing file/directory: {file_path}")
            try:
                if os.path.isdir(file_path):
                    subprocess.run(['cmd', '/c', f'rmdir /s /q "{file_path}"'], 
                                 shell=True, 
                                 check=True)
                else:
                    os.remove(file_path)
                print(f"Successfully removed {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {str(e)}")
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    cleanup_opencv() 