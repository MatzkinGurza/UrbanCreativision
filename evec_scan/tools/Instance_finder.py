import os
import glob

# Define the video file extensions to look for
VIDEO_EXTENSIONS = ('*.mp4') #, '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.webm')

def find_videos_and_encapsulate(root_directory):
    instance_counter = 0  # Counter for instance folders
    
    # Use glob to recursively find all video files
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(root_directory, '**', ext), recursive=True))

    for video_path in video_files:
        # Get the directory of the video
        video_dir = os.path.dirname(video_path)

        # Create a numbered instance folder
        instance_folder = os.path.join(video_dir, f"instance{instance_counter}")
        os.makedirs(instance_folder, exist_ok=True)

        # Move the video into the instance folder
        video_name = os.path.basename(video_path)
        new_video_path = os.path.join(instance_folder, video_name)
        os.rename(video_path, new_video_path)

        print(f"Moved '{video_path}' to '{new_video_path}'")
        instance_counter += 1

if __name__ == "__main__":
    # Specify the root directory to start the search
    root_dir = input("Enter the root directory to search for videos: ").strip()
    if os.path.isdir(root_dir):
        find_videos_and_encapsulate(root_dir)
    else:
        print("The specified directory does not exist.")