# -*- coding: utf-8 -*-
import whisper
import gradio as gr
import os
import glob
import torch
import subprocess
import shutil
import sys
from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip

# Configuration settings
PORT = 7890  # Define port for Gradio server
BATCH_FOLDER = "batch_folder"  # Default folder for batch transcription
VIDEO_FOLDER = "video_for_transcription"  # Folder for videos to transcribe
SUBTITLE_FOLDER = "subtitles"  # Folder to save SRT files

# Check for FFmpeg availability
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

if not check_ffmpeg():
    print("INFO: FFmpeg is not installed or not found in PATH.")
    print("Subtitle burning feature has been removed.")

# Create required folders if they don't exist
for folder in [BATCH_FOLDER, VIDEO_FOLDER, SUBTITLE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model on the specified device
modelname = "small"
print(f"Using model: {modelname}")
model = whisper.load_model(modelname, device=device)

# Helper function to extract audio from video file
def extract_audio_from_video(video_path, output_audio_path="temp_audio.wav"):
    """Extract audio from video file using moviepy"""
    try:
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(output_audio_path, codec='pcm_s16le', logger=None)
        return output_audio_path, None
    except Exception as e:
        return None, f"Error extracting audio: {str(e)}"

def clean_temp_file(file_path):
    """Safely remove a temporary file if it exists"""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Warning: Failed to delete temporary file {file_path}: {str(e)}")

def transcribe_file_with_option(file_path, timestamps=False):
    """Transcribe a file with or without timestamps"""
    result = model.transcribe(file_path)
    language = result["language"]

    if timestamps:
        segments_text = "\n".join(
            [f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}" for seg in result['segments']]
        )
        return language, segments_text, result
    else:
        return language, result["text"], result

# --- Mode 1: Single File ---
def transcribe_single(file, timestamps):
    if not file:
        return "", "No file provided"

    file_path = file.name  # fix for Gradio _TemporaryFileWrapper
    ext = os.path.splitext(file_path)[-1].lower()
    temp_audio_path = None

    try:
        if ext in [".mp4", ".mov", ".avi", ".mkv"]:
            temp_audio_path = "temp_audio.wav"
            extracted_audio, error = extract_audio_from_video(file_path, temp_audio_path)
            if error:
                return "", error
            lang, text, _ = transcribe_file_with_option(extracted_audio, timestamps)
        else:
            lang, text, _ = transcribe_file_with_option(file_path, timestamps)
        
        return lang, text
    except Exception as e:
        return "", f"Error processing file: {str(e)}"
    finally:
        if temp_audio_path:
            clean_temp_file(temp_audio_path)

# --- Mode 2: Folder Path ---
def transcribe_folder(folder_path, timestamps, create_srt_files, create_txt_files):
    # Use default folder if none provided
    if not folder_path:
        folder_path = BATCH_FOLDER
    
    # Clean up the folder path (handle Windows path issues)
    folder_path = folder_path.strip().strip('"\'')  # Remove quotes and whitespace
    
    # Create folder if it doesn't exist
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        return f"Error creating/accessing folder: {str(e)}\nPlease check that the path is correct and accessible."

    results = []
    audio_exts = [".mp3", ".wav", ".m4a", ".aac"]
    video_exts = [".mp4", ".mov", ".avi", ".mkv"]
    all_media_exts = audio_exts + video_exts

    try:
        file_list = glob.glob(os.path.join(folder_path, "*"))
    except Exception as e:
        return f"Error accessing folder: {str(e)}\nPlease check that the path is correct and accessible."
    
    if not file_list:
        return f"No files found in folder: {folder_path}"
        
    for file_path in file_list:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(file_path)[-1].lower()
        temp_audio_path = None
        
        # Skip non-media files
        if ext not in all_media_exts:
            continue
            
        try:
            if ext in audio_exts:
                lang, text, result = transcribe_file_with_option(file_path, timestamps)
            elif ext in video_exts:
                temp_audio_path = f"temp_audio_{base_name}.wav"
                extracted_audio, error = extract_audio_from_video(file_path, temp_audio_path)
                if error:
                    results.append(f"{filename}: Failed - {error}\n")
                    continue
                lang, text, result = transcribe_file_with_option(extracted_audio, timestamps)
            
            # Add result to summary
            results.append(f"{filename}:\n{text}\n")
            
            # Create SRT file if requested (for both audio and video files)
            if create_srt_files:
                srt_path = os.path.join(folder_path, f"{base_name}.srt")
                srt_content = create_srt_from_segments(result["segments"])
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                results.append(f"Created SRT: {base_name}.srt\n")
                
            # Create TXT file if requested
            if create_txt_files:
                txt_path = os.path.join(folder_path, f"{base_name}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                results.append(f"Created TXT: {base_name}.txt\n")
                
        except Exception as e:
            results.append(f"{filename}: Failed - {str(e)}\n")
        finally:
            if temp_audio_path:
                clean_temp_file(temp_audio_path)
    
    if not results:
        return f"No supported audio or video files found in folder: {folder_path}"
    
    return "\n".join(results)

    return "\n".join(results)

# --- Mode 3: URL-based ---
def transcribe_url(url, timestamps):
    downloaded_file = None
    audio_file = None
    
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "downloaded.%(ext)s",
            "quiet": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "no_warnings": True,
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info_dict)
            audio_file = downloaded_file.replace(info_dict.get('ext', ''), 'wav')

        if not os.path.exists(audio_file):
            return "", "Failed to download or convert audio"

        language, text, _ = transcribe_file_with_option(audio_file, timestamps)
        return language, text
    except Exception as e:
        return "", f"Error: {str(e)}"
    finally:
        # Clean up downloaded files
        if downloaded_file and os.path.exists(downloaded_file):
            clean_temp_file(downloaded_file)
        if audio_file and os.path.exists(audio_file):
            clean_temp_file(audio_file)

# --- Subtitle Tools ---
def format_time(seconds):
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def create_srt_from_segments(segments):
    """Convert Whisper segments to SRT format"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"].strip()

        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

    return srt_content

def list_videos_in_folder():
    """List video files in the video_for_transcription folder"""
    video_exts = [".mp4", ".mov", ".avi", ".mkv"]
    videos = []

    for ext in video_exts:
        videos.extend(glob.glob(os.path.join(VIDEO_FOLDER, f"*{ext}")))

    # Return just the base names
    return [os.path.basename(v) for v in videos]

def create_srt(video_name, output_name):
    """Transcribe video and create SRT file"""
    if not video_name or not output_name:
        return "Please provide both video name and output name"

    video_path = os.path.join(VIDEO_FOLDER, video_name)
    srt_path = os.path.join(SUBTITLE_FOLDER, f"{output_name}.srt")
    temp_audio_path = "temp_audio_for_srt.wav"

    # Check if video file exists
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"

    try:
        # Extract audio from video
        extracted_audio, error = extract_audio_from_video(video_path, temp_audio_path)
        if error:
            return error

        # Transcribe with Whisper
        result = model.transcribe(extracted_audio)

        # Create SRT content
        srt_content = create_srt_from_segments(result["segments"])

        # Save SRT file
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # Try to open SRT file with default text editor
        try:
            if sys.platform == "win32":
                subprocess.Popen(["notepad", srt_path])
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", "-t", srt_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", srt_path])
        except Exception as e:
            return f"SRT file created successfully at {srt_path}, but could not open automatically: {str(e)}"

        return f"SRT file created successfully at {srt_path}"
    except Exception as e:
        return f"Error creating SRT file: {str(e)}"
    finally:
        clean_temp_file(temp_audio_path)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# OpenAI Whisper Transcriber\nSelect a mode to transcribe audio or video.")

    with gr.Tabs():
        with gr.Tab("Instructions"):
            gr.Markdown("""
            # How to Use Whisper Transcriber
            
            This application uses OpenAI's Whisper model to transcribe speech from audio and video files. Here's how to use each tab:
            
            ## Single File Tab
            
            **Purpose:** Transcribe one audio or video file at a time.
            
            **How to use:**
            1. Click "Upload audio or video file" to select a file from your computer
            2. Check "Include timestamps" if you want timestamps in the transcription
            3. Click "Transcribe" and wait for results
            4. Results will show detected language and transcription text
            
            **Supported formats:** MP3, WAV, M4A, AAC, MP4, MOV, AVI, MKV
            
            ---
            
            ## Folder Transcription Tab
            
            **Purpose:** Batch transcribe multiple files in a folder.
            
            **How to use:**
            1. Enter the folder path containing your audio/video files (or use default "batch_folder")
            2. Select options:
               - Include timestamps: Add time markers to transcriptions
               - Create SRT files: Generate subtitle files for all audio/video files
               - Save as TXT files: Save each transcription as a separate text file
            3. Click "Transcribe Folder" and wait for results
            4. All generated files will be saved in the same folder as the source files
            
            **Folder location:** By default, the application looks in the "batch_folder" directory. You can create this folder in the same location where you run the application.
            
            ---
            
            ## From URL Tab
            
            **Purpose:** Transcribe audio or video from an online source (like YouTube).
            
            **How to use:**
            1. Paste a URL to an audio or video file (YouTube links work well)
            2. Check "Include timestamps" if desired
            3. Click "Transcribe URL" and wait for results
            4. Results will show detected language and transcription text
            
            **Note:** This requires an internet connection to download the content.
            
            ---
            
            ## Subtitle Tools Tab
            
            **Purpose:** Create SRT subtitle files from videos.
            
            **How to use:**
            1. Place your video files in the "video_for_transcription" folder
            2. Select a video from the dropdown (click "Refresh" if your video doesn't appear)
            3. Enter a name for the SRT file (without the .srt extension)
            4. Click "Create SRT File"
            5. The SRT file will be created in the "subtitles" folder and will open automatically
            
            **Folder locations:**
            - Input videos go in: "video_for_transcription" folder
            - Output SRT files go in: "subtitles" folder
            
            ---
            
            ## File Locations
            
            The application creates several folders to organize files:
            
            - **batch_folder:** Place files here for batch processing
            - **video_for_transcription:** Place videos here for the Subtitle Tools tab
            - **subtitles:** SRT files created in the Subtitle Tools tab are saved here
            
            All these folders are created automatically in the same directory where you run the application.
            """)

        with gr.Tab("Single File"):
            file_input = gr.File(label="Upload audio or video file")
            use_timestamps = gr.Checkbox(label="Include timestamps", value=True)
            lang_out = gr.Label(label="Detected Language")
            text_out = gr.Textbox(label="Transcription", lines=15)
            btn = gr.Button("Transcribe")
            btn.click(fn=transcribe_single, inputs=[file_input, use_timestamps], outputs=[lang_out, text_out])

        with gr.Tab("Folder Transcription"):
            folder_input = gr.Textbox(label="Enter folder path", placeholder=f"Default: {BATCH_FOLDER}", value=BATCH_FOLDER)
            with gr.Row():
                folder_timestamps = gr.Checkbox(label="Include timestamps", value=True)
                create_srt_files_checkbox = gr.Checkbox(label="Create SRT files", value=False, info="Creates SRT files for all audio and video files")
                create_txt_files_checkbox = gr.Checkbox(label="Save as TXT files", value=False, info="Saves transcriptions as separate TXT files")
            folder_output = gr.Textbox(label="Transcriptions", lines=20)
            folder_btn = gr.Button("Transcribe Folder")
            folder_btn.click(
                fn=transcribe_folder, 
                inputs=[folder_input, folder_timestamps, create_srt_files_checkbox, create_txt_files_checkbox], 
                outputs=[folder_output]
            )

        with gr.Tab("From URL"):
            url_input = gr.Textbox(label="Enter audio or video URL", placeholder="YouTube or direct link")
            url_timestamps = gr.Checkbox(label="Include timestamps", value=True)
            url_lang = gr.Label(label="Detected Language")
            url_out = gr.Textbox(label="Transcription", lines=15)
            url_btn = gr.Button("Transcribe URL")
            url_btn.click(fn=transcribe_url, inputs=[url_input, url_timestamps], outputs=[url_lang, url_out])

        with gr.Tab("Subtitle Tools"):
            gr.Markdown(f"### Create SRT File\nVideos should be in '{VIDEO_FOLDER}' folder. SRT files will be saved to '{SUBTITLE_FOLDER}' folder.")

            with gr.Column():
                with gr.Row():
                    video_dropdown = gr.Dropdown(
                        label="Select Video",
                        choices=list_videos_in_folder(),
                        interactive=True,
                        scale=3
                    )
                    refresh_btn = gr.Button(
                        "Refresh",
                        scale=1,
                        min_width=80
                    )

                srt_name = gr.Textbox(
                    label="Output SRT Name (without extension)",
                    placeholder="Enter name for SRT file"
                )
                create_srt_btn = gr.Button("Create SRT File")
                create_result = gr.Textbox(label="Result", lines=2)

            # Define update function for the video dropdown
            def update_videos():
                return gr.Dropdown.update(choices=list_videos_in_folder())

            # Connect buttons to functions
            create_srt_btn.click(fn=create_srt, inputs=[video_dropdown, srt_name], outputs=[create_result])

            # Connect refresh button to update the video dropdown
            refresh_btn.click(fn=update_videos, outputs=[video_dropdown])

# Launch the app with the specified port and allow network access
print(f"Launching Whisper Transcriber on port {PORT}")
print(f"Once launched, access via: http://localhost:{PORT}")
demo.launch(server_name="0.0.0.0", server_port=PORT, show_error=True)