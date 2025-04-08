import whisper
import gradio as gr
import os
import glob
import torch
from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model on the specified device(tiny,base,small,medium,large)
modelname = "small"
model = whisper.load_model(modelname, device=device)



def transcribe_file_with_option(file_path, timestamps=False):
    result = model.transcribe(file_path)
    language = result["language"]

    if timestamps:
        segments_text = "\n".join(
            [f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}" for seg in result['segments']]
        )
        return language, segments_text
    else:
        return language, result["text"]

# --- Mode 1: Single File ---
def transcribe_single(file, timestamps):
    if not file:
        return "", "No file provided"

    file_path = file.name  # fix for Gradio _TemporaryFileWrapper
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        with VideoFileClip(file_path) as video:
            audio_path = "temp_audio.wav"
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            lang, text = transcribe_file_with_option(audio_path, timestamps)
            os.remove(audio_path)
            return lang, text
    else:
        return transcribe_file_with_option(file_path, timestamps)

# --- Mode 2: Folder Path ---
def transcribe_folder(folder_path, timestamps):
    results = []
    audio_exts = [".mp3", ".wav", ".m4a", ".aac"]
    video_exts = [".mp4", ".mov", ".avi", ".mkv"]

    for file_path in glob.glob(os.path.join(folder_path, "*")):
        ext = os.path.splitext(file_path)[-1].lower()
        try:
            if ext in audio_exts:
                _, text = transcribe_file_with_option(file_path, timestamps)
            elif ext in video_exts:
                with VideoFileClip(file_path) as video:
                    audio_path = "temp_audio.wav"
                    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
                    _, text = transcribe_file_with_option(audio_path, timestamps)
                    os.remove(audio_path)
            else:
                continue
            results.append(f"{os.path.basename(file_path)}:\n{text}\n")
        except Exception as e:
            results.append(f"{os.path.basename(file_path)}: Failed - {str(e)}\n")

    return "\n".join(results)

# --- Mode 3: URL-based ---
def transcribe_url(url, timestamps):
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
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        downloaded = [f for f in os.listdir() if f.startswith("downloaded.") and f.endswith(".wav")]
        if not downloaded:
            return "", "Failed to download audio"

        language, text = transcribe_file_with_option(downloaded[0], timestamps)
        os.remove(downloaded[0])
        return language, text
    except Exception as e:
        return "", f"Error: {str(e)}"

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# OpenAI Whisper Transcriber\nSelect a mode to transcribe audio or video.")

    with gr.Tabs():
        with gr.Tab("Single File"):
            file_input = gr.File(label="Upload audio or video file")
            use_timestamps = gr.Checkbox(label="Include timestamps", value=True)
            lang_out = gr.Label(label="Detected Language")
            text_out = gr.Textbox(label="Transcription", lines=15)
            btn = gr.Button("Transcribe")
            btn.click(fn=transcribe_single, inputs=[file_input, use_timestamps], outputs=[lang_out, text_out])
        
        with gr.Tab("Folder Transcription"):
            folder_input = gr.Textbox(label="Enter folder path", placeholder="e.g., /path/to/files/")
            folder_timestamps = gr.Checkbox(label="Include timestamps", value=True)
            folder_output = gr.Textbox(label="Transcriptions", lines=20)
            folder_btn = gr.Button("Transcribe Folder")
            folder_btn.click(fn=transcribe_folder, inputs=[folder_input, folder_timestamps], outputs=[folder_output])
        
        with gr.Tab("From URL"):
            url_input = gr.Textbox(label="Enter audio or video URL", placeholder="YouTube or direct link")
            url_timestamps = gr.Checkbox(label="Include timestamps", value=True)
            url_lang = gr.Label(label="Detected Language")
            url_out = gr.Textbox(label="Transcription", lines=15)
            url_btn = gr.Button("Transcribe URL")
            url_btn.click(fn=transcribe_url, inputs=[url_input, url_timestamps], outputs=[url_lang, url_out])

demo.launch()
