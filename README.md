
# Whisper OpenAi Tool Gradio Web implementation
Whisper is an automatic speech recognition (ASR) system Gradio Web UI Implementation



## Installation



Install ffmeg on Your Device

```bash
  # on Ubuntu or Debian
  sudo apt update
  sudo apt install ffmpeg

  # on MacOS using Homebrew (https://brew.sh/)
  brew install ffmpeg

  # on Windows using Chocolatey (https://chocolatey.org/)
  choco install ffmpeg

  # on Windows using Scoop (https://scoop.sh/)
  scoop install ffmpeg

  # on conda
  open conda terminal 
  conda create --name Inwhisper python=3.10
  conda install ffmpeg
```

Install Pytorch for faster inference

```
  # Visit pytorch and choose the proper version for yourself
  in a command window type

  nvcc --version

  to find your cuda version
  
  https://pytorch.org/get-started/locally/
```
Download Program

```bash
 
  git clone https://github.com/RoyTateAI/INwhisper.git
  pip install -r requirements.txt

  pip install  pandas==1.4.2 numpy==1.25.0 moviepy==1.0.3 yt_dlp
```
    

Run Program

```bash
  python app.py

```

## Available models and languages ([Credit](https://github.com/innovatorved/whisper-openai-gradio-implementation/blob/main/README.md))

do not use any english only models with this implementation
on line 40 of the python file, you can specify the model size, larger models may be better, especially for unclear speech but take longer to run. The current default is "small"


 


|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |



## License

[MIT](https://choosealicense.com/licenses/mit/)


## Reference

- [https://github.com/openai/whisper](https://github.com/openai/whisper)
- [https://openai.com/blog/whisper/](https://openai.com/blog/whisper/)

  
## Original Author

- [Ved Gupta](https://www.github.com/innovatorved)



