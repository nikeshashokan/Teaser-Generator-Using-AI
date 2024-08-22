### **Video Teaser Generator**

![png-transparent-computer-icons-movieclips-video-clip-youtube-film-youtube-text-rectangle-logo](https://github.com/user-attachments/assets/45370adc-6da1-40c4-a5e6-28743c37a410)


**Overview**

The Video Teaser Generator is a Python-based tool that automates the process of creating a teaser from a video file. It extracts audio, transcribes the audio to text, summarizes the content, identifies key segments, and compiles these segments into a professional teaser video. The tool is available both as a standalone Python script and as a Flask-based web application. This tool is particularly useful for content creators, filmmakers, and marketers who need to quickly generate engaging preview content.
Features
•	Audio Extraction: Extracts audio from video files.
•	Audio Transcription: Transcribes the extracted audio using the Whisper model.
•	Text Summarization: Summarizes long transcripts using the GPT-2 model.
•	Key Segment Identification: Identifies the most important segments of the video based on the transcript.
•	Teaser Creation: Compiles the identified segments into a teaser video with customizable titles and credits.
•	Web Interface: Allows users to upload a video, input details, and download the generated teaser through a simple web interface.

**Installation
Requirements**

•	Python 3.8 or later
•	Flask for the web application framework
•	ffmpeg (required for audio processing)
•	Whisper for audio transcription
•	transformers for text summarization
•	MoviePy for video processing
Installing Dependencies
pip install moviepy transformers whisper sklearn torch numpy Flask
Installing ffmpeg
sudo apt-get install ffmpeg
On macOS:
brew install ffmpeg
Usage
Running the Script
1.	Edit main() Function:
o	Specify the path to your video file.
o	Provide the video title and director's name.
2.	Run the Script:
                     python video_teaser_generator.py
3.	Output:
o	A teaser video (teaser0.mp4) will be generated in the same directory.
Running the Web Application
1.	Project Setup:
o	Clone the repository.
o	Navigate to the project directory.
o	Install the required dependencies:

                     pip install -r requirements.txt
  	
3.	Directory Setup:
o	Ensure the uploads directory exists for storing uploaded videos:

                      mkdir uploads
5.	Start the Flask Web Server:
   
                      python app.py
7.	Access the Application:
o	Open your web browser and navigate to http://127.0.0.1:5000/.
o	Upload your video file, enter the title and director's name, and submit the form.
o	After processing, download the generated teaser from the result page.
Code Explanation
Modules and Dependencies

'''
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, TextClip, CompositeVideoClip
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import whisper
import torch
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.config import change_settings

'''
•	Flask: Provides the web framework for handling requests and rendering templates.
•	os: Handles file operations such as saving uploaded videos.
•	moviepy: Handles video processing tasks such as video editing and clip generation.
•	transformers: Provides the GPT-2 model for text summarization.
•	whisper: Handles audio transcription.
•	torch: Provides tensor operations used in model inference.
•	numpy: Supports numerical operations.
•	sklearn: Supports text vectorization and similarity measurements.
•	change_settings: Configures the path for ImageMagick, used for text rendering in video clips.
Helper Functions
•	Audio Extraction:
def extract_audio(video_path, output_path='audio0.wav'):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path)
    return output_path
	Extracts the audio from the specified video and saves it as a .wav file.
•	Audio Transcription:
def transcribe_audio(audio_path, model_name='base'):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result['text']
	Transcribes the extracted audio using the Whisper model.


•	Text Summarization:
def summarize_text(text, model_name='gpt2', max_tokens=512):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=max_tokens, truncation=True)
    attention_mask = torch.ones_like(inputs)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=150,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
	Summarizes the text using the GPT-2 model.
•	Identifying Key Segments:
def identify_key_segments(transcript, video_duration, num_segments=6, segment_length=10, process_filename='key_segments_process.txt'):
Identifies the most important segments of the video based on the transcript.
•	Creating the Teaser:
def create_teaser(video_path, segments, title, director_name, output_path='teaser0.mp4'):
	Compiles the identified key segments into a teaser video with title and director information.



•	Main Script:
def process_video(video_path, title, director_name):
	The main function that orchestrates the entire process from audio extraction to teaser creation.
Flask Application
•	Upload Directory:
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
	Ensures the uploads directory exists for storing video files.
•	Main Route:
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video_file']
        title = request.form['title']
        director_name = request.form['director_name']
        
        if video_file:
            video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(video_path)
            
            process_video(video_path, title, director_name)
            
            return redirect(url_for('result'))
    
    return render_template('index.html')
	Handles the main page where users can upload their video and submit the form.

**Output**:
![image](https://github.com/user-attachments/assets/f227b612-495c-452e-97e6-25649da07a72)

![image](https://github.com/user-attachments/assets/23e5eab5-b93f-43bf-8872-62ef6c9c5b0c)



![video](https://github.com/user-attachments/assets/3e65818f-9984-4365-8b43-2fd8a85e51b3)



•	Result Route:
@app.route('/result')
def result():
    return render_template('result.html')
	Displays the result page where the download link for the teaser video is provided.
•	Download Route:
@app.route('/download')
def download_file():
    return send_file("teaser0.mp4", as_attachment=True)
	Provides a link for users to download the generated teaser video.
Contributing
We welcome contributions to improve the Video Teaser Generator. Please follow these steps:
1.	Fork the repository.
2.	Create a new branch (git checkout -b feature-branch).
3.	Make your changes.
4.	Commit your changes (git commit -am 'Add some feature').
5.	Push to the branch (git push origin feature-branch).
6.	Create a new Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Credits
•	MoviePy for video processing.
•	Whisper for audio transcription.
•	Transformers for text summarization.
•	Flask for the web framework.
•	Teaser Processing Module for handling video processing.
Contact Information
For any inquiries, please contact [Your Name] at [Your Email].


Troubleshooting or FAQ
•	Why is my audio transcription taking so long?
o	Audio transcription time depends on the length of the video and the model being used. For faster processing, consider using the smaller Whisper models.
•	The script fails with a MoviePy error related to ImageMagick. What should I do?
o	Ensure that ImageMagick is installed and correctly configured in your system.
•	My summary is too brief. How can I get a more detailed summary?
o	Adjust the max_new_tokens parameter in the summarize_text function to generate a longer summary.
•	Why is the video processing taking so long?
o	Video processing time depends on the video length and the complexity of the processing. Ensure your server meets the necessary requirements.
•	The download link is broken. What should I do?

> [!NOTE]
> Use the higher version of the python preferably 3.8 or above or else the package installing would be difficult

> [!TIP]
> Running takes time for model to predict be patient.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Don't use in low spec devices else the device may start to over heat.




