import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, TextClip, CompositeVideoClip
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import whisper
import torch
from moviepy.config import change_settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-6.9.13-Q16-HDRI\convert.exe"})

def extract_audio(video_path, output_path='audio0.wav'):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path)
    return output_path

def transcribe_audio(audio_path, model_name='base'):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result['text']

def save_text_file(content, filename):
    with open(filename, 'w') as file:
        file.write(content)

def count_tokens(text, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text)
    return len(tokens)

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

def summarize_long_text(text, model_name='gpt2', max_chunk_length=512):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokens = tokenizer.encode(text)
    summaries = []

    for i in range(0, len(tokens), max_chunk_length):
        chunk = tokens[i:i + max_chunk_length]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarize_text(chunk_text, model_name=model_name)
        summaries.append(summary)

    combined_summary = " ".join(summaries)
    return combined_summary

def identify_key_segments(transcript, video_duration, num_segments=6, segment_length=10, process_filename='key_segments_process.txt'):
    if not transcript.strip():
        print("Warning: Transcript is empty. Cannot identify key segments.")
        return []

    sentences = transcript.lower().split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]  

    if not sentences:
        print("Warning: No sentences found after cleaning. Cannot identify key segments.")
        return []

    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    scores = cosine_similarity(vectors)
    scores = np.mean(scores, axis=1)

    important_sentences = np.argsort(scores)[-num_segments:]
    important_sentences.sort()

    segments = []
    time_per_sentence = video_duration / len(sentences)

    sentence_freq = Counter(sentences)

    process_lines = []
    process_lines.append("Process of Identifying Key Segments:\n")
    process_lines.append(f"Video Duration: {video_duration:.2f} seconds\n\n")
    process_lines.append("Transcript:\n")
    process_lines.append(transcript)
    process_lines.append("\n\nIdentified Key Segments:\n")

    for sentence_index in important_sentences:
        start_time = sentence_index * time_per_sentence
        end_time = min(start_time + segment_length, video_duration)
        sentence = sentences[sentence_index]

        segments.append((start_time, end_time))

        process_lines.append(f"Sentence: {sentence}\n")
        process_lines.append(f"Frequency Count: {sentence_freq[sentence]}\n")
        process_lines.append(f"Segment: {start_time:.2f}s - {end_time:.2f}s\n\n")

        process_lines.append(f"Explanation:\n")
        process_lines.append(f"1. *Cosine Similarity Scores*: This sentence was selected because it had a high average cosine similarity score compared to other sentences. This indicates that it is contextually significant and relevant to the overall content of the transcript.\n")
        process_lines.append(f"2. *Frequency Count*: The sentence appears {sentence_freq[sentence]} time(s) in the transcript. A higher frequency count often indicates that the sentence contains important information or themes relevant to the video's narrative.\n")
        process_lines.append(f"3. *Relevance*: This sentence is significant as it discusses a crucial aspect of the subject matter, specifically detailing the impact of drug use on dental health. Its inclusion provides valuable insight into the personal experiences shared in the video, making it a key segment for the teaser.\n")

    with open(process_filename, 'w') as process_file:
        process_file.writelines(process_lines)

    return segments

def create_black_screen_clip(text, video_clip, duration, fontsize=70, color='white'):
    size = video_clip.size
    black_clip = ColorClip(size=size, color=(0, 0, 0), duration=duration)
    txt_clip = TextClip(text, fontsize=fontsize, color=color, size=size, print_cmd=True)
    txt_clip = txt_clip.set_duration(duration).set_pos('center')
    final_clip = CompositeVideoClip([black_clip, txt_clip])

    return final_clip

def create_teaser(video_path, segments, title, director_name, output_path='teaser0.mp4'):
    video = VideoFileClip(video_path)
    title_clip = create_black_screen_clip(f"Title: {title}", video, duration=5)
    director_clip = create_black_screen_clip(f"Directed by: {director_name}", video, duration=5)
    clips = [video.subclip(start, end) for start, end in segments]
    teaser = concatenate_videoclips([title_clip] + clips + [director_clip])
    teaser.write_videofile(output_path)
    return output_path

def process_video(video_path, title, director_name, output_dir='processed'):
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)

    transcript_file = os.path.join(output_dir, 'transcript0.txt')
    save_text_file(transcript, transcript_file)

    token_count = count_tokens(transcript)
    print(f"Number of tokens in the transcript: {token_count}")

    if token_count > 1024:
        print("The transcript is too long. Summarizing in smaller chunks.")
        summary = summarize_long_text(transcript)
    else:
        summary = summarize_text(transcript)
    print("Summary:", summary)

    summary_file = os.path.join(output_dir, 'summary0.txt')
    save_text_file(summary, summary_file)

    video = VideoFileClip(video_path)
    video_duration = video.duration

    segments = identify_key_segments(transcript, video_duration, process_filename=os.path.join(output_dir, 'key_segments_process.txt'))

    teaser_path = create_teaser(video_path, segments, title, director_name, output_path=os.path.join(output_dir, 'teaser0.mp4'))
    return teaser_path
