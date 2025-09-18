import sys
import json
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import yt_dlp
import torch
from faster_whisper import WhisperModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline
import librosa
from pydub import AudioSegment
from dotenv import load_dotenv
load_dotenv()


class YouTubeAudioProcessor:
    def __init__(self, hf_token=None):
        # Get HF token from environment variable if not provided
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        if not self.hf_token:
            print("Warning: No Hugging Face token found. Set HF_TOKEN environment variable.")
            print("Some models may not work without authentication.")
        
    def download_audio(self, youtube_url):
        print(f"Downloading audio from: {youtube_url}")
        
        output_path = "audio"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': output_path + '.%(ext)s',
            'quiet': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            wav_path = output_path + '.wav'
            if os.path.exists(wav_path):
                return wav_path
            else:
                raise Exception("Audio download failed")
                
        except Exception as e:
            print(f"Error downloading audio: {e}")
            raise

    def detect_language(self, audio_path, duration=30):
        print("Detecting language...")
        try:
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio_path, language=None)
            detected_lang = info.language
            confidence = info.language_probability
            
            print(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")
            return detected_lang, confidence
            
        except Exception as e:
            print(f"Language detection failed: {e}, defaulting to 'en'")
            return "en", 0.5

    def perform_diarization(self, audio_path):
        print("Performing speaker diarization...")
        
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            diarization = pipeline(audio_path)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2), 
                    "speaker": speaker
                })
                
            print(f"Found {len(set(seg['speaker'] for seg in segments))} speakers in {len(segments)} segments")
            return segments
            
        except Exception as e:
            print(f"Diarization failed: {e}")
            audio = AudioSegment.from_wav(audio_path)
            duration = len(audio) / 1000.0
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "SPEAKER_1"}]

    def transcribe_segment(self, audio_path, start, end, language="en"):
        try:
            audio = AudioSegment.from_wav(audio_path)
            segment_audio = audio[int(start*1000):int(end*1000)]
            
            temp_segment = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            segment_audio.export(temp_segment.name, format='wav')

            if language in ['en', 'ru']:
                model = WhisperModel("medium", device="cpu", compute_type="int8")
                segments, info = model.transcribe(temp_segment.name, language=language)
                transcription = " ".join([segment.text for segment in segments]).strip()
            
            # lang detection is not working well for uzbek, so if language isn't en or ru, we assume it's uzbek
            else:
                try:
                    processor = WhisperProcessor.from_pretrained("islomov/navaistt_v2_medium")
                    model = WhisperForConditionalGeneration.from_pretrained("islomov/navaistt_v2_medium")
                    
                    audio_array, sampling_rate = librosa.load(temp_segment.name, sr=16000)
                    
                    input_features = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
                    
                    with torch.no_grad():
                        predicted_ids = model.generate(input_features)
                    
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    
                except Exception as e:
                    print(f"Uzbek model failed, using Whisper medium: {e}")
                    model = WhisperModel("medium", device="cpu", compute_type="int8")
                    segments, info = model.transcribe(temp_segment.name, language="uz")
                    transcription = " ".join([segment.text for segment in segments]).strip()
                
            os.unlink(temp_segment.name)
            return transcription
            
        except Exception as e:
            print(f"Transcription failed for segment {start}-{end}: {e}")
            return ""

    def process_audio(self, youtube_url, output_file="result.json"):
        print("Starting YouTube audio processing...")
        
        try:
            audio_path = self.download_audio(youtube_url)
            
            language, confidence = self.detect_language(audio_path)
            
            diarization_segments = self.perform_diarization(audio_path)
            
            print("Transcribing segments...")
            results = []
            
            for i, segment in enumerate(diarization_segments):
                print(f"Processing segment {i+1}/{len(diarization_segments)}: {segment['speaker']}")
                
                text = self.transcribe_segment(
                    audio_path, 
                    segment['start'], 
                    segment['end'],
                    language
                )
                
                if text.strip():
                    results.append({
                        "speaker": segment['speaker'],
                        "start": segment['start'],
                        "end": segment['end'], 
                        "text": text.strip()
                    })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"Processing complete! Results saved to {output_file}")
            print(f"Total segments: {len(results)}")
            print(f"Language: {language} (confidence: {confidence:.2f})")
            print(f"Audio file saved as: {audio_path}")
            
            return results
            
        except Exception as e:
            print(f"Processing failed: {e}")
            raise


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_audio.py <youtube_url>")
        print("Example: python process_audio.py https://www.youtube.com/watch?v=7NrICBqo-Lg")
        print("\nNote: Set HF_TOKEN environment variable for Hugging Face authentication:")
        print("export HF_TOKEN=your_token_here")
        sys.exit(1)
    
    youtube_url = sys.argv[1]
        
    processor = YouTubeAudioProcessor()
    
    try:
        results = processor.process_audio(youtube_url)
        print(f"\nSuccess! Check result.json for the output")
        
        print("\nSample results:")
        for result in results[:3]:
            print(f"[{result['start']:06.2f}-{result['end']:06.2f}] {result['speaker']}: {result['text'][:100]}...")
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()