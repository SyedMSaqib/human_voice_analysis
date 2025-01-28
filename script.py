import subprocess
import os
import sys
import time
import numpy as np
import librosa
import webrtcvad
from noisereduce import reduce_noise


def extract_audio(video_path, audio_path):
    """
    Extract audio from a video using FFmpeg.
    """
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y"
    subprocess.run(command, shell=True, stderr=subprocess.DEVNULL)


def reduce_noise_in_audio(audio, sr):
    """
    Apply advanced noise reduction to the audio signal.
    """
    return reduce_noise(y=audio, sr=sr)


def bandpass_filter(audio, sr, low_freq=85, high_freq=1200):
    """
    Apply a bandpass filter to isolate human voice frequencies and harmonics.
    """
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    fft[(freqs < low_freq) | (freqs > high_freq)] = 0
    return np.fft.irfft(fft)


def vad_with_webrtc(audio, sr, frame_duration=30, vad_mode=2):
    """
    WebRTC VAD-based speech detection:
    """
    if sr not in [8000, 16000, 32000, 48000]:
        raise ValueError(
            "WebRTC VAD supports sample rates of 8000, 16000, 32000, or 48000 Hz only."
        )

    audio_pcm = (audio * 32767).astype(np.int16).tobytes()
    frame_size = int(sr * (frame_duration / 1000))
    step_size = int(frame_size / 2)

    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)

    total_seconds = int(len(audio) / sr)
    speech_seconds = 0

    for sec in range(total_seconds):
        second_start = sec * sr
        second_end = min((sec + 1) * sr, len(audio))
        second_audio_pcm = (
            (audio[second_start:second_end] * 32767).astype(np.int16).tobytes()
        )

        frames = [
            second_audio_pcm[i : i + frame_size * 2]
            for i in range(0, len(second_audio_pcm), step_size * 2)
            if len(second_audio_pcm[i : i + frame_size * 2]) == frame_size * 2
        ]

        speech_frames = sum(1 for frame in frames if vad.is_speech(frame, sr))
        if speech_frames >= 0.3 * len(frames):
            speech_seconds += 1

    return speech_seconds


def calculate_percentage(speech_duration, total_duration):
    """
    Calculate the percentage of human voice activity in the audio.
    """
    if total_duration <= 0:
        return 0
    return (speech_duration / total_duration) * 100


# Main script
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    audio_path = "temp_audio.wav"
    filtered_audio_path = "filtered_audio.wav"

    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        sys.exit(1)

    start_time = time.time()

    try:

        extract_audio(video_path, audio_path)

        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        denoised_audio = reduce_noise_in_audio(audio, sr)
        filtered_audio = bandpass_filter(denoised_audio, sr)

        target_sr = 16000
        if sr != target_sr:
            filtered_audio = librosa.resample(
                filtered_audio, orig_sr=sr, target_sr=target_sr
            )
            sr = target_sr

        speech_seconds = vad_with_webrtc(filtered_audio, sr)
        total_seconds = int(librosa.get_duration(y=filtered_audio, sr=sr))

        human_voice_percentage = calculate_percentage(speech_seconds, total_seconds)
        print(f"Human Voice Percentage: {human_voice_percentage:.2f}%")

    finally:

        for temp_file in [audio_path, filtered_audio_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        end_time = time.time()
        print(f"Processing Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
