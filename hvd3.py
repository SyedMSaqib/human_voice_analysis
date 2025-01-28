import subprocess
import os
import sys
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


def vad_with_webrtc(audio, sr, frame_duration=30):
    """
    Use WebRTC VAD for voice activity detection.
    """
    # WebRTC VAD only supports specific sample rates
    if sr not in [8000, 16000, 32000, 48000]:
        raise ValueError(
            "WebRTC VAD supports sample rates of 8000, 16000, 32000, or 48000 Hz only."
        )

    # Convert audio to 16-bit PCM format
    audio_pcm = (audio * 32767).astype(np.int16).tobytes()

    # Calculate frame size (number of samples in each frame)
    frame_size = int(sr * (frame_duration / 1000))  # Frame size in samples

    # Ensure frames are correctly sized
    frames = [
        audio_pcm[i : i + frame_size * 2]
        for i in range(0, len(audio_pcm), frame_size * 2)
    ]

    # Initialize VAD
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # Mode 3 is the most aggressive (sensitive) setting

    # Process each frame
    speech_frames = 0
    for frame in frames:
        if len(frame) == frame_size * 2:  # Ensure correct frame size
            if vad.is_speech(frame, sr):
                speech_frames += 1

    # Calculate total duration of speech
    total_speech_duration = (speech_frames * frame_duration) / 1000
    return total_speech_duration


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

    try:
        # Step 1: Extract audio from video
        extract_audio(video_path, audio_path)

        # Step 2: Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        denoised_audio = reduce_noise_in_audio(audio, sr)
        filtered_audio = bandpass_filter(denoised_audio, sr)

        # Step 3: Perform voice activity detection with WebRTC VAD
        # Step 3: Resample audio for WebRTC VAD (default: 16000 Hz)
        target_sr = 16000
        if sr != target_sr:
            filtered_audio = librosa.resample(
                filtered_audio, orig_sr=sr, target_sr=target_sr
            )
            sr = target_sr

        # Step 4: Perform voice activity detection with WebRTC VAD

        speech_duration = vad_with_webrtc(filtered_audio, sr)
        total_duration = librosa.get_duration(y=filtered_audio, sr=sr)

        # Step 4: Calculate and display human voice percentage
        human_voice_percentage = calculate_percentage(speech_duration, total_duration)
        print(f"Human Voice Percentage: {human_voice_percentage:.2f}%")

    finally:
        # Clean up temporary files
        for temp_file in [audio_path, filtered_audio_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    main()
