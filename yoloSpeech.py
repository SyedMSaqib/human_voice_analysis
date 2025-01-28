import subprocess
import os
import sys
import time
import cv2
import numpy as np
import librosa
import librosa.display
import webrtcvad
from noisereduce import reduce_noise
from scipy.signal import butter, filtfilt
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_codec(video_path):
    """
    Detect the codec of the input video.
    """
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        codec_name = subprocess.check_output(probe_command, text=True).strip()
        return codec_name.lower()
    except:
        return None


def extract_audio(video_path, audio_path):
    """
    Extract audio from video file.
    """
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        audio_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio extraction error: {e.stderr.decode()}")
        raise


def detect_people_with_yolov8(video_path, conf_threshold=0.7, frame_skip=10):
    """
    Use YOLOv8 to detect frames with people in the video.
    Returns list of tuples with (start_time, end_time) for continuous segments with people.
    """
    model = YOLO("yolov8x.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps
    segments = []
    frame_idx = 0
    segment_start = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                results = model(frame, conf=conf_threshold)
                person_detected = any(
                    any(box.cls == 0 for box in result.boxes) for result in results
                )

                current_time = frame_idx * frame_duration

                if person_detected and segment_start is None:
                    segment_start = current_time
                elif not person_detected and segment_start is not None:
                    segments.append((segment_start, current_time))
                    segment_start = None

            frame_idx += 1

        # Close the last segment if video ends with person detection
        if segment_start is not None:
            segments.append((segment_start, frame_idx * frame_duration))
    finally:
        cap.release()

    return segments


def apply_stricter_bandpass_filter(audio, sr, low_freq=300, high_freq=3000):
    """
    Apply a stricter bandpass filter focused on human voice frequencies.
    """
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype="band")
    return filtfilt(b, a, audio)


def reduce_noise_in_audio(audio, sr):
    """
    Reduce noise in the audio using spectral gating with stricter parameters.
    """
    try:
        reduced = reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=0.98,  # Aggressive noise reduction
            stationary=True,  # Treat noise as stationary
            n_jobs=2,
        )
        return reduced
    except Exception as e:
        logger.warning(f"Noise reduction failed: {str(e)}. Returning original audio.")
        return audio


def vad_with_webrtc(audio, sr, frame_duration=20, vad_mode=3):
    """
    Perform stricter speech detection using WebRTC VAD.
    """
    supported_rates = [8000, 16000, 32000, 48000]
    if sr not in supported_rates:
        logger.warning(f"Sample rate {sr} not supported. Converting to 16000 Hz")
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio_pcm = (audio * 32767).astype(np.int16).tobytes()
    frame_size = int(sr * (frame_duration / 1000))
    step_size = int(frame_size / 2)

    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)

    speech_frames = 0
    total_frames = 0
    consecutive_speech = 0
    min_consecutive_speech = 5  # Increased threshold for stricter detection

    for i in range(0, len(audio_pcm) - frame_size * 2, step_size * 2):
        frame = audio_pcm[i : i + frame_size * 2]
        if len(frame) == frame_size * 2:
            total_frames += 1
            is_speech = vad.is_speech(frame, sr)

            if is_speech:
                consecutive_speech += 1
                if consecutive_speech >= min_consecutive_speech:
                    speech_frames += 1
            else:
                consecutive_speech = 0

    speech_percentage = (speech_frames / total_frames) * 100 if total_frames > 0 else 0
    return speech_percentage


def analyze_audio_segment(audio, sr, start_time, end_time):
    """
    Analyze a specific segment of audio for human voice activity.
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment_audio = audio[start_sample:end_sample]

    if len(segment_audio) == 0:
        return 0.0

    # Preprocess audio
    denoised_segment = reduce_noise_in_audio(segment_audio, sr)
    filtered_segment = apply_stricter_bandpass_filter(denoised_segment, sr)

    # VAD analysis
    speech_percentage = vad_with_webrtc(filtered_segment, sr)

    # Ignore short segments or those with low speech percentage
    min_speech_duration = 0.5  # Minimum 0.5 seconds of continuous speech
    if (end_time - start_time) < min_speech_duration or speech_percentage < 10:
        return 0.0

    return speech_percentage


def cleanup_files(files):
    """
    Clean up temporary files.
    """
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove {file_path}: {str(e)}")


def main():
    """Main function to process video and detect human voice activity."""
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    audio_path = "temp_audio.wav"
    temp_files = [audio_path]

    if not os.path.exists(video_path):
        logger.error(f"Error: File '{video_path}' not found.")
        sys.exit(1)

    processing_start = time.perf_counter()

    try:
        # Detect video codec
        codec = detect_codec(video_path)
        logger.info(f"Detected video codec: {codec}")

        # Extract audio first
        logger.info("Extracting audio from video...")
        extract_audio(video_path, audio_path)

        # Load the extracted audio
        logger.info("Loading audio file...")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Detect people in video
        if codec and codec != "av1":
            try:
                logger.info("Detecting people in video...")
                segments = detect_people_with_yolov8(video_path)
                if not segments:
                    logger.info("No people detected in the video.")
                    sys.exit(0)
            except Exception as e:
                logger.error(f"Error in people detection: {e}")
                sys.exit(1)
        else:
            logger.error("AV1 codec detected or codec unknown. Cannot process video.")
            sys.exit(1)

        # Analyze each segment where people are detected
        total_speech_percentage = 0
        total_duration = 0

        for start_time, end_time in segments:
            segment_duration = end_time - start_time
            speech_percentage = analyze_audio_segment(audio, sr, start_time, end_time)

            if speech_percentage > 10:  # Stricter threshold for human voice
                total_speech_percentage += speech_percentage * segment_duration
                total_duration += segment_duration

            logger.info(
                f"Segment {start_time:.2f}s - {end_time:.2f}s: {speech_percentage:.2f}% voice activity"
            )

        # Calculate weighted average of voice activity
        if total_duration > 0:
            average_speech_percentage = total_speech_percentage / total_duration
            logger.info(
                f"Overall Voice Activity in segments with people: {average_speech_percentage:.2f}%"
            )
        else:
            logger.info(
                "No significant voice activity detected in segments with people"
            )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        cleanup_files(temp_files)
        processing_end = time.perf_counter()
        processing_time = processing_end - processing_start
        logger.info(f"Processing Time: {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()
