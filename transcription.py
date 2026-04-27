import os
from pathlib import Path

import joblib
import librosa
import music21
import numpy as np
from music21 import note, stream


MODEL_PATH = Path(__file__).with_name("rf_13_classifier.pkl")
model = joblib.load(MODEL_PATH)


def segment_audio_pyin(y, sr):
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("A2"),
        fmax=librosa.note_to_hz("A5")
    )

    voiced = np.asarray(voiced_flag, dtype=bool)
    frame_samples = librosa.frames_to_samples(np.arange(len(voiced)))

    energy = librosa.feature.rms(y=y)[0]

    segments = []
    start = None

    for i in range(len(voiced)):
        is_voiced = voiced[i]
        low_energy = energy[i] < 0.01

        if is_voiced and start is None:
            start = frame_samples[i]

        elif (not is_voiced or low_energy) and start is not None:
            end = frame_samples[i]
            duration = (end - start) / sr
            if duration > 0.3:
                segments.append((int(start), int(end)))
            start = None

    if start is not None:
        duration = (len(y) - start) / sr
        if duration > 0.3:
            segments.append((int(start), len(y)))

    return segments


def merge_close_segments(segments, sr, gap_thresh=0.15):
    if len(segments) == 0:
        return []

    merged = []
    prev_start, prev_end = segments[0]

    for start, end in segments[1:]:
        gap = (start - prev_end) / sr

        if gap < gap_thresh:
            prev_end = end
        else:
            merged.append((prev_start, prev_end))
            prev_start, prev_end = start, end

    merged.append((prev_start, prev_end))
    return merged


def extract_segment_features(segment_audio, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=segment_audio, sr=sr).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(segment_audio))

    return np.hstack((mfcc, chroma, centroid, zcr))


def predict_segment_pitch(segment_audio, sr):
    features = extract_segment_features(segment_audio, sr)
    features = features.reshape(1, -1)
    pitch = model.predict(features)[0]

    return int(pitch)


def build_notes(y, sr, segments):
    notes = []
    durations = []

    for i in range(len(segments)):
        start, end = segments[i]
        segment_audio = y[start:end]

        if len(segment_audio) < 2000:
            continue

        if np.mean(np.abs(segment_audio)) < 0.01:
            continue

        pitch = predict_segment_pitch(segment_audio, sr)
        duration = (end - start) / sr

        notes.append(pitch)
        durations.append(duration)

        if i < len(segments) - 1:
            next_start = segments[i + 1][0]
            gap = (next_start - end) / sr

            if gap > 0.2:
                notes.append("rest")
                durations.append(gap)

    return notes, durations


def process_wav(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    segments = segment_audio_pyin(y, sr)
    segments = merge_close_segments(segments, sr)
    notes, durations = build_notes(y, sr, segments)

    return notes, durations


def process_midi(file_path):
    score = music21.converter.parse(file_path)

    notes = []
    durations = []

    for element in score.flat.notesAndRests:
        if element.isNote:
            notes.append(element.pitch.midi)
        elif element.isRest:
            notes.append("rest")

        durations.append(element.quarterLength)

    return notes, durations


def build_musicxml(notes, durations, output_path):
    score = stream.Stream()

    for pitch, d in zip(notes, durations):
        if pitch == "rest":
            n = note.Rest()
        else:
            n = note.Note()
            n.pitch.midi = int(pitch)

        n.quarterLength = d
        score.append(n)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    score.write("musicxml", output_path)


def process_file(file_path):
    lower_path = file_path.lower()

    if lower_path.endswith(".wav"):
        return process_wav(file_path)

    if lower_path.endswith(".mid") or lower_path.endswith(".midi"):
        return process_midi(file_path)

    raise ValueError("Unsupported file format")
