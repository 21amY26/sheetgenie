# Monophonic Music Transcription

This project transcribes monophonic music into MusicXML. It supports two input paths:

- WAV audio: segmented with `librosa.pyin`, classified with a trained RandomForest model, then exported as MusicXML.
- MIDI files: parsed directly with `music21` and exported as MusicXML.

The Streamlit app provides a simple upload interface for WAV, MID, and MIDI files.

## Project Files

- `app.py`: Streamlit interface for uploading files and downloading MusicXML.
- `transcription.py`: Core transcription logic and file-type routing.
- `rf_13_classifier.pkl`: Trained RandomForest pitch classifier used by the WAV pipeline.
- `pitch_segment.ipynb`: Notebook used during pipeline development.
- `audio_utils.py`: Earlier helper utilities.

## Requirements

Install the main Python dependencies:

```bash
pip install streamlit librosa music21 numpy joblib scikit-learn soundfile
```

`music21` may require a local notation app if you want to open rendered scores outside the generated MusicXML.

## Run The App

```bash
streamlit run app.py
```

Then upload a `.wav`, `.mid`, or `.midi` file. The app will show:

- detected notes
- durations
- number of notes

It will also provide a MusicXML download.

## WAV Pipeline

`process_wav(file_path)` does the following:

1. Loads audio at 16 kHz with `librosa`.
2. Segments voiced regions using `librosa.pyin`.
3. Merges close segments.
4. Extracts MFCC, chroma, spectral centroid, and zero-crossing-rate features.
5. Predicts MIDI pitch using `rf_13_classifier.pkl`.
6. Inserts rests between notes.
7. Cleans and quantizes short durations for valid MusicXML output.

## MIDI Pipeline

`process_midi(file_path)` parses MIDI directly with `music21`:

- notes are exported as MIDI pitch numbers
- rests are exported as `"rest"`
- durations use the MIDI quarter lengths

## MusicXML Output

`build_musicxml(notes, durations, output_path)` writes the final note and rest sequence to MusicXML using `music21`.

## Supported Formats

- `.wav`
- `.mid`
- `.midi`

Unsupported file types raise `ValueError("Unsupported file format")`.
