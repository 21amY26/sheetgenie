import numpy as np, os, pandas as pd
import librosa, librosa.display, music21
from music21 import note, stream
import joblib

model = joblib.load("rf_13_classifier.pkl")
def extract_features(file_path):
    '''
    extract features from monphonic note audio files
    '''
    y_audio, sr = librosa.load(file_path, sr=16000)

    mfcc = np.mean(librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y_audio, sr=sr).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_audio))

    return np.hstack((mfcc, chroma, centroid, zcr))

def predict_note(file_path):
    ''' 
    predict the note and pitch of a monophonic note audio file
    '''
    feature=extract_features(file_path)
    feature=feature.reshape(1,-1)

    pred_pitch=model.predict(feature)[0]
    pred_note=librosa.midi_to_note(pred_pitch)
    if "♯" in pred_note:
        pred_note=pred_note.replace("♯", "#")
    
    elif "♭" in pred_note:
        pred_note=pred_note.replace("♭", "b")
    
    return pred_pitch, pred_note

def create_seq_sheet_mono(predicted_notes,path):
    s=stream.Stream()
    for i in predicted_notes:
        n=note.Note(i)
        n.quarterLength=1
        s.append(n)

    s.write("musicxml", path)
    print("Sheet music created at",path)


def merge_consecutive(notes, durations):

    merged_notes = []
    merged_durations = []

    for n, d in zip(notes, durations):

        if merged_notes:
            prev_n = merged_notes[-1]
            prev_d = merged_durations[-1]

            # merge ONLY if:
            # same note AND clearly smaller fragment
            if n == prev_n and d < 0.6 * prev_d:
                merged_durations[-1] += d
                continue

        merged_notes.append(n)
        merged_durations.append(d)

    return merged_notes, merged_durations



def seconds_to_beats(d, bpm=120):
    return d / (60 / bpm)



def quantize_duration(d):
    vals=[0.25,0.5,1,2,4,8]
    return min(vals, key=lambda x: abs(x-d))



def create_seq_sheet(notes, durations,file_path):

    s = stream.Stream()

    for p, d in zip(notes, durations):

        n = note.Note()
        n.pitch.midi = int(p)
        beats=seconds_to_beats(d,bpm=120)
        n.quarterLength=quantize_duration(beats)

        s.append(n)

    s.write("musicxml", file_path+'.xml')

    print("Sheet music created at", file_path+'.xml')