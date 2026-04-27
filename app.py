import os
import tempfile

import streamlit as st

from transcription import build_musicxml, process_file


st.title("Monophonic Music Transcription")

uploaded_file = st.file_uploader("Upload WAV or MIDI", type=["wav", "mid", "midi"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded_file.name)
        output_path = os.path.join(tmpdir, "transcription.xml")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            notes, durations = process_file(input_path)
            build_musicxml(notes, durations, output_path)

            st.subheader("Debug Info")
            st.write("Detected notes:", notes)
            st.write("Durations:", durations)
            st.write("Number of notes:", len(notes))

            with open(output_path, "rb") as f:
                xml_data = f.read()

            download_name = os.path.splitext(uploaded_file.name)[0] + ".xml"
            st.download_button(
                "Download XML",
                data=xml_data,
                file_name=download_name,
                mime="application/vnd.recordare.musicxml+xml"
            )

        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
