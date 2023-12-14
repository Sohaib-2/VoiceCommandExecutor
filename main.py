from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from openai import OpenAI
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
from pathlib import Path

from playsound import playsound

# Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"


def record_audio(sample_rate=44100, duration=7):
    recorded_audio = []

    def callback(indata, frames, time, status):
        recorded_audio.extend(indata.copy())

    with sd.InputStream(samplerate=sample_rate, blocksize=int(sample_rate / 10), callback=callback):
        print("Recording...")
        sd.sleep(duration * 1000)

    print("Recording finished!")

    audio_data = np.array(recorded_audio, dtype=np.float32)
    wav_file_path = "recorded_audio.wav"
    wav.write(wav_file_path, sample_rate, audio_data)
    print(f"Audio saved to {wav_file_path}")
    return wav_file_path


def get_transcript(api_key, audio_file_path):
    client = OpenAI(api_key=api_key)

    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript


def delete_audio_file(file_path):
    try:
        os.remove(file_path)
        print(f"Audio file {file_path} deleted successfully.")
    except Exception as e:
        print(f"Error deleting audio file: {e}")


def speak_response(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)
    # Play the generated speech file using sounddevice or another library
    file_path = "speech.mp3"
    playsound(file_path)
    os.remove(speech_file_path)  # Delete speech file after playing


def main():
    print("Welcome to the Interactive Voice Assistant Program!")
    while True:
        # Step 1: Record audio
        audio_file_path = record_audio()

        # Step 2: Get transcript using Speech to text
        transcript = get_transcript(OPENAI_API_KEY, audio_file_path)
        print(transcript)

        # Step 3: Create and run the agent
        agent_executor = create_python_agent(
            llm=ChatOpenAI(temperature=0, max_tokens=250, model="gpt-3.5-turbo-0613", api_key=OPENAI_API_KEY),
            tool=PythonREPLTool(),
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )

        # Step 4: Define agent instructions using the obtained transcript
        agent_instructions = (
            f"You have to do what the user will ask you to do using python code."
            f" Here is the user's input: {transcript}"
        )

        # Step 5: Run the agent
        output = agent_executor.run(agent_instructions)
        # print("Agent Output:", output)
        # Step 6: Speak to user
        speak_response(output)

        # Step 7: Delete the audio file
        delete_audio_file(audio_file_path)
        user_input = input("To continue press -> 1\nTo exit press -> 0 ")
        if user_input == "0":
            break


if __name__ == "__main__":
    main()
