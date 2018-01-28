from pydub import AudioSegment
import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
import argparse

VOICE_FRAGMENT_LENGTH = 250

parse = argparse.ArgumentParser(description='')
parse.add_argument("-i","--input",required=True,help="input video")

def sample_voice_process(prefix, SampleVoice):
    song = AudioSegment.from_mp3(SampleVoice)
    sum = int(song.__len__()/VOICE_FRAGMENT_LENGTH)
    # sum = sum+1 最后不能整除的丢弃
    for i in range(sum):
        next = (i + 1) * VOICE_FRAGMENT_LENGTH
        first_10_seconds = song[i * VOICE_FRAGMENT_LENGTH:next]
        index =(i + 1);
        mp3name = prefix + '{:04d}.mp3'.format(index)
        first_10_seconds.export( mp3name, format="mp3")
        y1, sr1 = librosa.load(mp3name, sr=16000)
        mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13, hop_length=190, n_fft=2048)#13*25
        print("==>"+str(i)+"<==",len(mfccs).__str__() + "*" + len(mfccs[0]).__str__())
        np.savetxt(prefix+'{:04d}.txt'.format(index),mfccs)

def show_plot_chart(SampleVoice):
    y1, sr1 = librosa.load(SampleVoice, sr=16000)
    mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13, hop_length=190, n_fft=2048)  # 13*25
    plt.figure(figsize=(20, 4))
    librosa.display.specshow(mfccs, x_axis='time',y_axis='log')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(12, 8))
    # D = librosa.amplitude_to_db(librosa.stft(y1), ref=np.max)
    # plt.subplot(4, 2, 1)
    # librosa.display.specshow(D, y_axis='linear')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Linear-frequency power spectrogram')
    # plt.show()


    # D = librosa.amplitude_to_db(librosa.stft(y1),plt.subplot(4, 2, 2))
    # librosa.display.specshow(D, y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Log-frequency power spectrogram')
    # plt.figure(figsize=(12, 8))
    # plt.show()

def main():
    args = vars(parse.parse_args())
    audioName = args["input"] + ".mp3"
    show_plot_chart(audioName)
    # sample_voice_process(args["input"],audioName)

if __name__ == '__main__':
    main()