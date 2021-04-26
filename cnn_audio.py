from pydub import AudioSegment
from pydub.utils import make_chunks
import pandas as pd
import os
import numpy as np
import librosa
from scipy import signal
from scipy.fft import fftshift
from scipy.io import wavfile
from python_speech_features import mfcc,logfbank
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D,Flatten,LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.image import ImageDataGenerator
import skimage.io
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import sounddevice as sd
from scipy.io.wavfile import read,write
from bottle import route, run, template
import threading
from flask import Flask, render_template



sound_to_template = "listening"
SAMPLING_RATE = 44100
def record():
    fs =  SAMPLING_RATE #44100
    seconds = 5
    sd.default.device = ['MacBook Pro Microphone','MacBook Pro Speakers']
    myrecording = sd.rec(int(seconds*fs),samplerate=fs,channels=1,dtype=np.int16)
    # instream = sd.InputStream(samplerate=fs,dtype=np.int16,device = 'MacBook Pro Microphone')
    # outstream = sd.OutputStream(samplerate=fs,dtype=np.int16,device = 'MacBook Pro Speakers')
    # with sd.Stream(device=('MacBook Pro Microphone', 'MacBook Pro Speakers'),
    #                samplerate=fs, 
    #                dtype=np.int16, callback=callback):
    #     print('#' * 80)
    #     print('press Return to quit')
    #     print('#' * 80)
    #     strem = sd.get_stream()
    #     strem.
    sd.wait()
    fileName = np.random.randint(999)
    write(f"spectrograms/live/{fileName}.wav",fs,myrecording)
    # sd.stop()
    # print(myrecording)
    # sd.play(myrecording)
    # exit()
    # instream.start()
    # outstream.start()
    return fileName
def trainingPreproc(filename): #image augmentation
    train_datagen = ImageDataGenerator(rescale=1./255,shear_range=.2,zoom_range=.2,horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(filename,target_size = (64,64),batch_size = 32,class_mode='binary')
    return training_set

def testPreproc(filename):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(filename,target_size = (64,64),batch_size = 32,class_mode='binary')

def process_sound():
    sound_library = []
    names = [_ for _ in os.listdir("audio_samples") if "background_noise" in _]
    # names = [_ for _ in os.listdir("audio_samples") if ".wav" in _]
    
    for i in range(len(names)):
        myaudio = AudioSegment.from_file(f"audio_samples/{names[i]}" , "wav") 
        chunk_length_ms = 5000 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of 5 sec
        #Export all of the individual chunks as wav files
        for j, chunk in enumerate(chunks):
            chunk_name = f"chunk{j}_{names[i]}"
            sound_library.append([chunk_name,names[i]])
            #print ("exporting", chunk_name)
            chunk.export("audio_samples/background_chunks/"+chunk_name, format="wav")
        # return sound_library
    
    sound_library = pd.DataFrame(data=sound_library)
    # print(sound_library)

    return sound_library

def plot_signals(signals):
    fig,axes = plt.subplots(nrows=2,ncols=5, sharex=False,sharey = True,figsize=(20,5))
    fig.suptitle('Time Series', size =16)
    i = 0
    for x in range(1):
        for y in range(3):
            print("THIS IS I "+ str(i))
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1
def plot_mfccs(mfccs):
    fig,axes = plt.subplots(nrows=2,ncols=5, sharex=False,sharey = True,figsize=(20,5))
    fig.suptitle('Mel frequency Series', size =16)
    i = 0
    print(mfccs.keys())
    for x in range(1):
        for y in range(3):
            print("THIS IS I "+ str(i) )
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],cmap='hot',interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1
def calc_fft(y,rate):
    n = len(y)
    freq = np.fft.rfftfreq(n,d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)
def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min

    return X_scaled
data = 'foo'
app = Flask(__name__)

@app.route("/")
def show_page():
    return render_template("index.html", sound = sound_to_template)
    
def main():
    # signals = {}
    # fft = {}
    # fbank = {}
    # mfccs = {}
    # audio_signal = AudioSegment.from_file(f"./audio_samples/chunks/chunk41_in_room_bp_stitched.wav" , "wav")
    # x = audio_signal.get_array_of_samples()
    # x = np.array(x)
    # process_sound()
    # exit()
    ############# CREATE SPECTROGRAMS ################
    fs = 48000
    hop_length = 512 # number of samples per time-step in spectrogram
    n_mels = 128 # number of bins in spectrogram. Height of image
    time_steps = 384 # number of time-steps. Width of image
    #names = [_ for _ in os.listdir("audio_samples/background_chunks") if ".wav" in _]
    # #names = [_ for _ in os.listdir("audio_samples/chunks") if ".wav" in _]
    # print(len(names))
    # for name in tqdm(names):
    #     print("trying " + name)
    #     # path = f"./audio_samples/chunks/{name}"
    #     path = f"./audio_samples/background_chunks/{name}"
    #     y, sr = librosa.load(path, offset=1.0, duration=10.0, sr=fs)
    #     out = f'spectrograms/background_spectrograms/{name}_img.png'
    # # # extract a fixed length window
    #     start_sample = 0 # starting at beginning
    #     length_samples = time_steps*hop_length
    #     window = y[start_sample:start_sample+length_samples]
    # # # convert to PNG
    #     spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
    #     print('wrote file', out)
############# END CREATE SPECTROGRAMS ################
    

################ CREATE TRAIN/TEST ################
    # names = [_ for _ in os.listdir("audio_samples/chunks") if ".wav" in _]
    names = [_ for _ in os.listdir("spectrograms/background_spectrograms") if ".png" in _]
    # labels = [
    #     "call_light",
    #     "scd_machine",
    #     "iv_alarm",
    #     "in_room_bp"
    # ]
    # labels = [
    #     "background_noise"
    # ]
    # df = pd.DataFrame(names)
    # label_col = []
    # for name in tqdm(names):
    #     for label in labels:
    #         if label in name:
    #             label_col.append(label)
    # df.insert(1,"Labels",label_col)
    # print("Added labels")
    # # df = df.sample(frac=1).reset_index(drop=True)
    # # print("Shuffled")
    # train, test = train_test_split(df,test_size=0.2, random_state=42, shuffle=True)
    # # print("data split", type(test))
    
    # for index,row in test.iterrows():
    #     os.rename(f"spectrograms/background_spectrograms/{row[0]}",f"spectrograms/test_set/{row['Labels']}/{row[0]}")
    # for index,row in train.iterrows():
    #     os.rename(f"spectrograms/background_spectrograms/{row[0]}",f"spectrograms/training_set/{row['Labels']}/{row[0]}")
    
    # # train_set = trainingPreproc(train)
    # # test_set = testPreproc(test)
    # print(test)
    # print(len(train_set),len(test_set))
################ BRING IN TRAIN/TEST ################
    try:
        cnn = tf.keras.models.load_model('cnn_model')
        print("loaded file")
    except:
        print("couldn't locate trained network. retraining and saving to file")

        trainFile = 'spectrograms/training_set'
        testFile =  'spectrograms/test_set'
        training_set = trainingPreproc(trainFile)
        test_set = testPreproc(testFile)
####    ############ BRING IN TRAIN/TEST END ################


####    ############ TRAIN CNN ################

        cnn = tf.keras.models.Sequential()
        cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = [64,64,3]))#conv layer
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2)) #max pooling
        cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))#second conv layer layer
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2)) #max pooling 2
        cnn.add(tf.keras.layers.Flatten())#flatten

        cnn.add(tf.keras.layers.Dense(units = 200,activation='relu')) #fully connected layer
        cnn.add(tf.keras.layers.Dense(units = 5,activation='softmax')) #output layer

        cnn.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) #compiler
        cnn.fit(x = training_set,validation_data=test_set,epochs=35)
        cnn.save("cnn_model")

    classifications = ['background_noise','call_light','in_room_bp','iv_alarm','scd_machine']    
    
    i=0
    while(1):
        fileName = record()
        y, sr = librosa.load(f"spectrograms/live/{fileName}.wav", offset=1.0, duration=10.0, sr=fs)
        start_sample = 0 # starting at beginning
        length_samples = time_steps*hop_length
        window = y[start_sample:start_sample+length_samples]
        out = f'spectrograms/live/test_img{i}.png'
        spectrogram_image(window, sr=fs, out=out, hop_length=hop_length, n_mels=n_mels)
        
        test_image = image.load_img(f'spectrograms/live/test_img{i}.png',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)

        result = cnn.predict_classes(test_image/255.0)
        result_prob = cnn.predict(test_image/255.0)
        os.remove(f"spectrograms/live/{fileName}.wav")
        os.remove(f'spectrograms/live/test_img{i}.png')
        
        decision = []
        for x in result_prob[0]:
            decision.append(str(round(x*100,2)))
        max_index = decision.index(max(decision))
        for class_name in classifications:
            if class_name == classifications[max_index]:
                sound_name = class_name
        if int(float(max(decision))) < 96 or sound_name == "background_noise": 
           sound_name = "listening"
        print(sound_name, max(decision))
        global sound_to_template
        sound_to_template = sound_name
        i+=1


    # print(len(names))
    # print(int(len(names)*.8))
    # print(int(len(names) - len(names)*.8))
    #for name in name[:int(len(names)*.8)]:

    # print(type(x))
    # f,t,Sxx = signal.spectrogram(x,fs,)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.savefig('spectrograms/123')
    # sound_library = process_sound()
    # print(sound_library)

    # classes = list(np.unique(sound_library[1]))
    # for c in classes:
    #     wav_file = sound_library[sound_library[1] == c].iloc[0,0]
    #     signal, rate = librosa.load('audio_samples/chunks/'+wav_file, sr=SAMPLING_RATE)
    #     signals[c] = signal
    #     fft[c] = calc_fft(signal,rate)

    #     bank = logfbank(signal[:rate],rate,nfilt = 26,nfft=1103).T
    #     fbank[c] = bank
    #     mel = mfcc(signal[:rate],rate,numcep=13,nfilt=26,nfft=1103).T
    #     mfccs[c] = mel
    # # plot_signals(signals)
    # # plt.show()

    # plot_mfccs(mfccs)
    # plt.show()
    
if __name__ == '__main__':
    #threading.Thread(target=app.run).start()
    main()