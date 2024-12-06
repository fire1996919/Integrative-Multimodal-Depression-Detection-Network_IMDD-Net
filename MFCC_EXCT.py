import librosa
import pandas as pd
import os
import numpy as np


input_folder = 'D:/Depression/DATA/AVEC/AVEC2014/second_bang/text3'
output_folder = 'D:/Depression/DATA/AVEC/AVEC2014/second_bang/text3'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
       
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.wav', '.csv'))

        
        y, sr = librosa.load(input_path, sr=None)  

        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        #  (Delta MFCC)
        mfcc_delta = librosa.feature.delta(mfccs)

        #  (Delta-Delta MFCC)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        
        features = np.concatenate((mfccs, mfcc_delta, mfcc_delta2), axis=0)

       
        features_df = pd.DataFrame(features.T) 
        features_df.to_csv(output_path, index=False)

print("MFCC extraction and saving completed.")
