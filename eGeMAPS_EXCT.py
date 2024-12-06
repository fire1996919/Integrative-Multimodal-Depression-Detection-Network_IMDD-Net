import opensmile
import os


wav_folder_path = 'D:/Depression/DATA/AVEC/AVEC2014/second_bang/text3'

csv_folder_path = 'D:/Depression/DATA/AVEC/AVEC2014/second_bang/text3'


if not os.path.exists(csv_folder_path):
    os.makedirs(csv_folder_path)


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


for file_name in os.listdir(wav_folder_path):
    if file_name.endswith('.wav'):
       
        wav_path = os.path.join(wav_folder_path, file_name)

      
        features = smile.process_file(wav_path)

        
        csv_path = os.path.join(csv_folder_path, file_name.replace('.wav', '.csv'))

       
        features.to_csv(csv_path)

print('Saved in D:/Depression/DATA/AVEC/AVEC2014/second_bang/egemaps')
