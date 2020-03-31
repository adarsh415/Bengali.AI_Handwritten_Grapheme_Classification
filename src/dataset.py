import albumentations
import pandas as pd
import numpy as np
import joblib
from PIL import Image



class BengaliDatasetTrain:
    def __init__(self, folds, image_height, image_width, mean, std):
        df = pd.read_csv('../input/train_folds.csv')
        df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'kfold']]

        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_ids.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values

        self.folds = folds
        self.image_height = image_height
        self.image_width = image_width

        if len(self.folds) == 1:
            self.aug = albumentations.Compose([
                albumentations.Resize(self.image_height, self.image_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(self.image_height, self.image_width, always_apply=True),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=5,
                    p=0.9
                ),
                albumentations.Normalize(mean, std, always_apply=True)
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image = joblib.load(f'../input/image_pickles/Train_{item}.pkl')
        image = image.reshape(137, 236).astype(float)
        image = Image.fromarray(image).convert('RGB')
        # image = tf.keras.preprocessing.image.array_to_img(image)
        image = self.aug(image=np.array(image))['image']
        #image = np.transpose(image, (2, 0, 1)).astype(float)
        return { 'image': image,
                 'grapheme_root': self.grapheme_root,
                 'vowel_diacritic' : self.vowel_diacritic,
                 'consonant_diacritic' : self.consonant_diacritic
                 }

    # def _albumenation(self, image):
    #     if len(self.folds) == 1:
    #         #image = image.resize(self.image_height, self.image_width)
    #         image = tf.image.resize(image, [self.image_height, self.image_width,3])
    #         image = tf.image.per_image_standardization(image)
    #     else:
    #         image = tf.image.resize(image, [self.image_height, self.image_width, 3])
    #         image = tf.image.
    #         image = tf.image.per_image_standardization(image)
