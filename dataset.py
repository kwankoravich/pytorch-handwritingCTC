# HandwritingDataset
# Datset class
import pandas as pd
import numpy as np
from skimage import io
import os
from torch.utils.data import Dataset

class CTCData(Dataset):
    """Handwriting dataset Class."""

    def __init__(self, csv_file, root_dir, transform=None, get_char=True, char_dict=None,
                 word_col):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert isinstance(word_col, (int, str))
        self.word_df = pd.read_csv(os.path.join(root_dir, csv_file))
        #self.word_df = self.word_df.iloc[:,word_col].astype(str)
       
            
        
        if get_char and char_dict is None:
            chars = []
            print(self.word_df.head(10))
            #self.word_df = self.word_df.iloc[:,word_col].astype(str)
            self.word_df.iloc[:, word_col].apply(lambda x: chars.extend(list(x)))
            print(chars)
            chars = sorted(list(set(chars)))
            self.char_dict = {c:i for i, c in enumerate(chars, 1)}
            print(self.char_dict)
        else:
            self.char_dict = char_dict
            
        self.root_dir = root_dir
        self.transform = transform
        self.word_col = word_col
        self.max_len = self.word_df.iloc[:, word_col].apply(lambda x: len(x)).max() 

    def __len__(self):
        return len(self.word_df)

    def __getitem__(self, idx):
        
        img_name = self.word_df.iloc[idx, 1]
        #print(idx)
        print(img_name)
        img_filepath = os.path.join(self.root_dir,
                                   img_name)
        #try:
        image = io.imread(img_filepath)
        print(image.shape)    
        #except OSError:
        #image = np.random.randint(0, 255, size=(50, 100), dtype=np.uint8)
            
        if type(self.word_col) == int:
            word = self.word_df.iloc[idx, self.word_col]
        else:
            word = self.word_df[self.word_col].iloc[idx]
            
        sample = {'image': image, 'word': word}
   
        if self.transform:
            sample = self.transform(sample)

        return sample
   
