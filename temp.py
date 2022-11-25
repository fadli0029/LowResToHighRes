class ISR_Unsplash_Dataset(Dataset):
    def __init__(self, df, col, transform=None):
        '''
        Args:
            df: Dataframe.
            col: low_res or high_res column (str) 
        '''
        self.df = df
        self.col = col
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        sample = io.imread(row[self.col])
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image = sample
        
        height, width = image.shape[:2]
        if isinstance(self.output_size, int):
            if height > width:
                height_, width_ = self.output_size * height/width, self.output_size
            else:
                height_, width_ = self.output_size, self.output_size * width/height
        else:
            height_, width_ = self.output_size
            
        height_, width_ = int(height_), int(width_)
        img = transform.resize(image, (height_, width_))
        
        return img

class ToTensor(object):
    def __call__(self, sample):
        image = sample
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
    