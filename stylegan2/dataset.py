from io import BytesIO
import pandas as pd
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):

    # def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = pd.read_csv("../annotations_slices_medium.csv", engine='python')
        self.root_dir = path
        self.transform = transform
    
    def __len__(self):
        return (len(self.annotations))

    def __getitem__(self,index):
        volume_name = os.path.join(self.root_dir,
        self.annotations.iloc[index,0])
        np_volume = np.load(volume_name)
        volume = Image.fromarray(np_volume)
        # annotations = self.annotations.iloc[index,0].as_matrix()
        # annotations = annotations.astype('float').reshape(-1,2)
        sample = volume

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    # def __init__(self, path, transform, resolution=256):
    #     self.env = lmdb.open(
    #         path,
    #         max_readers=32,
    #         readonly=True,
    #         lock=False,
    #         readahead=False,
    #         meminit=False,
    #     )

    #     # if not self.env:
    #     #     raise IOError('Cannot open lmdb dataset', path)

    #     # with self.env.begin(write=False) as txn:
    #     #     self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    #     self.resolution = resolution
    #     self.transform = transform

    #     self.data = 

    # def __len__(self):
    #     return 360

    # def __getitem__(self, index):
    #     # with self.env.begin(write=False) as txn:
    #     #     key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
    #     #     img_bytes = txn.get(key)

    #     # buffer = BytesIO(img_bytes)


    #     img = Image.open(buffer)
    #     img = self.transform(img)

    #     return img
