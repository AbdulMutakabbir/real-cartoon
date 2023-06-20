from torch.utils.data import Dataset
from torch import is_tensor, Tensor
from PIL import Image
from pandas import concat, read_csv
import os

class RealCartoon(Dataset):
    """This is a dataset class to load the real and fake dataset with bool labels. 
    """

    def __init__(self, csv_file:str, set_type:str, root_dir:str="./", transforms=None, random_state:int=1) -> None:
        super().__init__()
        self.root_dir = root_dir

        # identify the type of dataset
        self.set_type = set_type
        assert self.set_type in ['train', 'test', 'val'], f"Invalid set type!!!"

        # load the dataset
        self.csv_file = csv_file
        self.dataset = read_csv(csv_file)
        # split into real and cartoon
        real_df = self.dataset[self.dataset.is_real == 1].sample(random_state=random_state,frac=1)
        fake_df = self.dataset[self.dataset.is_real == 0].sample(random_state=random_state, frac=1)
        # get test, train, val dataset
        train_size = int(len(real_df)*0.8)
        test_size = int(len(real_df)*0.1)
        real_df = real_df[:train_size] if self.set_type == 'train' else real_df[train_size:train_size+test_size] if self.set_type == 'test' else real_df[train_size+test_size:]
        train_size = int(len(fake_df)*0.8)
        test_size = int(len(fake_df)*0.1)
        fake_df = fake_df[:train_size] if self.set_type == 'train' else fake_df[train_size:train_size+test_size] if self.set_type == 'test' else fake_df[train_size+test_size:]
        # combine the datasets
        self.dataset = concat([real_df,fake_df], axis=0)
        # randomly sample the dataset
        self.dataset = self.dataset.sample(random_state=random_state, frac=1, ignore_index=True)
        # load transforms
        self.transforms = transforms


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        image = Image.open(img_name)
        if image.mode == "L":
            img_name = os.path.join(self.root_dir, self.dataset.iloc[1, 0])
            image = Image.open(img_name)

        label = Tensor([self.dataset.iloc[idx, -1]])

        if self.transforms:
            image = self.transforms(image)

        return (image, label)
