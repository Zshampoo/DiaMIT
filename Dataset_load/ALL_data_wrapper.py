import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from Dataset_load.JS_dataload import JS_get_data
from Dataset_load.BRAINMC_dataload import brainmc_get_data
from Dataset_load.MRA_dataload import mra_get_data
from Dataset_load.BRAINTUMOR_dataload import braintumor_get_data
from Dataset_load.BRAINIDH_dataload import brainidh_get_data


class DataSetWrapper(object):
    def __init__(self,
                 dataset_type,
                 path,
                 batch_size,
                 channels,
                 keep_slices,
                 shape):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.path = path
        self.keep_slices = keep_slices
        self.shape = shape
        self.channels = channels

    def get_data_loaders(self, is_train, droplast, foldindex):
        dataset = None
        if self.dataset_type == 'spinal':
            dataset = SPINALDataset(path=self.path, is_train=is_train, channels = self.channels, keep_slices=self.keep_slices, shape=self.shape)
        elif self.dataset_type == 'braintumor':
            dataset = BRAINTUMORDataset(path=self.path, fold_index= foldindex, is_train=is_train, channels=self.channels,keep_slices=self.keep_slices, shape=self.shape)
        elif self.dataset_type == 'brainidh':
            dataset = BRAINIDHDataset(path=self.path, is_train=is_train, channels=self.channels,keep_slices=self.keep_slices, shape=self.shape)
        elif self.dataset_type == 'mra':
            dataset = MRADataset(path= self.path, is_train=is_train, channels =  self.channels, keep_slices=self.keep_slices, shape=self.shape)
        elif self.dataset_type == 'brainmc':
            dataset = BRAINMCDataset(path= self.path, is_train=is_train, channels =  self.channels, keep_slices=self.keep_slices, shape=self.shape)
        else:
            print('dataset_type value is wrong.')

        data_loader = self.data_loaders(dataset, droplast)

        return data_loader

    def data_loaders(self, dataset, droplast=False):

        num = len(dataset)

        indices = list(range(num))

        np.random.shuffle(indices)

        datasampler = SubsetRandomSampler(indices)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=datasampler, drop_last=droplast, shuffle=False)

        return data_loader


class SPINALDataset(Dataset):

    def __init__(self,
                 path,
                 is_train,
                 channels,
                 keep_slices,
                 shape,
                 ):
        self.is_train = is_train
        if self.is_train == 2:
            self.image, self.report, self.y,self.age,self.sex, self.img_origin, self.subno = JS_get_data(path=path, channels=channels,data_type=is_train,keep_slices=keep_slices, shape=shape)
        else:
            self.image, self.report, self.y,self.age,self.sex = JS_get_data(path=path, channels=channels, data_type=is_train,keep_slices=keep_slices, shape=shape)

    def __len__(self):
        return len(self.report)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # delete "\n" in report
        reports = []
        for t in self.report:
            t = t.replace("\n", "")
            reports.append(t)

        image = self.image[idx]
        report = reports[idx]
        label = self.y[idx]
        age = self.age[idx]
        sex = self.sex[idx]
        # test time
        if self.is_train == 2:
            img_save = self.img_origin[idx]
            subname = str(self.subno[idx])
            return image, report, age, sex, label, img_save, subname

        return image, report, age, sex, label

class BRAINTUMORDataset(Dataset):
    def __init__(self,
                 path,
                 is_train,
                 channels,
                 keep_slices,
                 shape,
                 ):
        self.is_train = is_train

        if self.is_train == 2:
            self.image, self.report, self.y, self.age, self.sex, self.img_origin, self.subname = braintumor_get_data(path=path, channels=channels,data_type=is_train,keep_slices=keep_slices,shape=shape)
        else:
            self.image, self.report, self.y, self.age, self.sex = braintumor_get_data(path=path, channels=channels, data_type=is_train,keep_slices=keep_slices, shape=shape)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # delete "\n" in report
        reports = []
        for t in self.report:
            t = t.replace("\n", "")
            reports.append(t)

        image = self.image[idx]
        report = reports[idx]
        label = self.y[idx]
        age = self.age[idx]
        sex = self.sex[idx]
        # test time
        if self.is_train == 2:
            img_save = self.img_origin[idx]
            subname = str(self.subno[idx])
            return image, report, age, sex, label, img_save, subname

        return image, report, age, sex, label

class BRAINIDHDataset(Dataset):
    def __init__(self,
                 path,
                 is_train,
                 channels,
                 keep_slices,
                 shape,
                 ):
        self.is_train = is_train

        if self.is_train == 2:
            self.image, self.report, self.y, self.age, self.sex, self.img_origin, self.subname = brainidh_get_data(path=path, channels=channels,data_type=is_train,keep_slices=keep_slices,shape=shape)
        else:
            self.image, self.report, self.y, self.age, self.sex = brainidh_get_data(path=path, channels=channels, data_type=is_train,keep_slices=keep_slices, shape=shape)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # delete "\n" in report
        reports = []
        for t in self.report:
            t = t.replace("\n", "")
            reports.append(t)

        image = self.image[idx]
        report = reports[idx]
        label = self.y[idx]
        age = self.age[idx]
        sex = self.sex[idx]
        # test time
        if self.is_train == 2:
            img_save = self.img_origin[idx]
            subname = str(self.subno[idx])
            return image, report, age, sex, label, img_save, subname

        return image, report, age, sex, label

class MRADataset(Dataset):
    def __init__(self,
                 path,
                 is_train,
                 channels,
                 keep_slices,
                 shape,
                 ):
        self.is_train = is_train

        if self.is_train == 2:
            self.image, self.report, self.y, self.age, self.sex, self.img_origin, self.subname = mra_get_data(path=path,channels=channels,data_type=is_train,keep_slices=keep_slices,shape=shape)
        else:
            self.image, self.report, self.y, self.age, self.sex = mra_get_data(path=path, channels=channels,data_type=is_train,keep_slices=keep_slices, shape=shape)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # delete "\n" in report
        reports = []
        for t in self.report:
            t = t.replace("\n", "")
            reports.append(t)

        image = self.image[idx]
        report = reports[idx]
        label = self.y[idx]
        age = self.age[idx]
        sex = self.sex[idx]
        # test time
        if self.is_train == 2:
            img_save = self.img_origin[idx]
            subname = str(self.subno[idx])
            return image, report, age, sex, label, img_save, subname

        return image, report, age, sex, label

class BRAINMCDataset(Dataset):
    def __init__(self,
                 path,
                 is_train,
                 channels,
                 keep_slices,
                 shape,
                 ):
        self.is_train = is_train
        if self.is_train == 2:
            self.image, self.report, self.y, self.age, self.sex, self.img_origin, self.checkno = brainmc_get_data(path=path, channels=channels,
                                                                                data_type=is_train,
                                                                                keep_slices=keep_slices, shape=shape)
        else:
            self.image, self.report, self.y, self.age, self.sex = brainmc_get_data(path=path, channels=channels, data_type=is_train,keep_slices=keep_slices, shape=shape)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # delete "\n" in report
        reports = []
        for t in self.report:
            t = t.replace("\n", "")
            reports.append(t)

        image = self.image[idx]
        report = reports[idx]
        label = self.y[idx]
        age = self.age[idx]
        sex = self.sex[idx]
        # test time
        if self.is_train == 2:
            img_save = self.img_origin[idx]
            subname = str(self.subno[idx])
            return image, report, age, sex, label, img_save, subname

        return image, report, age, sex, label