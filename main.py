from Models.IRENE.modeling_irene import CONFIGS
from train import Trainer
from Dataset_load.ALL_data_wrapper import DataSetWrapper
from Configs.all_configs import *

if __name__ == '__main__':
    model_parameters = get_brainidh_configs()
    Datawrapper = DataSetWrapper(dataset_type = model_parameters['dataset'],
                                 path = model_parameters['path'],
                                 channels= model_parameters['channels'],
                                 batch_size = model_parameters['batchsize'],
                                 keep_slices = model_parameters['keep_slices'],
                                 shape = model_parameters['shape'])

    trainer = Trainer(datawrapper= Datawrapper, model_parameters=model_parameters)

    trainer.train(repeat_counts=model_parameters['repeat_counts'])
