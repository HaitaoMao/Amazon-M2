from recbole.data.dataloader.general_dataloader import TrainDataLoader, FullSortEvalDataLoader

from data.transform import unisrec_construct_transform


class CustomizedTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['unisrec_transform'] is not None:
            self.transform = unisrec_construct_transform(config)


class CustomizedFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['unisrec_transform'] is not None:
            self.transform = unisrec_construct_transform(config)
