import datasets as ds
import numpy as np
import cv2 
import torch
import json

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, is_training, config_path, mode=None, returns=["image_in", "image_out", "description"]):
        super().__init__()
        
        if mode == "forward":
            returns = ["image_in", "image_out", "description"]
        elif mode == "reversed":
            returns = ["image_out", "image_in", "description"]
        elif mode == "classifier":
            returns = ["image_in", "label", "description"]
        elif mode == "classifier_spectrum":
            returns = ["image_out", "label", "description"]
        elif mode != None:
            raise ValueError(f"Invalid mode: '{mode}'")

        with open(config_path, 'r') as file:
            config = json.load(file)

        self.returns = returns
        self.input_shape = tuple(config.get("INPUT_SHAPE"))
        self.output_shape = tuple(config.get("OUTPUT_SHAPE"))
        self.train_fraction = config.get("TRAIN_FRACTION")
        self.test_fraction = config.get("TEST_FRACTION")
        self.batch_size = config.get("BATCH_SIZE")
        self.buffer_size = config.get("BUFFER_SIZE")
        self.input_folders = config.get("INPUT_FOLDERS")
        self.output_folders = config.get("OUTPUT_FOLDERS")

        self.file_names_in = []
        for folder in self.input_folders:
            self.file_names_in.extend(ds.glob(f"{folder}/*.ds", no=("meta",)))

        self.file_names_out = []
        for folder in self.output_folders:
            self.file_names_out.extend(ds.glob(f"{folder}/*.ds", no=("meta",)))

        rng = np.random.default_rng(seed=42)

        indices = np.arange(len(self.file_names_in))
        rng.shuffle(indices)

        self.file_names_in = np.array(self.file_names_in)[indices]
        self.file_names_out = np.array(self.file_names_out)[indices]

        index1 = int(self.train_fraction*len(self.file_names_in))
        index2 = int((1-self.test_fraction)*len(self.file_names_in))

        assert index1 <= index2

        if is_training:
            self.file_names_in = self.file_names_in[:index1]
            self.file_names_out = self.file_names_out[:index1]
            self.n_samples = config.get("N_SAMPLES_TRAIN")
        else:
            self.file_names_in = self.file_names_in[index2:]
            self.file_names_out = self.file_names_out[index2:]
            self.n_samples = config.get("N_SAMPLES_TEST")

        if self.n_samples == None:
            self.n_samples = 0
            for file_name_in in self.file_names_in:
                file_in = ds.load(file_name_in)
                self.n_samples += len(file_in.raw)
            print(f"Counted {self.n_samples} samples")

    def __iter__(self):
        indices = np.arange(len(self.file_names_in))
        np.random.shuffle(indices)

        self.file_names_in = np.array(self.file_names_in)[indices]
        self.file_names_out = np.array(self.file_names_out)[indices]

        n_samples = 0

        for file_name_in, file_name_out in zip(self.file_names_in, self.file_names_out):
            file_in = ds.load(file_name_in)
            data_in = np.float32(file_in.raw)/255
            n_samples += len(data_in)

            if "image_out" in self.returns:
                file_out = ds.load(file_name_out)
                assert np.array_equal(file_out.label, file_in.label)
                data_out = np.float32(file_out.raw)/2**16
                data_out = data_out[:,8:72,60:220]

            for i in range(len(data_in)):
                assert self.input_shape == data_in[i].shape

                values = []
                for value_name in self.returns:
                    if value_name == "image_in":
                        values.append(data_in[i])
                    elif value_name == "image_out":
                        image_out = data_out[i]
                        if self.output_shape != image_out.shape:
                            image_out = cv2.resize(image_out, dsize=self.output_shape[::-1], interpolation=cv2.INTER_LINEAR)
                            image_out = np.float32(image_out)
                        values.append(image_out)
                    elif value_name == "description":
                        values.append(file_in.description[i])
                    elif value_name == "label":
                        values.append(torch.tensor(file_in.label[i]).long())
                    else:
                        raise ValueError(f"Invalid return: '{value_name}'")
                yield values
        
        assert n_samples == self.n_samples

    def __len__(self):
        return self.n_samples

class ShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = np.random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass
    
    def __len__(self):
        return len(self.dataset)
