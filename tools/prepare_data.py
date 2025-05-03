import os
import datasets as ds
import cv2
import numpy as np
from tqdm import tqdm

class Preparer:
    def __init__(self, input_data_path, output_data_path, input_shape, divide):
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.input_shape = input_shape
        self.divide = divide

    def preprocess(self, data, labels, name, rot, mult, grayscale=True, invert=False, pad_level=255, power=350):
        print("\t".join([str(x) for x in [data.shape[0], data.shape[1], data.shape[2], name, rot, mult, grayscale, invert, pad_level, power]]))

        # Rotate data
        data = np.rot90(data, rot, axes=(1, 2))

        if not grayscale:
            data = np.where(data >= 127, 255, 0)

        # Invert image
        if invert:
            data = 255-data

        data = data.astype(np.float32)

        if mult % 4 == 0:
            data = np.kron(data, np.ones((1, mult//4, mult//4)))
        else:
            target_shape = (data.shape[0], (data.shape[1]*mult)//4, (data.shape[2]*mult)//4)
            resized_data = np.zeros(target_shape, dtype=np.float32)

            for i, image in enumerate(data):
                resized_data[i] = cv2.resize(image, dsize=target_shape[:0:-1], interpolation=cv2.INTER_NEAREST)

            data = resized_data

        pad_1 = (self.input_shape[0] - data.shape[1])//2
        pad_2 = self.input_shape[0] - pad_1 - data.shape[1]
        pad_3 = (self.input_shape[1] - data.shape[2])//2
        pad_4 = self.input_shape[1] - pad_3 - data.shape[2]
        if pad_1 < 0 or pad_2 < 0 or pad_3 < 0 or pad_4 < 0:
            print("Warning: input image too large")
        if pad_1 < 0:
            data = data[:,-pad_1:,:]
            pad_1 = 0
        if pad_2 < 0:
            data = data[:,:pad_2,:]
            pad_2 = 0
        if pad_3 < 0:
            data = data[:,:,-pad_3:]
            pad_3 = 0
        if pad_4 < 0:
            data = data[:,:,:pad_4]
            pad_4 = 0
        data = np.pad(data, pad_width=((0, 0), (pad_1, pad_2), (pad_3, pad_4)), constant_values=pad_level)

        # Correct for diffrent power settings
        data = data * power / 400

        # Create input folder
        folder_name_in = f"{self.input_data_path}/{name}_{self.input_shape[0]}_{self.input_shape[1]}"
        if not os.path.exists(folder_name_in):
            os.makedirs(folder_name_in)

        if self.output_data_path != None:
            # Get output filenames
            folder_name_out = f"{self.output_data_path}/{name}"
            file_names_out = ds.glob("%s/*.ds" % folder_name_out, no=("meta",))

            n_samples = 0

            for file_name_out in tqdm(file_names_out):
                block_number = int(file_name_out[-7:-3]) - 1
                assert folder_name_out in file_name_out
                file_name_in = f'{file_name_out[:-3]}_input.ds'
                file_name_in = file_name_in.replace(folder_name_out, folder_name_in)
                file_name_in = file_name_in.replace("Toriel120+_VoNetW350L05D150_", "") # workaround for windows max pathlenght

                data_block = data[block_number*self.divide:(block_number+1)*self.divide]

                label_block = labels[block_number*self.divide:(block_number+1)*self.divide]
                description = []
                for x, label in enumerate(label_block):
                    description.append(f"{name} {x+block_number*self.divide} {label}")

                datafile = ds.dataset(data_block, label=label_block, y=np.arange(data_block.shape[1]), x=np.arange(data_block.shape[2]))
                datafile.description = description
                datafile.save(file_name_in, compress=1)

                n_samples += len(data_block)

            assert n_samples == len(data)
        else:
            for block_number in range(len(data)//self.divide):
                file_name_in = os.path.join(folder_name_in, f'{name}_{block_number+1:04d}_input.ds')

                data_block = data[block_number*self.divide:(block_number+1)*self.divide]

                label_block = labels[block_number*self.divide:(block_number+1)*self.divide]
                description = []
                for x, label in enumerate(label_block):
                    description.append(f"{name} {x} {label}")

                datafile = ds.dataset(data_block, label=label_block, y=np.arange(data_block.shape[1]), x=np.arange(data_block.shape[2]))
                datafile.description = description
                datafile.save(file_name_in, compress=1)
