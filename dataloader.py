from utils import merge_cbc_diffs
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    print("GPU is enabled. Device:", device)
else:
    print("GPU is not enabled.")



def create_train_val_df(train_file, val_size=0.15):
    """This function will create a train, val dataframe from the data directory and the diagnosis.csv file.
    The dataframes will have the following columns:
    1. Patient: The name of the patient folder.
    2. Label: The corresponding label for each patient.

        Args:    
            data_dir: The directory path where the patient data is stored.
            label_file: The file path of the diagnosis.csv file.
            val_size: The proportion of the dataset to include in the validation split. Default is 0.15

        Returns: 
            :Two dataframes - train_patients, val_patients
            :A list of classes to keep
            :The original class counts
    """
    
    train_df = pd.read_csv(train_file, header=0)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['Label'], random_state=42)

    # Class Counts
    train_class_counts = train_df['Label'].value_counts()
    val_class_counts = val_df['Label'].value_counts()

    return train_df, val_df, sorted(train_class_counts.index.tolist()), train_class_counts, val_class_counts


def create_experiment(data_dir, label_file, test_size=0.1, val_size=0.25, exp_name='', description=''):
    """
    This creates a new experiment. It will merge the data directory and the 
    diagnosis.csv file, choose the classes we want to keep and then split into train and test csv files.
    Note that some patients might be repeated in the csv files because the same patient may have multiple blood smears.
    We use this information as it adds on to the data for training.
    The dataframes will have the following columns:
    1. Patient FName: The name of the patient folder.
    2. Label: The corresponding label for each patient.
    3. Accession Number: The accession number of the patient.

        Args:    
            data_dir: The directory path where the patient data is stored.
            label_file: The file path of the diagnosis.csv file.
            test_size: The proportion of the dataset to include in the test split. Default is 0.1
            outfile_suffix: The suffix to be added to the output file names. Default is an empty string.

        Returns: 
            :Two dataframes - train_patients, test_patients
            :A list of classes to keep
    """

    # Define the save paths for the train and test csv files
    os.makedirs(os.path.join('experiments', exp_name), exist_ok=True)
    train_save_path = os.path.join('experiments', exp_name, f'{exp_name}_train_patients.csv')
    test_save_path = os.path.join('experiments', exp_name, f'{exp_name}_test_patients.csv')

    # Read the diagnosis file which will have the Accession Number and the General Dx
    labels_df = pd.read_csv(label_file, header=0)
    # labels_df = pd.read_excel(label_file, header=0, sheet_name='Sheet1')
    labels_df['General Dx'] = labels_df['General Dx'].str.strip()

    # Get the patient folders from the data directory and extract the Accession Number
    patient_folders = pd.DataFrame({'Patient FName': os.listdir(data_dir)})
    patient_folders['Accession Number'] = patient_folders['Patient FName'].apply(lambda x: x.split(';')[0])

    # Merge the data directory and the diagnosis file on the Accession Number
    # all_patients_df will have the following columns: Patient FName, Accession Number, Label
    all_patients_df = patient_folders.merge(labels_df, on='Accession Number', how='inner')
    all_patients_df = all_patients_df[['Patient FName', 'Accession Number', 'General Dx']]
    all_patients_df = all_patients_df.rename(columns={'General Dx': 'Label'})


    # Merging classes and listing out the classes to keep from all_patients_df

    '''
    # Exp 1 - Keep most classes
    classes_to_keep = ['AML', 'Normal BMA', 'B-ALL', 'MDS', 'MCL', 'HCL', 'MZL', 'large B cell lymphoma',\
                    'LGL-PB', 'FL', 'CLL', 'LPL', 'APML', 'Burkitt', 'CML', 'AML (Therapy related)', \
                        'CMML', 'T lymphoblastic lymphoma/leukemia', 'Hodgkin lymphoma']
    
    # Exp 2 - Acute Leaukemias
    classes_to_keep = ['AML', 'B-ALL', 'APML', 'AML (Therapy related)', 'T lymphoblastic lymphoma/leukemia', 'large B cell lymphoma']

    # Exp 3 - Mature Lymphocutic Leukemias
    classes_to_keep = ['MCL', 'HCL', 'MZL', 'large B cell lymphoma', 'LGL-PB', 'FL',\
                        'CLL',  'LPL', 'Burkitt', 'Hodgkin lymphoma']

    # Exp 4 - Simple classes
    classes_to_keep = ['AML', 'B-ALL', 'Normal BMA', 'HCL']
    '''
    # Exp 5 - Acute Leukemias (AML and B-ALL) vs Normal BMA
    Acute_Leukemia = {'AML', 'B-ALL'}
    all_patients_df.loc[all_patients_df['Label'].isin(Acute_Leukemia), 'Label'] = 'Acute Leukemia'
    classes_to_keep = ['Acute Leukemia', 'Normal BMA']
    '''

    # Exp 6 - MDS vs Normal BMA
    classes_to_keep = ['MDS', 'Normal BMA']
    

    # Exp 7 - HCL vs Normal BMA
    classes_to_keep = ['HCL', 'Normal BMA']


    # Exp 8 - 4 way classification
    Acute_Leukemia = {'AML', 'B-ALL'}
    all_patients_df.loc[all_patients_df['Label'].isin(Acute_Leukemia), 'Label'] = 'Acute Leukemia'
    classes_to_keep = ['Acute Leukemia', 'HCL', 'MDS', 'Normal BMA']


    # Exp 9 - 5 way classification
    Bcl = {'CLL', 'HCL', 'LPL', 'large B cell lymphoma'}
    Acute_Leukemia = {'AML', 'B-ALL'}
    Chronic_Leukemia = {'CML', 'CMML'}

    all_patients_df.loc[all_patients_df['Label'].isin(Bcl), 'Label'] = 'BCL'
    all_patients_df.loc[all_patients_df['Label'].isin(Acute_Leukemia), 'Label'] = 'Acute Leukemia'
    all_patients_df.loc[all_patients_df['Label'].isin(Chronic_Leukemia), 'Label'] = 'Chronic Leukemia'

    classes_to_keep = ['Acute Leukemia', 'BCL', 'MDS', 'Chronic Leukemia', 'Normal BMA']
    '''

    # Filter out the classes that we want to keep, separate and save the train and test csv files.
    all_patients_df = all_patients_df.loc[all_patients_df['Label'].isin(classes_to_keep)]

    # Merge the CBC diffs and the AI predicted diffs from hemeparse and chads file
    all_patients_df = merge_cbc_diffs(patient_labels_df=all_patients_df)

    # Drop the duplicated rows
    all_patients_df = all_patients_df.drop_duplicates()

    # Save the train and test csv files
    train_patients, test_patients = train_test_split(all_patients_df, test_size=test_size, stratify=all_patients_df['Label'], random_state=42)
    train_patients.to_csv(train_save_path, index=False)
    test_patients.to_csv(test_save_path, index=False)

    # The validation file is not saved. It is created on the fly when the MILData class is called using the train_patients file.
    # We are only creating the val_df here to add the class counts of the train, val and test in description.txt
    train_df, val_df, classes, og, new = create_train_val_df(train_save_path, val_size=val_size)



    # Every experiment will have a description.txt file that will contain the class count details of the experiment.
    with open(os.path.join('experiments', exp_name, f'description.txt'), 'w') as f:
        f.write(f'Experiment Name: {exp_name}\n')
        f.write(f'Description: {description}.\n\n')

        f.write(f'Train class counts:\n {og}\n')
        f.write(f'Number of samples in train set: {og.sum()}\n\n')

        f.write(f'Validation class counts:\n {val_df["Label"].value_counts()}\n')
        f.write(f'Number of samples in train set: {len(val_df)}\n\n')

        f.write(f'Test class counts:\n {test_patients["Label"].value_counts()}\n')
        f.write(f'Number of samples in test set: {len(test_patients)}\n')


    return train_patients, test_patients, sorted(classes_to_keep)


class MILData(Dataset):
# Create a dataset that can be used for Multiple Instance Learning.
# When called by the dataloader, it will return tensors and matadata for a single patient.
# A bag for a patient consists of multiple instances of cell images in the patients blood smear.
# Each instance is created by taking an embedding of a particular cell image, where
# the embedding is taken from the last layer of a pretrained resnet model.
# The bag is created by taking a random sample of cells from the given patient.
# The size of every bag will be T (the number of cells in the bag) x 1000 (the size of the embedding).
# Where T is a random number between 200 and 300.

    def __init__(self, data_dir, classes, dset_df, device=torch.device('cuda'), \
                 transform=None, T_min=300, T_max=400, patient_debug=None, features_from='heme'):
        """
        Create a dataset that can be used for Multiple Instance Learning.

            Parameters:
                data_dir (str): The directory containing the patient directories of all their cells.
                transform (Optional, Any transform to be applied to the cell images): None
                T_min (int): The minimum number of cells to be sampled in a bag.
                T_max (int): The maximum number of cells to be sampled in a bag.
                label_file (str): The path to the diagnosis.csv file.
                min_class_size (int): The minimum number of patients in a class. If a class has less than this number of patients, then it will be ignored.
                patient_debug (Optional, List of patients to be used for debugging. If not None, then only these patients will be used.): None
        """

        self.data_dir = data_dir
        self.transform = transform
        self.T_min = T_min
        self.T_max = T_max

        self.classes = sorted(classes)
        self.all_patients_df = dset_df.reset_index(drop=True)\
            .drop(['text_data_clindx', 'text_data_final'], axis=1)
        self.label_to_int = {label: i for i, label in enumerate(self.classes)}
        self.int_to_label = {i: label for i, label in enumerate(self.classes)}
        self.device = device

        # Within the patient folder, cells folder, {cell_type} folder, there can be images and features folder.
        # The features folder will contain the embeddings of the cell images, which will be used to create the bags.
        # They can either come from hemeAI model or imagenet model and the corresponding folder will be
        # 'fetures' or 'features_imagenet' respectively.
        self.features_folder = 'features' if features_from == 'heme' else 'features_imagenet'

        self.patient_debug = patient_debug


    def __getitem__(self, index):
        # Returns a tensor of the bag of instances, the paths of the cell images in the bag and the label of the bag.

        # Get the patient folder name (eg: 'H17-3875;S10; - 2023-05-10 17.51.17')
        patient_dir, label = self.all_patients_df.loc[index, ['Patient FName', 'Label']]
        label = self.label_to_int[label]

        # If patient_debug is not None, then only use the patient_debug list to get items.
        if self.patient_debug is not None:
            patient_dir = self.patient_debug[index]

        try:
            patient_cells_file = pd.read_csv(os.path.join(self.data_dir, patient_dir, 'cells', 'cells_info.csv'), header=0)
        except:
            print(f'Error in reading file: {os.path.join(self.data_dir, patient_dir, "cells", "cells_info.csv")}')
        num_of_cells = len(patient_cells_file)

        # Create a bag of the embeddings of T number of cells from the patient. 
        # Stack all the embeddings to create a bag.
        T = np.random.randint(self.T_min, self.T_max)
        if T > num_of_cells:
            Warning(f'Number of cells to sample in the bag is greater than the number of cells in {patient_dir}. Setting T to {num_of_cells}')
            T = num_of_cells


        # Have a look at the patient directory and the cells_info.csv file to understand the structure of the data.
        # cd /resultsv5/H17-3875;S- 2023-05-10 17.51.17
        cell_img_names = patient_cells_file['name'].sample(T, replace=False).values
        cell_embedding_paths = []
        cell_img_paths = []
        for fname in cell_img_names:
            cell_embedding_paths.append(os.path.join(self.data_dir, patient_dir, 'cells', fname.split("-")[0], \
                                           self.features_folder, fname.split(".")[0] + '.pt'))
            
            cell_img_paths.append(os.path.join(self.data_dir, patient_dir, 'cells', fname.split("-")[0], fname))

        tensors = [torch.from_numpy(torch.load(cell_path)) for cell_path in cell_embedding_paths]
        bag = torch.stack(tensors).to(self.device)

        return bag, cell_img_paths, label, patient_dir

    def __len__(self):
        length = len(self.all_patients_df) if self.patient_debug is None else len(self.patient_debug)
        return length
    
    def get_sample_weights(self):
        '''
        Return sample weights for all items in the dataset. The sample weights are used to balance the classes in the 
        dataset during training. The weights are inversely proportional to the class frequencies
        '''
        # Create a weight dictionary for each class proportional to the inverse of the class frequencies
        class_counts = self.all_patients_df['Label'].value_counts()
        class_weights = 1 / class_counts
        class_weights = class_weights / class_weights.sum()
        label_to_sample_weight = class_weights.to_dict()

        # Map the samples to their corresponding weights based on the label.
        # It will be an array of shape (num_samples,)
        sample_weights = self.all_patients_df['Label'].map(label_to_sample_weight)
        return sample_weights.to_numpy()

    def get_single_patient(self, patient):
        # Returns a tensor of the bag of instances, the paths of the cell images in the bag and the label of the bag.
        # This is used to get the data for a single patient. It is used in the evaluate_data.py file.

        index = self.all_patients_df[self.all_patients_df['Accession Number'] == patient].index[0]
        bag, cell_paths, label, patient_dir =  self.__getitem__(index)

        # Save to device
        padding = torch.tensor([bag.shape[0]]).to(self.device)
        label = torch.tensor([label]).to(self.device)

        # Return the data in the same format as the data collator.
        return bag.unsqueeze(0), padding, label, [cell_paths], [patient_dir]


def data_collator(data):
    """
    This function creates a batch from the input data. The input data is a list of tuples, where each tuple contains 
    a bag, image paths, label and patient name. Each bag is a tensor of shape T x E, where T is the number of cells in the bag 
    and E is the embedding size. It will be used to create the dataloader for the MIL model.

    The function pads each bag to the length of the longest bag in the batch, resulting in a batch of shape B x T_max x E, 
    where B is the batch size, T_max is the length of the longest bag in the batch, and E is the embedding size.

    The function returns four items:
    1. The batch of padded bags. Shape: B x T_max x E
    2. The padding lengths of each bag in the batch. Shape: B (eg: [T1, T2, T3, ...])
    3. The labels of each bag in the batch. Shape: B (eg: [0, 1, 0, ...])
    4. The image paths of each cell in the batch. It will be a list of lists. Outer list length B, \
        Inner list legnth T1, T2, T3...
    5. The patient directories of each bag in the batch. A similar list of lists.
    """

    # Check whether all the loaded cells have the same embedding size.
    embedding_size = data[0][0].shape[1]
    assert len(set([sample[0].shape[1] for sample in data])) == 1, "The embedding size for the cells in every bag should be the same."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    T_max = max([sample[0].shape[0] for sample in data])
    batch_out = torch.zeros((len(data), T_max, embedding_size))

    # Iterate over all the bags in the batch and pad them to the length of the longest bag in the batch.
    padding_lengths = []
    batch_labels = []
    batch_img_paths = []
    patient_dirs = []
    for i, (bag, cell_img_paths, label, patient_dir) in enumerate(data):
        batch_out[i, :bag.shape[0], :] = bag
        padding_lengths.append(bag.shape[0])
        batch_labels.append(label)
        batch_img_paths.append(cell_img_paths)
        patient_dirs.append(patient_dir)

    return batch_out.to(device), torch.tensor(padding_lengths).to(device), torch.tensor(batch_labels).to(device), batch_img_paths, patient_dirs


# Usage:
# Create a dataloader that will return a bag of instances.
# loader = DataLoader(MILData('test-data'), batch_size=4, shuffle=True, collate_fn=data_collator)