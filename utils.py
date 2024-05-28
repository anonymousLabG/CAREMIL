import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import mlflow
import numpy as np
import evaluate
import shutil
import torch
import ipywidgets as widgets
from IPython.display import display
import matplotlib.gridspec as gridspec


def create_attention_df(data_loader, model, classes, label_to_int, heads):
    """
    Create a dataframe with the attention weights for every patient in the val_loader.
    Since we are using 4 heads for cross-attention, i.e 4 queries to create the bag representation, \
        we will have 4 attention weights for every cell image.
    The attention head will take values of 'mean', 'max', 'min' or 'std' in the dataframe

    The size of the dataframe will be num_of_patients x T x 4 where T is the number of cell images in the patient.
    
    The columns of the attention dataframe will be:
        Patient: The patient ID ('H00-9999')
        Patient Dir: The directory of the patient ('H00-9999;S10; - 2023-09-06 00.12.56')
        Cell Path: The full path of the cell image ('resultsv5/H00-9999;S10- ...')
        Cell fName: The file name of the cell image ('M5-M6-U1-ER1_2362-2.jpg')
        Attention Head: The aggregation function used to create the bag representation ('mean')
        Attention Score: The attention score for the cell image (0.002917) 
    
    The columns for the patient probabilities dataframe will be:
        Patient: The patient ID ('H00-9999')
        Actual Label: The actual label of the patient ('AML')
        Predicted Label: The predicted label of the patient ('Normal BMA')
        Class1: The probability of the patient belonging to class 0 (0.2)
        Class2: The probability of the patient belonging to class 1 (0.8)
        ... same for all classes
    
    """

    # Create global dataframes to store the attention scores and patient probabilities
    classes_ = [f'{cls}:{label_to_int[cls]}' for cls in classes]
    patient_probs = pd.DataFrame(columns=['Patient Dir', 'Patient', 'Actual Label', 'Predicted Label', *classes_])
    attention_scores = pd.DataFrame(columns=['Patient', 'Patient Dir', 'Cell Path', 'Cell fName', 'Attention Head', 'Attention Score'])

    num_heads = len(heads)
    model.eval()
    total_size = 0

    # Iterate over the data_loader and append the attention scores and patient probabilities to the global dataframes
    same_size = []
    for j, batch in enumerate(data_loader):
        batch_x, batch_padding, batch_label, batch_cell_paths, batch_patient_dirs = batch
        pred_probs, instance_attention = model(batch_x, batch_padding) # B x n_classes, B x num_heads x T_max
        pred_labels = torch.argmax(pred_probs, dim=1)

        # Check for duplicates in batch_cell_paths
        same_size.append(len(batch_patient_dirs) == len(set(batch_patient_dirs)))

        # Creating a batch dataframe for the patient probabilities which will be appended to the global dataframe
        batch_probs = pd.DataFrame(columns=['Patient Dir', 'Patient', 'Actual Label', 'Predicted Label', *classes_])
        
        batch_probs['Patient Dir'] = batch_patient_dirs
        batch_probs['Patient'] = [patient_dir.split(';')[0] for patient_dir in batch_patient_dirs]
        batch_probs['Actual Label'] = batch_label.cpu().numpy()
        batch_probs['Predicted Label'] = pred_labels.cpu().numpy()
        batch_probs[classes_] = pred_probs.detach().cpu().numpy()

        patient_probs = pd.concat([patient_probs, batch_probs], ignore_index=True)

        # Appending to the attention_scores dataframe
        # We will iterate over every patient in the batch, create a attention_df for the patient and append it to the global attention_scores dataframe
        patients = []
        patient_dirs = []
        cell_paths = []
        attn_scores =[]
        attn_heads = []

        # pad will be the T value for every patient or the number of cell images in the patient
        # This loop will iterate through each bag in the batch.
        for i, pad in enumerate(batch_padding):
            patients.extend( [batch_patient_dirs[i].split(';')[0]] * num_heads * pad )  # ['H00-9999')']*4*pad
            patient_dirs.extend( [batch_patient_dirs[i]] * num_heads * pad )            # ['H00-9999');S10;-2023-09-06 00.12.56']*4*pad
            cell_paths.extend( batch_cell_paths[i] * num_heads)                         

            repeated_heads = [[head] * pad for head in heads] #   [['mean']*pad, ['max']*pad, ['min']*pad, ['stddev']*pad]
            head_scores = [instance_attention[i, j, 0:pad].detach().cpu().numpy() for j in range(num_heads)]
            [attn_heads.extend(repeated_head) for repeated_head in repeated_heads]
            [attn_scores.extend(head_score) for head_score in head_scores]
            total_size += num_heads * pad

        batch_df = pd.DataFrame(columns=['Patient', 'Patient Dir', 'Cell Path', 'Cell fName', 'Attention Head', 'Attention Score'])
        
        batch_df['Patient'] = patients
        batch_df['Patient Dir'] = patient_dirs
        batch_df['Cell Path'] = cell_paths
        batch_df['Cell fName'] = [os.path.basename(cell_path) for cell_path in cell_paths]
        batch_df['Attention Head'] = attn_heads
        batch_df['Attention Score'] = attn_scores

        attention_scores = pd.concat([attention_scores, batch_df], ignore_index=True)

    model.train()

    # Duplicates maybe 
    print(f'They are all same: {all(same_size)}')
    print(same_size)

    duplicate_check = attention_scores['Cell Path'].value_counts().sort_values(ascending=False)
    print(duplicate_check)

    return patient_probs, attention_scores

def save_cell_image_panel(patient, patient_attention_scores, pred_label, act_label, \
                          tmp_path, heads, patient_dir=None, k=20, patient_dict=None):
    """
    Create a concatenated image of the cell images the model is focusing on for a particular patient.
    The cell images are selected based on the attention scores for the patient.
    For every attention head, we will have a separate concatenated image. In each image, we will take the top k
    cell images based on the attention scores, the bottom k cell images and the k cell images with random attention scores.

        Parameters:
            patient: The patient Accession Number for which we want to create the concatenated image
            patient_attention_scores: The dataframe with the attention scores for the selected patient
            pred_label: The predicted label for the patient
            act_label: The actual label for the patient
            save_path: The path where the concatenated images will be saved
            k: The number of cell images to be selected for each attention head
    """

    # heads = ['mean', 'max', 'min', 'stddev']
    cbc_keys=["Blast", "Lymph", "Neutrophil", "Mono", "WBC"]
    ai_predicted_keys = ["AI_Blast", "AI_Lymph", "AI_Neutrophil", "AI_Mono"]

    for head in heads:
        head_scores = patient_attention_scores[patient_attention_scores['Attention Head'] == head].copy()
        head_scores['rank'] = head_scores['Attention Score'].rank(ascending=False, method='first').astype(int)
        
        if head_scores.shape[0] < k:
            print(f'Number of cell images for patient {patient} ({head_scores.shape[0]}) is too low.')
            print(f'Skipping the patient {patient}')
            return
        
        topk = head_scores.iloc[:k]
        bottomk = head_scores.iloc[-k:]
        randomk = head_scores.sample(k)

        # Create the concatenated image
        # fig, axs = plt.subplots(nrows=4, ncols=k, figsize=(20, 5))
        gs = gridspec.GridSpec(5, k)
        fig = plt.figure(figsize=(20, 5))
        plt.subplots_adjust(hspace=0.05, wspace=0.8)        

        for i, df in enumerate([topk, randomk, bottomk]):
            for j, row in df.reset_index().iterrows():
                img_path = row['Cell Path']
                img = plt.imread(row['Cell Path'])
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'Rnk: {row["rank"]}, Att: {round(row["Attention Score"], 4)}')
                ax.title.set_fontsize(5)

        fig.suptitle(f'{head} head images: Topk, Randomk and Bottomk for {patient}. Actual: {act_label}, Predicted: {pred_label}')
        
        if patient_dict:
            # Create a subplot at the bottom for the actual cbc diffs
            table_ax = fig.add_subplot(gs[3, :])
            plt.subplots_adjust(top=0.85)

            ## Create a table with the diff data
            table_data = [[patient_dict[x] for x in cbc_keys]]
            row_labels = [f'{patient}']
            # table_data = [[patient_dict[x] for x in ai_predicted_keys]]

            table = table_ax.table(cellText=table_data, colLabels=cbc_keys, rowLabels=row_labels,\
                                    cellLoc='center', loc='center')
            table_ax.set_title('Actual CBC Diffs', pad=10, fontsize=10)
            
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.scale(1, 1.5)

            ## Remove the axis and ticks from the table subplot
            table_ax.axis('off')
            table_ax.set_xticks([])
            table_ax.set_yticks([])

            # Create a table for the AI predicted diffs
            table_ax = fig.add_subplot(gs[4, :])
            plt.subplots_adjust(top=0.85)
            
            ## Create a table with the diff data
            table_data = [[patient_dict[x] for x in ai_predicted_keys]]
            table_data[0].append('N/A') # Add an empty cell for the last column
            row_labels = [f'{patient}']
            col_labels = ai_predicted_keys + ['AI_WBC']

            table = table_ax.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels,\
                                    cellLoc='center', loc='center')

            table_ax.set_title('AI Predicted CBC Diffs', pad=10, fontsize=10)
            
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.scale(1, 1.5)

            ## Remove the axis and ticks from the table subplot
            table_ax.axis('off')
            table_ax.set_xticks([])
            table_ax.set_yticks([])

        # Add a subtitle to the axes
        # table_ax.text(0.5, 1.1, 'Subtitle', transform=table_ax.transAxes, fontsize=10, ha='center')

        fig.tight_layout()
        plt.savefig(f'{tmp_path}/{head}_attn_panel.png')
        if patient_dir: mlflow.log_artifact(f'{tmp_path}/{head}_attn_panel.png', patient_dir)
        plt.close()

def save_confusion_matrix(cm, classes, filename):
    '''
    Save the confusion matrix as an image
        Parameters:
            cm: Confusion matrix created from evaluate.load('confusion_matrix'), with normalize set to None
            classes: List of class names
            filename: Name of the file to save the image
    '''
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
    fig.tight_layout()
    cm = cm.astype('float')
    total_samples = cm.sum(axis=1).astype('float')
    for i, s in enumerate(total_samples):
        cm[i, :] = cm[i, :] / max(s, 0.1)  # Avoid division by zero
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix",
           ylabel='True label (Total Samples)',
           xlabel='Predicted label')

    # Add frequency numbers within the image
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha="center", va="center", color="black")  # Display only the first decimal place

    # Make xticks vertical
    plt.xticks(rotation='vertical')

    # Add total samples to the right of the plot
    for i in range(cm.shape[0]):
        ax.text(-0.85, i + 0.13, f'({total_samples[i]})', ha="left", va="center_baseline", color="black")

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def copy_cell_images(patient_attention_scores, patient_dir, root_dir, heads):
    """
    Copy all the cell images for a patient to a new folder for visualization.
    The cell images will be saved with head_<attention score>_<first 2 cell classes>.jpg
    Patient_dir
        |__ cells
            |__ mean_0.0029_M5-M6.jpg
            |__ max_0.0029_M5-M6.jpg
            ...
    Args:
        patient_attention_scores: The dataframe with the attention scores for the patient.
        patient_dir (str): The directory where the patient data wants be saved, eg: 'Validate/Patient_Data/AML_B-ALL_H00-9999')'
        root_dir (str): The root directory for saving the patient data
    """
    # heads = ['mean', 'max', 'min', 'stddev']
    os.makedirs(os.path.join(root_dir, patient_dir, 'cells'), mode=0o777, exist_ok=True)    # mode=0o777 to give full permissions to the folder

    for head in heads:
        head_scores = patient_attention_scores[patient_attention_scores['Attention Head'] == head]
        for index, row in head_scores.iterrows():
            cell_path = row['Cell Path']
            cell_name = row['Cell fName']
            score = row['Attention Score']
            classnames = '-'.join(cell_name.split("-")[:2]) # Taking the first 2 class names eg: M5-M6
            new_name = f'{head}_{score:.5f}_{cell_name}' # head_score_first2cellclasses.jpg
            new_path = os.path.join(root_dir, patient_dir, 'cells', new_name)
            if os.path.exists(new_path):
                print(f'File {new_path} already exists. Skipping this file.')
                # TODO: We do not know why this is happening. The same file for some reason is being copied multiple times.
                # Currently our approach here is just going to be to skip the file if it already exists.
                continue

            # TODO: This copy is giving error Permission Denied [Errno 13], need to fix this.
            try:
                shutil.copy(cell_path, new_path)
            except PermissionError as e:
                print(f'Error copying {cell_path} to {new_path}')
                head, tail = os.path.split(new_path)
                if os.access(head, os.W_OK | os.R_OK):
                    print('Permission to read/write the directory')
                else:
                    print('No permission to read/write the directory')
                    print('Trying copyfile')
                    print(f'Copying {cell_path} to {new_path}')
                    shutil.copyfile(cell_path, new_path)
                    print('That worked!')
                print(e)

def create_patient_data(attention_scores, patient_probs, temp_path, root_dir, \
                        heads, save_path, cbc_diffs_f=None):
    '''
    This will create a folder for each patient, save a concatenated image panel and copy the cell images.
    There can be multiple patient dir for the same accession number so we need to be careful while creating the patient data.
    Args:
        attention_scores: The dataframe with the attention scores for all the patients
        patient_probs: The dataframe with the predicted and actual labels for all the patients
        temp_path: The temporary path where the concatenated images will be saved before being logged to mlflow
        root_dir: The root directory for saving all the data eg: run.info.artifact_uri
        save_path: The path created inside the root dir where the patient data will be saved, eg: 'Validate/Patient_Data'

    - The data will be stored in the following structure:
        - root_dir
            - save_path
                - patient_dir
                    - cells
                        - mean_0.0029_M5-M6.jpg
                        - max_0.0029_M5-M6.jpg
                        ...
                    - max_attn_panel.png 
                    - mean_attn_panel.png
                    ...   
      
    - A patient dir will be created for each patient with the name: {Actual Label}_{Predicted Label}_{Patient}, eg: AML_B-ALL_H00-9999')
    - Save a concatenated image panel for each patient
    - Copy the cell images in the patient folder for easy visualization as: head_<attention-score>_first2cellclasses.jpg
    '''

    # Creating image panels for all the patients
    patient_dirs = patient_probs['Patient Dir']

    # For the same patient, multiple patient dirs could exist because of multiple samples taken.
    # This dictionary will return True for the patient with multiple samples
    patient_probs['multiple_sample'] = patient_probs['Patient'].duplicated(keep=False)
    multiple_samples_exist = patient_probs[['Patient', 'multiple_sample']].set_index('Patient').to_dict()['multiple_sample']

    # patients = patient_probs['Patient Dir'].map(lambda x: x.split(';')[0])
    int_to_label = {k:v for k, v in enumerate(sorted(patient_probs.columns[4:]))}
    cbc_diffs = pd.read_csv(cbc_diffs_f).drop(['text_data_clindx', 'text_data_final'], axis=1)

    patients_with_diffs = pd.merge(patient_probs, cbc_diffs, left_on='Patient', \
                                  right_on='Accession Number', how='left')
    # cbc_diffs_patients = cbc_diffs['Hnumber'].values

    for patient_dir in patient_dirs:

        patient = patient_dir.split(';')[0]

        # Patient actual label, predicted label and the patient cbc diffs
        actual_label = int_to_label[patient_probs[patient_probs['Patient Dir'] == patient_dir].iloc[0]['Actual Label']]
        predicted_label = int_to_label[patient_probs[patient_probs['Patient Dir'] == patient_dir].iloc[0]['Predicted Label']]
        patient_diff = patients_with_diffs[patients_with_diffs['Patient'] == patient].iloc[0].to_dict()

        # Reading the attention scores for a single patient and sorting the scores in descending order
        # Columns: [Patient, Patient Dir, Cell Path, Cell fName, Attention Head, Attention Score]
        patient_attention_scores = attention_scores[attention_scores['Patient Dir'] == patient_dir]
        patient_attention_scores = patient_attention_scores.sort_values(by=['Attention Head', 'Attention Score'], ascending=[True, False])

        # Creating a folder for the patient
        if multiple_samples_exist[patient]:
            patient_save_dir = os.path.join(save_path, \
                f'{actual_label.split(":")[0]}_{predicted_label.split(":")[0]}_{patient_dir}')
        else:
            patient_save_dir = os.path.join(save_path, f'{actual_label.split(":")[0]}_{predicted_label.split(":")[0]}_{patient}')

        save_cell_image_panel(patient, patient_attention_scores, predicted_label, \
                              actual_label, tmp_path=temp_path, patient_dir=patient_save_dir, k=20,\
                                  patient_dict=patient_diff, heads=heads)
        copy_cell_images(patient_attention_scores, patient_save_dir, root_dir, heads=heads)
        # Copying the images with the topk attention scores to a new folder for visualization