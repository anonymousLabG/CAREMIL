from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from models import MultiInstance
from selfattnmodel import MILSelfAttention
from gatedMILmodel import GatedMIL
from dataloader import MILData, data_collator, create_experiment, create_train_val_df
from utils import create_attention_df, save_confusion_matrix, create_patient_data
import torch
from datetime import datetime
import os
import pandas as pd
import mlflow
from torchsummary import summary
import evaluate
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
import torch.nn as nn


# Experiment dictionaries
exp_args = ['Exp5-AL_NL', 'Exp6-MDS_NL', 'Exp7-HCL_NL', 'Exp8-AL_MDS_HCL_NL', \
            'Exp9-AL_BCL_MDS_CL_NL', 'Exp10_AL_NL', 'Exp11_MDS_NL', 'Exp12_AL_HCL_MDS_NL', 'Exp13_AL_NL']

heads_dict = {
    'normal': ['mean', 'max', 'min', 'stddev'],
    'gm': ['gm5.0', 'gm2.5', 'gm1.0'],
    'lse': ['lse2.5', 'lse1.0', 'lse0.5'],
    'gated': ['gated']
    }


# Training arguments
epochs = 70
mil_aggregation_method = 'normal'   # 'normal' or 'gm' or 'lse'
self_attn_model = False
gated_model = True
features_from = 'imagenet'  # 'heme' or 'imagenet
Experiment = exp_args[8]
run_name = f'Gated-imagenet-Run4'
description = f'New AL NL experiment with new random cohort'
debugging = False

initial_lr = 1e-4   
total_steps = 1000
warmup_steps = 100
step_up_size = 200
batch_size = 16
mil_embed = 1000
mil_head = 256
mil_out = 256
hl_size = 256
init_mil_embed = 1000
attn_head = 128
compute_metrics_every = 20 # Compute metrics every 20 batches
n_classes = None # If None, then the number of classes is taken from the train file
T_min = 300
T_max = 400

if gated_model: mil_aggregation_method = 'gated'
heads = heads_dict[mil_aggregation_method]
train_file = f'/Slide-Level-Classification/experiments/{Experiment}/{Experiment}_train_patients.csv'
test_file = f'/Slide-Level-Classification/experiments/{Experiment}/{Experiment}_test_patients.csv'
data_dir = '/resultsv5'
cbc_diffs_f = '/Slide-Level-Classification/diagnosis/cbc_vectors.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if debugging:
    train_file = '/Slide-Level-Classification/experiments/debugging_train_file.csv'
    run_name = 'Debugging'
    epochs = 1

# Create the model, optimizer and loss function
# Start a mlflow server using: mlflow server --host 127.0.0.1 --port 8080
mlflow.set_tracking_uri("http://127.0.0.1:5004")
# mlflow.create_experiment(f'MIL-{Experiment}', artifact_location=f'MIL-artifacts/{Experiment}', \
#                          tags={'trainfile': train_file,
#                                'testfile': test_file})
mlflow.set_experiment(f"MIL-{Experiment}")


# Create the test dataset only once. After which we use the train file to create the train and val datasets
# The train_df is resampled so that the number of patients in each class is the same.
train_df, val_df, classes, _, _ = create_train_val_df(train_file, val_size=0.25)

n_classes = len(classes)
assert len(classes) == n_classes, f'Number of classes in train file: {len(classes)} \
    is not equal to n_classes: {n_classes}'

# Create the train and val datasets and dataloaders
train_dataset = MILData(data_dir, classes, dset_df=train_df, T_min=T_min, T_max=T_max, features_from=features_from)
val_dataset = MILData(data_dir, classes, dset_df=val_df, T_min=T_min, T_max=T_max, features_from=features_from)

# To oversample minority classes, in the train dataset we will use the WeightedRandomSampler
# Replacement is set to True so that the same patient can be sampled multiple times
sample_weights = train_dataset.get_sample_weights()
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Create the dataloaders
# Shuffle cannot be used with the WeightedRandomSampler, and is set to True for the val_loader only.
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, sampler=sampler)

# This train dataloader will be used at the end for creating patient data
train_loader_wo_oversampling = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

# Will be used for evaluating metrics and creating patient data
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

if gated_model:
    model = GatedMIL(embed_dim=init_mil_embed, mil_hl_size=hl_size, mil_dim=mil_head, output_dim=mil_out, \
                    n_classes=n_classes, dropout=0.25).to(device)
    best_model = GatedMIL(embed_dim=init_mil_embed, mil_hl_size=hl_size, mil_dim=mil_head, output_dim=mil_out, \
                    n_classes=n_classes, dropout=0.25).to(device)

elif self_attn_model:
    model = MILSelfAttention(init_mil_embed, mil_head, n_classes, attn_head_size=attn_head, \
                            agg_method=mil_aggregation_method).to(device)
    best_model = MILSelfAttention(init_mil_embed, mil_head, n_classes, attn_head_size=attn_head, \
                            agg_method=mil_aggregation_method).to(device)
else:
    model = MultiInstance(mil_embed, mil_head, mil_out, n_classes, hl_size=hl_size, \
                            agg_method=mil_aggregation_method).to(device)
    best_model = MultiInstance(mil_embed, mil_head, mil_out, n_classes, hl_size=hl_size, \
                            agg_method=mil_aggregation_method).to(device)

model.train()
optimizer = torch.optim.Adagrad(model.parameters(), lr=initial_lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
lr_scheduler = CyclicLR(optimizer, base_lr=initial_lr, max_lr=initial_lr*10, mode='triangular2', step_size_up=step_up_size, cycle_momentum=False)
# lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (1 + step / warmup_steps) / (total_steps / warmup_steps))
loss_fn = nn.CrossEntropyLoss()



# Log the model summary and the parameters
with mlflow.start_run(run_name=run_name) as run:
    params = {
        'epochs': epochs,
        'learning_rate': initial_lr,
        'batch_size': batch_size,
        'mil_embed': mil_embed,
        'mil_head': mil_head,
        'mil_out': mil_out,
        'n_classes': n_classes,
        'step_up_size': step_up_size,
        'hl_size': hl_size,
        'mil_aggregation_method': mil_aggregation_method,
        'features_from': features_from,
    }
    mlflow.log_params(params)

    run_id = run.info.run_id[:5]
    temp_dir = os.path.join('Tmp', run_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Log the model summary
    with open(f'{temp_dir}/model_summary.txt', 'w') as f:
        f.write(str(model))
        
    mlflow.log_artifact(f'{temp_dir}/model_summary.txt')
    mlflow.set_tag('description', description)

    # Creating the metrics for logging. Confusion matrix is a special case since its not a scalar
    metrics_to_use = ['precision', 'recall', 'accuracy', 'confusion_matrix', 'f1']
    metric_args = [{'average': 'macro'}, {'average': 'macro'}, {}, {'normalize': None, 'labels': list(range(n_classes))}, {'average': 'macro'}]
    metrics = {metric: evaluate.load(metric) for metric in metrics_to_use}

    # Save patient data and model based on best f1 score on the validation set
    best_f1 = 0

    # Training loop
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            step = epoch*len(train_loader) + i
            optimizer.zero_grad()
            batch_x, batch_padding, batch_label, batch_cell_paths, batch_patient_dirs = batch
            pred_probs, instance_attention = model(batch_x, batch_padding)
            pred_labels = torch.argmax(pred_probs, dim=1)
            loss = loss_fn(pred_probs, batch_label)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
            mlflow.log_metric('Train/Loss', loss.item(), step=step)
            mlflow.log_metric('Train/Learning Rate', lr_scheduler.get_last_lr()[0], step=step)
            for metric in metrics.values():
                metric.add_batch(references=batch_label, predictions=pred_labels)
            # [metric.add_batch(references=batch_label, predictions=pred_labels) for metric in metrics.values()]

            if step % compute_metrics_every == 0:
                
                # This will return a dictionary of the form {metric_name: computed_value}
                metric_scores = {tag: metric.compute(**args)[tag] for tag, metric, args in \
                                 zip(metrics.keys(), metrics.values(), metric_args)}
                
                # We will log the confusion matrix as an artifact and the rest as metrics
                for tag, score in metric_scores.items():
                    if tag == 'confusion_matrix':
                        save_confusion_matrix(score, classes, f'{temp_dir}/train_confusion_matrix_{step}.png')
                        mlflow.log_artifact(f'{temp_dir}/train_confusion_matrix_{step}.png', \
                                            'Train/Confusion_Matrix')
                    else:
                        mlflow.log_metric(f'Train/{tag}', score, step=step)


                # Logging actual class label and class probabilities for current patients in the batch
                batch_probs = pd.DataFrame(columns=['Patient', 'Actual Label', 'Predicted Label', *classes])
                rename_dict = {cls: f'{cls}:{train_dataset.label_to_int[cls]}' for cls in classes}
                batch_probs = batch_probs.rename(columns=rename_dict)
                batch_probs['Patient'] = [patient_dir.split(';')[0] for patient_dir in batch_patient_dirs]
                batch_probs['Actual Label'] = batch_label.cpu().numpy()
                batch_probs['Predicted Label'] = pred_labels.cpu().numpy()
                batch_probs[list(rename_dict.values())] = pred_probs.detach().cpu().numpy()

                batch_probs.to_csv(f'{temp_dir}/batch_probs_{step}.csv')
                mlflow.log_artifact(f'{temp_dir}/batch_probs_{step}.csv', f'Train/Batch_Predictions')
                # mlflow.log_table(batch_probs, f'SamplePredictions/probs_{step}.json')

                # Compute metrics on the validation set
                model.eval()

                # Iterate over the validation set and add information to the metrics
                # Use the same step as the train loader to log the metrics
                total_loss = 0
                for j, batch in enumerate(val_loader):
                    batch_x, batch_padding, batch_label, batch_cell_paths, batch_patient_dirs = batch
                    pred_probs, instance_attention = model(batch_x, batch_padding) # B x n_classes, B x n x T_max
                    pred_labels = torch.argmax(pred_probs, dim=1)
                    loss = loss_fn(pred_probs, batch_label)
                    total_loss += loss.item()
                    # Add the batch to the metrics
                    [metric.add_batch(references=batch_label, predictions=pred_labels) for metric in metrics.values()]

                # Call compute on all the metrics and log them along with the confusion matrix and the Val Loss
                mlflow.log_metric('Val/Loss', total_loss / len(val_loader), step=step)


                # Logging the metrics for the validation set

                # This will return a dictionary of the form {metric_name: computed_value}
                metric_scores = {tag: metric.compute(**args)[tag] for tag, metric, args in \
                                 zip(metrics.keys(), metrics.values(), metric_args)}
                
                # If the f1 score is the best:
                # Save the model_dict, log the metrics and the confusion matrix by mentioning it in the tags
                if metric_scores['f1'] > best_f1:
                    best_f1 = metric_scores['f1']
                    best_model_dict = model.state_dict()
                    mlflow.set_tags({'Best_F1': metric_scores['f1'],
                                     'Best_accuracy': metric_scores['accuracy'],
                                     'Best_precision': metric_scores['precision'],
                                     'Best_recall': metric_scores['recall'],
                                     'Best_Loss': total_loss / len(val_loader),
                                     'Best_Step': step,
                                     })

                    for tag, score in metric_scores.items():
                        if tag == 'confusion_matrix':
                            save_confusion_matrix(score, classes, \
                                                  f'{temp_dir}/confusion_matrix_{step}.png')
                            save_confusion_matrix(score, classes, \
                                                  f'{temp_dir}/confusion_matrix_best_f1.png')
                            
                            mlflow.log_artifact(f'{temp_dir}/confusion_matrix_{step}.png', \
                                                'Validate/Confusion_Matrix')
                            mlflow.log_artifact(f'{temp_dir}/confusion_matrix_best_f1.png', \
                                                'Validate/Confusion_Matrix')
                        else:
                            mlflow.log_metric(f'Val/{tag}', score, step=step)
      
                else:
                    for tag, score in metric_scores.items():
                        if tag == 'confusion_matrix':
                            save_confusion_matrix(score, classes, f'{temp_dir}/confusion_matrix_{step}.png')
                            mlflow.log_artifact(f'{temp_dir}/confusion_matrix_{step}.png', 'Validate/Confusion_Matrix')
                        else:
                            mlflow.log_metric(f'Val/{tag}', score, step=step)
                
                model.train()

        # Save the model after every epoch
        # mlflow.pytorch.log_model(model, f'model_epoch_{epoch}.pt')
        # torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')
                
    # Save the best model
    best_model.load_state_dict(best_model_dict)
    mlflow.pytorch.log_model(best_model, 'best_model')

    # Saving attention values on the validation set
    patient_probs, attention_scores = create_attention_df(val_loader, best_model, \
                                classes, train_dataset.label_to_int, heads=heads)
    attention_scores.to_csv(f'{temp_dir}/attention_scores.csv')
    patient_probs.to_csv(f'{temp_dir}/patient_probs.csv')
    mlflow.log_artifact(f'{temp_dir}/attention_scores.csv', 'Validate')
    mlflow.log_artifact(f'{temp_dir}/patient_probs.csv', 'Validate')
    create_patient_data(attention_scores, patient_probs, temp_path=temp_dir, heads = heads,
                        save_path='Validate/Patient_Data', root_dir=run.info.artifact_uri, \
                        cbc_diffs_f=train_file)


    # Saving attention values on the training set
    patient_probs, attention_scores = create_attention_df(train_loader_wo_oversampling, best_model, \
                                        classes, train_dataset.label_to_int, heads=heads)
    attention_scores.to_csv(f'{temp_dir}/train_attention_scores.csv')
    patient_probs.to_csv(f'{temp_dir}/train_patient_probs.csv')
    mlflow.log_artifact(f'{temp_dir}/train_attention_scores.csv', 'Train')
    mlflow.log_artifact(f'{temp_dir}/train_patient_probs.csv', 'Train')
    create_patient_data(attention_scores, patient_probs, temp_path=temp_dir, heads=heads,
                        save_path='Train/Patient_Data', root_dir=run.info.artifact_uri, cbc_diffs_f=train_file)


    # # Save the model after training
    # mlflow.pytorch.log_model(model, 'model')
    # mlflow.end_run()

#%%
# import evaluate
# from utils import save_confusion_matrix

# cfm_metric = evaluate.load("confusion_matrix")
# cfm_metric.add_batch(references=[1, 2, 3, 2, 1, 1, 0, 2], predictions=[1, 0, 3, 2, 2, 1, 0, 3])
# cm = cfm_metric.compute(normalize=None, labels=[0, 1, 2, 3])
# classes = ['AL', 'MDS', 'HCL', 'NL']
# print(cm['confusion_matrix'].sum(axis=1))
# save_confusion_matrix(cm['confusion_matrix'], classes, 'confusion_matrix_delete.png')

# print(cm)

# %%
