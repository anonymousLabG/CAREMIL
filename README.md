CAREMIL: Cell AggRegation, Explainable, Multiple
Instance Learning for Leukemia Diagnosis

This repository is an official implementation of CAREMIL: Cell AggRegation, Explainable, Multiple
Instance Learning for Leukemia Diagnosis, at Neurips 2024. 

We aimed to build a practical computational pipeline to help hematologists and present a new explainable model for making diagnosis using cell images.
<Insert figure>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
Due to the sensitive nature of our data and the presence of electronic health records, we will not be releasing any data and will only release code and steps needed to reproduce our results.

We need to create our train and test splits. For each patient we need the patient name, label, AI predicted differentials, and cell images extracted from their whole slide image.

To create a new experiment:

```
train_df, test_df, classes = create_experiment(data_dir=data_dir, label_file=diagnosis_file, \
                                                  test_size=0.25, exp_name=Exp_Name, description=description)
```

The train.py file will:
- Consist of config parameters to choose specifications about the model.
- Split the train data into train and validate.
- Run training on the selected model
- Will use mlflow to keep track of the training, and measure AUC, F1, confusion matrix and accuracy of the train and validate sets
- Create attention figures shown in the paper to help the pathologist understand every prediction the model is making.

```
python train.py
```

## Evaluate Results
We use the evalauate_data.py file to run evaluation on our test sets and measure performance.

```
python evaluate_data.py
```

## Results
CAREMIL outperforms Gated MIL in two of three experiments. For MDS vs NL, Gated MIL shows a 
modestly higher performance.  In all three experiments, the use of the DeepHeme encoder improves performance when compared to the use of
an ImageNet encoder by a wide margin

| Models / Experiments               | AL vs NL         | MDS vs NL         | AL, HCL, MDS, NL  |
|------------------------------------|------------------|-------------------|-------------------|
| Cytometry-Based ML Classifiers     | 0.901±0.018      | 0.667±0.000       | 0.528±0.042       |
| Gated MIL - ImageNet               | 0.472±0.026      | 0.516±0.021       | 0.289±0.026       |
| Self Attention - ImageNet          | 0.828±0.020      | 0.577±0.018       | 0.439±0.023       |
| Gated MIL - DeepHeme               | 0.938±0.007      | **0.808±0.045**   | 0.610±0.008       |
| Self Attention - DeepHeme (CAREMIL)| **0.951±-0.006** | 0.800±0.026       | **0.670±0.027**   |
