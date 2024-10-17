# place your downloaded kaggle.json file in ~/.kaggle/kaggle.json

import kaggle

kaggle.api.authenticate()

print(kaggle.api.dataset_list_files('andrewmvd/heart-failure-clinical-data').files)

kaggle.api.dataset_download_files('andrewmvd/heart-failure-clinical-data', path='../data', unzip=True)
print("Heart Failure Prediction: Dataset downloaded successfully")
kaggle.api.dataset_metadata('andrewmvd/heart-failure-clinical-data', path='../data')
print("Heart Failure Prediction: Metadata downloaded successfully")

# datasets = kaggle.api.dataset_list(search='heart failure prediction')
# print(datasets)
