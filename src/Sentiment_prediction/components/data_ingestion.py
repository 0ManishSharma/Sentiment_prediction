from dataclasses import dataclass
import os
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionCongif:
    dataset_dir = "IMDB Dataset.csv"
    train_data_file_path = os.path.join("artifacts","train.csv")
    test_data_file_path = os.path.join("artifacts","test.csv")
    raw_data_file_path = os.path.join("artifacts","raw.csv")
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionCongif()
        os.makedirs(os.path.dirname(self.config.raw_data_file_path),exist_ok=True)
    def load_data(self):
        df = pd.read_csv(self.config.dataset_dir)

        df.to_csv(self.config.raw_data_file_path,header=True,index=False)
        
        train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)

        train_data.to_csv(self.config.train_data_file_path,header=True,index=False)

        test_data.to_csv(self.config.test_data_file_path,header=True,index=False)

        return (
            train_data,
            test_data,
            
        )
