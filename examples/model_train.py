"""
Created By: ishwor subedi
Date: 2024-03-28
"""
from src.services.train.captcha_train import ModelTrainer

if __name__ == "__main__":
    trainer = ModelTrainer(test_dir='resources/dataset/dataset_20240328/test',
                           train_dir='resources/dataset/dataset_20240328/train')
    trainer.train()
