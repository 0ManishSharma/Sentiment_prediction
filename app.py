from src.Sentiment_prediction.components.data_ingestion import DataIngestion
from src.Sentiment_prediction.components.data_transformation import DataTransformation
from src.Sentiment_prediction.components.model_tranier import SentimentModel
from src.Sentiment_prediction.pipelines.prediction_pipeline import PredictionPipeline
if __name__ == "__main__":
    # # Data_ingestion = DataIngestion()
    # # Data_ingestion.load_data()
    # print("Data ingestion successfully")

    # data_transformer = DataTransformation()
    # X_train, y_train, X_test, y_test = data_transformer.initiate_transformation()

    # trainer = SentimentModel()
    # model, history = trainer.train(X_train, X_test,y_train, y_test)

    predictor = PredictionPipeline()
    print(predictor.predict("This movie was absolutely amazing. The story was engaging from start to finish, and the performances were outstanding. The characters felt real and well-developed, and the emotional moments were handled beautifully. The cinematography and background music added great depth to the scenes. Overall, it was an enjoyable and memorable experience, and I would definitely recommend this movie to everyone."))

  


    