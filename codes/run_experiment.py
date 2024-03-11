from utils import read_data, preprocessing, data_split, lgb_fit_mlflow
from config import model_settings

# defining the model settings
config = model_settings()

# reading the data
df = read_data(config.data_path, config.selected_features, config.target)

# preprocessing the data
df, labels = preprocessing(df, config.target)

# splitting the data
x_train, x_test, x_val, y_train, y_test, y_val, train_cols, target = data_split(
    df, config.target
)

# fitting the model
model, fi = lgb_fit_mlflow(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
    config.params,
    train_cols,
    tracking_uri=config.connection_string,
    artifact_uri=config.artifact_uri,
)
