import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# ================== CONFIG ==================
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
FORCE_TRAIN = True   # 🔥 change to False for inference only


def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])


if FORCE_TRAIN or not os.path.exists(MODEL_FILE):

    print("\n🔁 Training started...\n")

    # ================== 1. LOAD DATA ==================
    housing = pd.read_csv("data/housing.csv")

    # ================== 2. STRATIFIED CATEGORY ==================
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # ================== 3. STRATIFIED SPLIT ==================
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        housing_train = housing.loc[train_index].drop("income_cat", axis=1)
        housing_test = housing.loc[test_index].drop("income_cat", axis=1)

    # Save test set for inference
    housing_test.to_csv("input.csv", index=False)
    housing = housing_train

    # ================== 4. FEATURES & LABELS ==================
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # ================== 5. COLUMN TYPES ==================
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    # ================== 6. PIPELINE ==================
    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    # ================== 7. GRID SEARCH ==================
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    forest = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        forest,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(housing_prepared, housing_labels)

    best_model = grid_search.best_estimator_

    # ================== 8. RESULTS ==================
    print("Best Parameters:", grid_search.best_params_)
    best_rmse = np.sqrt(-grid_search.best_score_)
    print("Best RMSE:", best_rmse)

    # ================== 9. SAVE ==================
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("\n✅ Model and pipeline saved successfully.")

else:

    print("\n⚡ Inference mode...\n")

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_data = pipeline.transform(input_data)

    predictions = model.predict(transformed_data)
    input_data["median_house_value"] = predictions

    input_data.to_csv("predictions.csv", index=False)

    print("Inference completed and saved to predictions.csv")
