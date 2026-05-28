
import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, load_data_hpe_true
from sklearn.model_selection import train_test_split

import joblib
import json
import os
<<<<<<< HEAD
import time
=======
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644

# Model imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Import parameter configs from params_config.py

from params_config import (
    knn_params, svm_params, dt_params, rf_params, gbt_params, xgb_params,
    lgbm_params, catboost_params, gnb_params, mnb_params, mlp_params
    )
from skopt import BayesSearchCV

<<<<<<< HEAD
def train_and_save_one_model(name, model, param_space, x, y, saved_model="saved_models"):
    """
    Train and save a single model, return evaluation results.
    """
    if not os.path.exists(saved_model):
        os.makedirs(saved_model)
    print(f"\n=== Training {name} ===")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=3000, stratify=y, random_state=42
=======
def train_and_save_one_model(name, model, param_space, x, y, save_dir="saved_models"):
    """
    Train and save a single model, return evaluation results.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"\n=== Training {name} ===")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
    )
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro'
    }
    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
<<<<<<< HEAD
        n_iter=10,
=======
        n_iter=5,
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
        scoring=scoring,
        refit='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        return_train_score=True
    )
<<<<<<< HEAD
    
    # Đo thời gian train
    start_time = time.time()
    search.fit(x_train, y_train)
    train_time = time.time() - start_time

    best_model = search.best_estimator_
    model_path = os.path.join(saved_model, f"{name}.joblib")
    joblib.dump(best_model, model_path)

    loaded_best_model = joblib.load(model_path)

    # Dự đoán toàn bộ test set
    start_time = time.time()
    y_pred = loaded_best_model.predict(x_test)
    test_time_total = time.time() - start_time
    test_time_per_image = test_time_total / len(x_test)

    # 🔹 Tính toán chỉ số và làm tròn sẵn
    test_accuracy = round(accuracy_score(y_test, y_pred), 4)
    test_precision = round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4)
    test_recall = round(recall_score(y_test, y_pred, average='macro'), 4)
    test_f1 = round(f1_score(y_test, y_pred, average='macro'), 4)
        
    cv = search.cv_results_
    idx = search.best_index_
    mean_acc = round(cv['mean_test_accuracy'][idx] * 100, 4)
    std_acc = round(cv['std_test_accuracy'][idx] * 100, 4)
    mean_prec = round(cv['mean_test_precision'][idx] * 100, 4)
    std_prec = round(cv['std_test_precision'][idx] * 100, 4)
    mean_rec = round(cv['mean_test_recall'][idx] * 100, 4)
    std_rec = round(cv['std_test_recall'][idx] * 100, 4)
    mean_f1 = round(cv['mean_test_f1'][idx] * 100, 4)
    std_f1 = round(cv['std_test_f1'][idx] * 100, 4)


    train_time = round(train_time, 4)
    test_time_total = round(test_time_total, 4)
    test_time_per_image = round(test_time_per_image * 1000, 4)  # ms

    # 7️ In kết quả (in thuần, không format inline)
    print(f"\n📌 Results for {name}:")
    print("✅ Best hyperparameters:", search.best_params_)

    print("\n📊 Cross-validation (mean ± std):")
    print("  - Accuracy:", mean_acc, "±", std_acc)
    print("  - Precision:", mean_prec, "±", std_prec)
    print("  - Recall:", mean_rec, "±", std_rec)
    print("  - F1-score:", mean_f1, "±", std_f1)

    print("\n🎯 Test set results:")
    print("  - Accuracy:", test_accuracy)
    print("  - Precision:", test_precision)
    print("  - Recall:", test_recall)
    print("  - F1-score:", test_f1)

    print("✅ Training time:", train_time, "s")
    print("✅ Total test time:", test_time_total, "s for", len(x_test), "samples")
    print("✅ Single image test time:", test_time_per_image, "ms")

    result = {
        "best_params": search.best_params_,
        "train_time": train_time,
        "test_time_total":test_time_total,
        "test_time_per_image":test_time_per_image,
=======
    import time
    start_time = time.time()
    search.fit(x_train, y_train)
    train_time = time.time() - start_time
    best_model = search.best_estimator_
    model_path = os.path.join(save_dir, f"{name}.joblib")
    joblib.dump(best_model, model_path)
    
    loaded_best_model = joblib.load(model_path)
    y_pred = loaded_best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='macro')
    test_f1 = f1_score(y_test, y_pred, average='macro')
    cv = search.cv_results_
    idx = search.best_index_
    mean_acc = cv['mean_test_accuracy'][idx]
    std_acc = cv['std_test_accuracy'][idx]
    mean_prec = cv['mean_test_precision'][idx]
    std_prec = cv['std_test_precision'][idx]
    mean_rec = cv['mean_test_recall'][idx]
    std_rec = cv['std_test_recall'][idx]
    mean_f1 = cv['mean_test_f1'][idx]
    std_f1 = cv['std_test_f1'][idx]
        # 7. In kết quả
    print(f"\n📌 Results for {name}:")
    print("✅ Best hyperparameters:", search.best_params_)
    print(f"⏱ Training time: {train_time:.2f} seconds")

    print("\n📊 Cross-validation (mean ± std):")
    print(f"  - Accuracy:  {mean_acc:.2%} ± {std_acc:.4f}")
    print(f"  - Precision: {mean_prec:.2%} ± {std_prec:.4f}")
    print(f"  - Recall:    {mean_rec:.2%} ± {std_rec:.4f}")
    print(f"  - F1-score:  {mean_f1:.2%} ± {std_f1:.4f}")

    print("\n🎯 Test set results:")
    print(f"  - Accuracy:  {test_accuracy:.2%}")
    print(f"  - Precision: {test_precision:.2%}")
    print(f"  - Recall:    {test_recall:.2%}")
    print(f"  - F1-score:  {test_f1:.2%}")
    result = {
        "best_params": search.best_params_,
        "train_time": train_time,
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
        "cv": {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_prec": mean_prec,
            "std_prec": std_prec,
            "mean_rec": mean_rec,
            "std_rec": std_rec,
            "mean_f1": mean_f1,
            "std_f1": std_f1
        },
        "test": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1
<<<<<<< HEAD
            
=======
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
        },
        "model_path": model_path
    }
    print(f"✅ Saved model to {model_path}")
    return result

<<<<<<< HEAD
def train_and_saved_models(model_dict, param_dict, x, y, saved_model="savedmodel",saved_logs_json="saved_logs_json"):
    """
    Train multiple models, save each, and return evaluation results for all.
    saved_logs/pose_data_yoloxpose_tiny_4xb64-300e_coco-416.csv.json
    """
    skeleton_model_name = os.path.splitext(os.path.basename(saved_logs_json))[0]
    if not os.path.exists(saved_model):
        os.makedirs(saved_model)
    results = {}
    for name, model in model_dict.items():
        param_space = param_dict[name]
        name=name + "_" + skeleton_model_name
        results[name] = train_and_save_one_model(name, model, param_space, x, y, saved_model)
    return results

def train_all_models(x, y, saved_model="savedmodel", saved_logs_json="saved_logs"):
=======
def train_and_save_models(model_dict, param_dict, x, y, save_dir="savedmodel"):
    """
    Train multiple models, save each, and return evaluation results for all.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results = {}
    for name, model in model_dict.items():
        param_space = param_dict[name]
        results[name] = train_and_save_one_model(name, model, param_space, x, y, save_dir)
    return results


def save_results_json(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def train_all_models(x, y, save_dir="savedmodel", results_json="results.json"):
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
    """
    Train all supported models, save them, and save results to JSON.
    """
    model_dict = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        # "GradientBoosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(verbose=0),
        "LightGBM": LGBMClassifier(verbose=-1),
        # "CatBoost": CatBoostClassifier(verbose=0),
        "GaussianNB": GaussianNB(),
        # "MultinomialNB": MultinomialNB(),
        "MLPClassifier": MLPClassifier()
    }
    param_dict = {
        "KNN": knn_params,
        "SVM": svm_params,
        "DecisionTree": dt_params,
        "RandomForest": rf_params,
        # "GradientBoosting": gbt_params,
        "XGBoost": xgb_params,
        "LightGBM": lgbm_params,
        # "CatBoost": catboost_params,
        "GaussianNB": gnb_params,
        # "MultinomialNB": mnb_params,
        "MLPClassifier": mlp_params
    }
<<<<<<< HEAD
    
    results = train_and_saved_models(model_dict, param_dict, x, y, saved_model=saved_model,saved_logs_json=saved_logs_json)
=======
    results = train_and_save_models(model_dict, param_dict, x, y, save_dir=save_dir)
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644

    return results


<<<<<<< HEAD

def save_saved_logs_json(results, saved_logs_json):
    with open(saved_logs_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


=======
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train all ML models and save results.")
    parser.add_argument('--csv', type=str,default='feature_ske_csv/pose_data_movenet_tflite.csv', help='Path to input CSV file')
<<<<<<< HEAD
    parser.add_argument('--saved_model', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--saved_logs_json', type=str, default='saved_logs/pose_data_movenet_tflite.json', help='Path to save results JSON')
    args = parser.parse_args()
    x, y, hpe_status = load_data(args.csv)
    # x, y, hpe_status = load_data_hpe_true(args.csv)
    print(x, y, hpe_status)
    results= train_all_models(x, y, saved_model=args.saved_model, saved_logs_json=args.saved_logs_json)
    detection_rate = sum(hpe_status) / len(hpe_status) if len(hpe_status) > 0 else 0
    print(f"Detection Rate: {detection_rate:.4f}")
    results["detection_rate"] = detection_rate
    save_saved_logs_json(results, args.saved_logs_json)
    print(f"All results saved to {args.saved_logs_json}")
=======
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--results_json', type=str, default='saved_models/pose_data_movenet_tflite.json', help='Path to save results JSON')
    args = parser.parse_args()
    # x, y, hpe_status = load_data(args.csv)
    x, y, hpe_status = load_data_hpe_true(args.csv)
    print(x, y, hpe_status)
    results= train_all_models(x, y, save_dir=args.save_dir, results_json=args.results_json)
    detection_rate = sum(hpe_status) / len(hpe_status) if len(hpe_status) > 0 else 0
    results["detection_rate"] = detection_rate
    save_results_json(results, args.results_json)
    print(f"All results saved to {args.results_json}")
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
    
