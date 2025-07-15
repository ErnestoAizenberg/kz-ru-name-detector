from joblib import load

model = load('name_classifier.joblib')

if __name__ == "__main__":
    classify_and_split_names(model, "kz_names_1.xlsx", [0, 1, 2])
