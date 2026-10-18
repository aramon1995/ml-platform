from mlplatform.models import logreg, mlp, random_forest, svm, xgboost

MODELS = {m.NAME: m for m in (logreg, svm, random_forest, xgboost, mlp)}
