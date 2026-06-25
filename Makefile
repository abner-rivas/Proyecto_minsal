PYTHON = venv/Scripts/python
FEATURE_TYPE ?= QN

.PHONY: all data features train train-reg train-clf evaluate test clean help

## Pipeline completo: data -> features -> train -> evaluate
all: data features train evaluate

## Limpia dataset, construye targets y genera splits Train/Val/Test
data:
	$(PYTHON) src/data/cleaner.py --features $(FEATURE_TYPE)

## Muestra el feature set seleccionado (Q o QN)
features:
	$(PYTHON) -c "import sys; sys.path.insert(0,'.'); from src.features.engineering import get_feature_cols; cols=get_feature_cols('$(FEATURE_TYPE)'); print(f'Feature set $(FEATURE_TYPE): {len(cols)} columnas'); print(cols)"

## Entrena todos los modelos (regresion + clasificacion)
train: train-reg train-clf

## Entrena modelos de Regresion IMC (LinearRegression + RandomForest)
train-reg:
	$(PYTHON) src/models/regression.py --features $(FEATURE_TYPE) --n-iter 20 --cv 5

## Entrena modelos de Clasificacion (LogReg + RF + XGBoost)
train-clf:
	$(PYTHON) src/models/classification.py --features $(FEATURE_TYPE) --n-iter 20 --cv 5

## Evalua los mejores modelos en test set y genera figuras
evaluate:
	$(PYTHON) src/predict.py --task all --evaluate --features $(FEATURE_TYPE)

## Inferencia final sobre test set (alias de evaluate)
test:
	$(PYTHON) src/predict.py --task all --evaluate --features $(FEATURE_TYPE)

## Elimina datos procesados, modelos y figuras
clean:
	del /Q data\processed\*.csv 2>nul || true
	del /Q models\*.joblib 2>nul || true
	del /Q reports\figures\*.png 2>nul || true

## Muestra los targets disponibles
help:
	@echo ========================================
	@echo   Pipeline GSHS 2013 El Salvador
	@echo   FEATURE_TYPE=$(FEATURE_TYPE) (Q o QN)
	@echo ========================================
	@echo.
	@echo Uso: make [target] FEATURE_TYPE=Q
	@echo.
	@echo Targets:
	@echo   data      - Limpia y genera splits
	@echo   features  - Muestra feature set seleccionado
	@echo   train     - Entrena todos los modelos (reg + clf)
	@echo   train-reg - Solo modelos de regresion
	@echo   train-clf - Solo modelos de clasificacion
	@echo   evaluate  - Evalua modelos en test set + figuras
	@echo   test      - Inferencia final sobre test
	@echo   all       - Pipeline completo
	@echo   clean     - Elimina datos procesados y modelos
	@echo   help      - Muestra esta ayuda
