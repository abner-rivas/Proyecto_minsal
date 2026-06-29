# Detecta el SO para usar el Python del entorno virtual adecuado:
#   Windows -> venv/Scripts/python | Linux/macOS -> venv/bin/python
# En Windows, GNU Make define automaticamente OS=Windows_NT.
ifeq ($(OS),Windows_NT)
	PYTHON = venv/Scripts/python
else
	PYTHON = venv/bin/python
endif

FEATURE_TYPE ?= QN
REPORT = docs/Reporte_Tecnico_MINSAL_IEEE

.PHONY: all data features train train-reg train-clf evaluate test report report-clean clean help

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

## Compila el reporte LaTeX a PDF en docs/ (requiere MiKTeX + latexmk)
report:
	latexmk -pdf -interaction=nonstopmode -cd $(REPORT).tex

## Elimina los archivos auxiliares de LaTeX (conserva el PDF)
report-clean:
	latexmk -c -cd $(REPORT).tex

## Elimina datos procesados, modelos y figuras (portable via Python)
clean:
	$(PYTHON) -c "import glob, os; [os.remove(f) for p in ('data/processed/*.csv','models/*.joblib','reports/figures/*.png') for f in glob.glob(p)]"

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
	@echo   report    - Compila el reporte LaTeX a PDF en docs/
	@echo   report-clean - Borra auxiliares LaTeX (conserva PDF)
	@echo   clean     - Elimina datos procesados y modelos
	@echo   help      - Muestra esta ayuda
