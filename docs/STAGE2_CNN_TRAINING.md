# Stage 2 CNN Training Commands

These commands reproduce the verified Stage 2 CNN training and evaluation flow in WSL2 with the GTX 1650 GPU setup.

Run all commands from the project root after activating the WSL GPU environment:

```bash
cd "/mnt/c/Antony/Christ uni/3rd year/SIC/car-recognition-maintenance-advisor"
source scripts/activate_wsl_gpu.sh
```

## Medium Calibration

EfficientNetB0:

```bash
python -m src.model.train_cnn --architecture efficientnetb0 \
  --data-dir data/car_data/car_data \
  --image-size 160 --batch-size 8 --epochs 1 \
  --smoke-batches 20 --weights imagenet \
  --output-dir "$HOME/artifacts/wsl_gpu_calibration_medium_v2"
```

ResNet50:

```bash
python -m src.model.train_cnn --architecture resnet50 \
  --data-dir data/car_data/car_data \
  --image-size 160 --batch-size 4 --epochs 1 \
  --smoke-batches 20 --weights imagenet \
  --output-dir "$HOME/artifacts/wsl_gpu_calibration_medium_v2"
```

## Full Training

EfficientNetB0:

```bash
python -m src.model.train_cnn --architecture efficientnetb0 \
  --data-dir data/car_data/car_data \
  --image-size 160 --batch-size 8 --epochs 5 \
  --weights imagenet \
  --output-dir "$HOME/artifacts/stage2_full"
```

ResNet50:

```bash
python -m src.model.train_cnn --architecture resnet50 \
  --data-dir data/car_data/car_data \
  --image-size 160 --batch-size 4 --epochs 5 \
  --weights imagenet \
  --output-dir "$HOME/artifacts/stage2_full"
```

## Curves, Comparison, and Evaluation

Training curves:

```bash
python -m src.model.plot_training_curves \
  --log-path "$HOME/artifacts/stage2_full/efficientnetb0/training_log.csv" \
  --output-dir "$HOME/artifacts/stage2_full/efficientnetb0/curves"

python -m src.model.plot_training_curves \
  --log-path "$HOME/artifacts/stage2_full/resnet50/training_log.csv" \
  --output-dir "$HOME/artifacts/stage2_full/resnet50/curves"
```

Validation comparison:

```bash
python -m src.model.compare_runs \
  --efficientnet-dir "$HOME/artifacts/stage2_full/efficientnetb0" \
  --resnet-dir "$HOME/artifacts/stage2_full/resnet50" \
  --output-dir "$HOME/artifacts/stage2_full/comparison"
```

Validation metrics:

```bash
python -m src.model.evaluate_cnn \
  --model-path "$HOME/artifacts/stage2_full/efficientnetb0/best_model.keras" \
  --data-dir data/car_data/car_data \
  --split validation --image-size 160 --batch-size 8 \
  --output-dir "$HOME/artifacts/stage2_full/efficientnetb0/eval_validation"

python -m src.model.evaluate_cnn \
  --model-path "$HOME/artifacts/stage2_full/resnet50/best_model.keras" \
  --data-dir data/car_data/car_data \
  --split validation --image-size 160 --batch-size 4 \
  --output-dir "$HOME/artifacts/stage2_full/resnet50/eval_validation"
```

Held-out test evaluation for the selected winner only:

```bash
python -m src.model.evaluate_cnn \
  --model-path "$HOME/artifacts/stage2_full/efficientnetb0/best_model.keras" \
  --data-dir data/car_data/car_data \
  --split test --image-size 160 --batch-size 8 \
  --output-dir "$HOME/artifacts/stage2_full/efficientnetb0/eval_test"
```

## Verified Results

EfficientNetB0 validation:

```text
accuracy=0.6468058968058968
top_5_accuracy=0.8372235872235873
macro_precision=0.2764573799890322
macro_recall=0.19982607844695993
macro_f1=0.2196282979216144
```

ResNet50 validation:

```text
accuracy=0.5608108108108109
top_5_accuracy=0.7874692874692875
macro_precision=0.2938748111875461
macro_recall=0.18413878491520522
macro_f1=0.2116374471964463
```

Selected winner: EfficientNetB0 by validation top-1 accuracy.

EfficientNetB0 held-out test:

```text
accuracy=0.3528168138291257
top_5_accuracy=0.6512871533391369
macro_precision=0.45459300697012256
macro_recall=0.35508250328337215
macro_f1=0.3466540373708492
```
