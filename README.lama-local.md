# Документация по запуску и диагностике LaMa (inpainting) на кастомных данных

## 1. Установка и подготовка окружения

1. **Клонируйте репозиторий LaMa (или используйте свою папку lama-local):**
   ```sh
   git clone https://github.com/advimman/lama.git lama-local
   ```

2. **Создайте и активируйте виртуальное окружение (Python 3.8–3.10):**
   ```sh
   python -m venv venv39
   venv39\\Scripts\\activate
   ```

3. **Установите зависимости:**
   ```sh
   pip install -r lama-local/requirements.txt
   ```
   Если файла requirements.txt нет — используйте список из README оригинального репозитория.

4. **Проверьте наличие CUDA и torch:**
   ```sh
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 2. Подготовка данных

### Для инференса на автогенерируемых масках (online-режим):

- В папке с изображениями (`indir`) должны лежать только нужные изображения (например, `.png`).
- Маски не нужны — они будут генерироваться на лету.
- Для sanity check создайте отдельную папку с 2–3 изображениями.

### Для инференса по валидационному датасету с реальными масками:

- В папке должны быть пары файлов: `имя.png` и `имя_mask.png`.
- Расширение указывается через параметр `img_suffix`.

---

## 3. Конфиг для инференса (пример для online-режима)

**Файл:** `lama-local/configs/prediction/sanity_check.yaml`
```yaml
indir: C:/ai-product-image-project/downloads/lama_dataset/sanity
outdir: sanity_check
model:
  path: C:/ai-product-image-project/lama-local/outputs/2025-06-25/21-15-36
  checkpoint: last.ckpt
dataset:
  kind: online
  img_suffix: .png
  mask_generator_kind: mixed
  mask_gen_kwargs:
    irregular_proba: 0.5
    irregular_kwargs:
      min_times: 2
      max_times: 20
      max_width: 80
      max_angle: 6
      max_len: 200
    box_proba: 0.3
    box_kwargs:
      margin: 0
      bbox_min_size: 10
      bbox_max_size: 200
      max_times: 5
      min_times: 1
    ellipse_proba: 0.2
    ellipse_kwargs:
      min_axis: 10
      max_axis: 120
      min_times: 1
      max_times: 3
    squares_proba: 0
    segm_proba: 0
out_key: inpainted
```

---

## 4. Конфиг для инференса по валидационному датасету с масками

**Файл:** `lama-local/configs/prediction/val.yaml`
```yaml
indir: C:/ai-product-image-project/valid-dataset/val_images
outdir: val_inpainted
model:
  path: C:/ai-product-image-project/lama-local/outputs/2025-06-25/21-15-36
  checkpoint: last.ckpt
dataset:
  img_suffix: .png
out_key: inpainted
```
- Здесь не указывается `kind` — используется стандартный датасет, который ищет пары `имя.png` и `имя_mask.png`.

---

## 5. Команда запуска инференса

**Общий шаблон:**
```sh
$env:PYTHONPATH=".."; ..\..\\venv39\\Scripts\\python.exe predict.py --config-name=<имя_конфига>.yaml --config-path=../configs/prediction
```

**Пример для sanity check:**
```sh
$env:PYTHONPATH=".."; ..\..\\venv39\\Scripts\\python.exe predict.py --config-name=sanity_check.yaml --config-path=../configs/prediction
```

**Пример для валидационного датасета:**
```sh
$env:PYTHONPATH=".."; ..\..\\venv39\\Scripts\\python.exe predict.py --config-name=val.yaml --config-path=../configs/prediction
```

---

## 6. Типовые ошибки и их решения

- **Ошибка: `TypeError: __init__() got an unexpected keyword argument 'mask_suffix'`**
  - Удалите параметр `mask_suffix` из конфига.

- **Ошибка: `AttributeError: 'InpaintingEvalOnlineDataset' object has no attribute 'mask_filenames'`**
  - В predict.py замените все обращения к `dataset.mask_filenames` на `dataset.img_filenames` для online-режима.

- **Ошибка: `Key 'out_key' is not in struct`**
  - Проверьте, что параметр `out_key` находится на верхнем уровне конфига, а не внутри `dataset`.

- **Ошибка: не обрабатываются только нужные изображения**
  - Для online-режима скопируйте нужные изображения в отдельную папку и укажите её как `indir`.

- **Ошибка путей (Windows):**
  - Используйте абсолютные пути с прямыми слэшами (`/`), либо двойные обратные (`\\`).
  - Убедитесь, что все пути в конфиге существуют.

---

## 7. Пример успешного запуска sanity check

```sh
$env:PYTHONPATH=".."; ..\..\\venv39\\Scripts\\python.exe predict.py --config-name=sanity_check.yaml --config-path=../configs/prediction
```
**Вывод:**
```
[DEBUG] indir: C:/ai-product-image-project/downloads/lama_dataset/sanity
[DEBUG] outdir: sanity_check
...
[DEBUG] Найдено изображений: 2
[DEBUG] Примеры img_filenames: [...]
100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  5.53it/s]
[DEBUG] --- Инференс завершён ---
```
Результаты — в папке `sanity_check`.

---

## 8. Пример успешного запуска инференса по валидационному датасету

```sh
$env:PYTHONPATH=".."; ..\..\\venv39\\Scripts\\python.exe predict.py --config-name=val.yaml --config-path=../configs/prediction
```
**Вывод:**
```
[DEBUG] indir: C:/ai-product-image-project/valid-dataset/val_images
[DEBUG] outdir: val_inpainted
...
[DEBUG] Найдено изображений: 100
[DEBUG] Примеры mask_filenames: [...]
[DEBUG] Примеры img_filenames: [...]
100%|████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00,  5.53it/s]
[DEBUG] --- Инференс завершён ---
```
Результаты — в папке `val_inpainted`.

---

## 9. Советы

- Всегда проверяйте структуру и отступы в yaml-файлах.
- Для online-режима используйте только параметры, поддерживаемые генератором масок.
- Для запуска на подмножестве изображений — используйте отдельную папку.
- После изменений в predict.py не забывайте сохранять файл и перезапускать инференс.

---

**Если возникнут новые ошибки — внимательно читайте traceback: почти всегда причина в структуре конфига или несовпадении параметров с реальным кодом.**

## Стратегия поэтапного дообучения LaMa (рекомендации на основе анализа логов)

### Этап 1: "Снижение сложности, мягкий adversarial"
- epochs: 20
- Снижена сложность масок (уменьшены max_len, max_times, max_axis, max_width)
- irregular_proba: 0.3 (было 0.5)
- lr дискриминатора: 0.00005 (в 2 раза меньше)
- adversarial loss: вес 3 (было 10)
- perceptual loss: вес 0.05
- l1 loss: вес 1
- box_proba: 0.5 (было 0.3)
- ellipse_proba: 0.2
- Файл: `your_config-1.yaml`

### Этап 2: "Разгон генератора, простые маски"
- epochs: 20
- Отключён adversarial loss (вес 0)
- irregular_proba: 0.0 (только box/ellipse)
- box_proba: 0.7, ellipse_proba: 0.3
- perceptual loss: вес 0.01
- l1 loss: вес 2
- lr дискриминатора: 0.00005
- Файл: `your_config-2.yaml`

### Этап 3: "Плавное возвращение adversarial, усложнение масок"
- epochs: 20
- adversarial loss: вес 1
- irregular_proba: 0.2, max_len: 60, max_times: 5
- box_proba: 0.5, ellipse_proba: 0.3
- perceptual loss: вес 0.02
- l1 loss: вес 1.5
- lr дискриминатора: 0.00005
- Файл: `your_config-3.yaml`

### Этап 4: "Почти полный возврат к изначальной сложности, мягкий adversarial"
- epochs: 20
- adversarial loss: вес 5
- irregular_proba: 0.4, max_len: 120, max_times: 10
- box_proba: 0.4, ellipse_proba: 0.2
- perceptual loss: вес 0.05
- l1 loss: вес 1
- lr дискриминатора: 0.00005
- Файл: `your_config-4.yaml`

---

**Рекомендация:**
- Запускайте этапы последовательно, начиная с первого.
- После каждого этапа анализируйте динамику лоссов и качество восстановления.
- При необходимости добавляйте этапы с постепенным возвращением adversarial loss и усложнением масок.

**Файлы конфигов для этапов:**
- `lama-local/configs/training/your_config-1.yaml`
- `lama-local/configs/training/your_config-2.yaml`
- `lama-local/configs/training/your_config-3.yaml`
- `lama-local/configs/training/your_config-4.yaml`

---

Если потребуется добавить новые этапы — пишите, и стратегия будет дополнена!

## Оценка метрик качества inpainting

Для автоматического расчёта метрик PSNR, SSIM, MSE, LPIPS между оригиналами и inpainted-результатами используйте скрипт:

```
python lama-local/eval_inpaint_metrics.py --originals_dir <путь_к_оригиналам> --inpainted_dir <путь_к_inpainted> --out_csv <имя_выходного_csv>
```

- `--originals_dir` — папка с оригинальными изображениями
- `--inpainted_dir` — папка с inpainted-результатами
- `--out_csv` — имя выходного CSV-файла (по умолчанию metrics.csv)
- `--device` — cuda или cpu (по умолчанию cuda)

**Результат:**
- CSV-файл с метриками по каждому изображению и строкой средних значений.
- Поддерживаются любые расширения изображений, пары ищутся по имени файла (без расширения).

---

## Отладочный режим (DEBUG)

В некоторых скриптах (например, `bin/predict.py`, `analyze_tb_logs.py`) реализован отладочный режим:

- Для включения подробного отладочного вывода установите в начале файла:
  ```python
  DEBUG = True
  ```
- По умолчанию DEBUG = False (отладочный вывод отключён).

**Что даёт DEBUG:**
- Подробные print-логи о путях, параметрах, примерах файлов, статусе инференса, ошибках, путях к логам и графикам.
- В `predict.py` дополнительно активируется обработка сигнала для дампа traceback (на Linux).
- Помогает при диагностике проблем с путями, конфигами, структурой данных, ошибками инференса.

**Рекомендация:**
- Включайте DEBUG только при необходимости диагностики или отладки. В продуктиве держите DEBUG = False.

--- 