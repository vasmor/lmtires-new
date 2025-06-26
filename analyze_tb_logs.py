import os
import sys
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import csv

DEBUG = False  # Включить отладочный вывод: DEBUG = True

# Путь к директории с логами
if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    log_dir = os.path.join(os.path.dirname(__file__), 'outputs/2025-06-25/21-15-36/tb_logs/21-15-36/version_0')

def main():
    # Найти events-файл
    files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not files:
        if DEBUG:
            print('Не найден events.out.tfevents.* в', log_dir)
        return
    event_file = os.path.join(log_dir, files[0])
    if DEBUG:
        print('Используется лог:', event_file)

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags()['scalars']
    if DEBUG:
        print('Доступные теги:', tags)

    # Экспортируем все значения в CSV
    csv_path = os.path.join(log_dir, 'losses_metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['tag', 'step', 'value'])
        for tag in tags:
            events = ea.Scalars(tag)
            for e in events:
                writer.writerow([tag, e.step, e.value])
    print(f'Все значения экспортированы в {csv_path}')

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.figure()
        plt.plot(steps, values, marker='o')
        plt.title(tag)
        plt.xlabel('step')
        plt.ylabel('value')
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(log_dir, f'{tag.replace("/", "_")}.png')
        plt.savefig(out_path)
        if DEBUG:
            print(f'График {tag} сохранён в {out_path}')
        plt.close()

if __name__ == '__main__':
    main() 