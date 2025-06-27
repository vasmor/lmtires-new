import pandas as pd
import sys

if len(sys.argv) < 2:
    print('Usage: python analyze_losses_summary.py <losses_metrics.csv>')
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
tags = sorted(df['tag'].unique())

print(f'Файл: {csv_path}')
print('Доступные лоссы/метрики:', tags)

for tag in tags:
    vals = df[df['tag'] == tag]['value'].values
    print(f'\n{tag}:')
    print('  min=%.5f, max=%.5f, mean=%.5f, last=%.5f' % (vals.min(), vals.max(), vals.mean(), vals[-1])) 