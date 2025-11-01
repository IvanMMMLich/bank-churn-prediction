"""
ЭТАП 1: Загрузка и первичный анализ данных
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройки для красивых графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Пути к данным
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results'

print("=" * 70)
print("ЭТАП 1: Загрузка и первичный анализ данных")
print("=" * 70)

def main():
    """Главная функция этапа 1"""
    
    # 1. Загрузка данных
    print("\nЗагружаем данные...")
    df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"✅ Загружено: {df.shape[0]:,} строк, {df.shape[1]} столбцов")

# 2. Первый взгляд на данные
    print("\nПервые 5 строк:")
    print(df.head())
    
    print("\nНазвания столбцов:")
    print(df.columns.tolist())
    
    # 3. Информация о типах данных
    print("\nИнформация о данных:")
    df.info()
    
    # 4. Статистика по числовым признакам
    print("\nСтатистика:")
    print(df.describe())

# 5. Анализ целевой переменной (Exited)
    print("\n🎯 Анализ целевой переменной 'Exited':")
    
    target_counts = df['Exited'].value_counts()
    target_pct = df['Exited'].value_counts(normalize=True) * 100
    
    print(f"\nРаспределение:")
    print(f"  Остались (0): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
    print(f"  Ушли (1):     {target_counts[1]:,} ({target_pct[1]:.2f}%)")
    
    ratio = target_pct[0] / target_pct[1]
    print(f"\nСоотношение: {ratio:.2f}:1")
    
    # Оценка дисбаланса
    if target_pct[1] < 15:
        print("  ⚠️ СИЛЬНЫЙ дисбаланс классов!")
    elif target_pct[1] < 30:
        print("  ⚠️ Средний дисбаланс — нужен class_weight при обучении")
    else:
        print("  ✅ Дисбаланс приемлемый")

# 6. Визуализация целевой переменной
    print("\n📊 Создаём графики...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Столбчатая диаграмма
    axes[0].bar(['Остались (0)', 'Ушли (1)'], 
                target_counts.values, 
                color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Распределение клиентов', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Количество', fontsize=12)
    axes[0].set_xlabel('Целевая переменная', fontsize=12)
    
    # Добавляем подписи на столбцы
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + 1000, f'{v:,}', ha='center', va='bottom', fontsize=11)
    
    # График 2: Круговая диаграмма
    axes[1].pie(target_counts.values, 
                labels=['Остались (0)', 'Ушли (1)'],
                autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c'],
                startangle=90,
                textprops={'fontsize': 11})
    axes[1].set_title('Процентное соотношение', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Сохраняем график
    figures_dir = RESULTS_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / '01_target_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ График сохранён: {output_path}")
    
    plt.show()

# 7. Сохраняем статистику в CSV
    print("\n💾 Сохраняем результаты...")
    
    tables_dir = RESULTS_DIR / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Базовая статистика
    df.describe().to_csv(tables_dir / '01_basic_statistics.csv')
    
    # Распределение таргета
    target_summary = pd.DataFrame({
        'Класс': ['Остались (0)', 'Ушли (1)'],
        'Количество': target_counts.values,
        'Процент': target_pct.values
    })
    target_summary.to_csv(tables_dir / '01_target_distribution.csv', index=False)
    
    print(f"✅ Таблицы сохранены в: {tables_dir}")
    
    # Финал
    print("\n" + "=" * 70)
    print("🎉 ЭТАП 1 ЗАВЕРШЁН!")
    print("=" * 70)
    print("\n📁 Результаты:")
    print(f"  - График: results/figures/01_target_distribution.png")
    print(f"  - Таблицы: results/tables/01_*.csv")
    print("\n🚀 Следующий шаг:")
    print("  cd ../02_data_quality")
    print("  python main.py")


if __name__ == "__main__":
    main()

    