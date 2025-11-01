"""
ЭТАП 3: Анализ распределений признаков
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройки
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Пути
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results'

print("=" * 70)
print("ЭТАП 3: Анализ распределений признаков")
print("=" * 70)


def main():
    """Главная функция этапа 3"""
    
    # Загрузка данных
    df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"\n📊 Датасет: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
    
    # 1. ЧИСЛОВЫЕ ПРИЗНАКИ - ГИСТОГРАММЫ
    print("\n" + "=" * 70)
    print("📈 1. РАСПРЕДЕЛЕНИЯ ЧИСЛОВЫХ ПРИЗНАКОВ")
    print("=" * 70)
    
    # Выбираем числовые столбцы (кроме id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'CustomerId', 'Exited']]
    
    print(f"\nЧисловых признаков: {len(numeric_cols)}")
    print(f"Список: {', '.join(numeric_cols)}")
    
    # Создаём гистограммы
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    
    print("\n📊 Анализ распределений:")
    
    for i, col in enumerate(numeric_cols):
        # Гистограмма
        axes[i].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Значение', fontsize=10)
        axes[i].set_ylabel('Частота', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Статистика
        mean_val = df[col].mean()
        median_val = df[col].median()
        skew_val = df[col].skew()
        
        # Добавляем вертикальные линии для mean и median
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        axes[i].legend(fontsize=8)
        
        # Определяем тип распределения
        if abs(skew_val) < 0.5:
            dist_type = "✅ Симметричное"
        elif skew_val > 0.5:
            dist_type = "⚠️ Скошено вправо"
        else:
            dist_type = "⚠️ Скошено влево"
        
        print(f"  {col:20s}: Mean={mean_val:8.1f}, Median={median_val:8.1f}, Skew={skew_val:6.2f}  {dist_type}")
    
    # Удаляем лишние графики
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # Сохраняем
    figures_dir = RESULTS_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / '03_numeric_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ График сохранён: {output_path}")
    
    plt.show()
# 2. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
    print("\n" + "=" * 70)
    print("📊 2. РАСПРЕДЕЛЕНИЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
    print("=" * 70)
    
    categorical_cols = ['Geography', 'Gender']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, col in enumerate(categorical_cols):
        # Считаем распределение
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        
        # Столбчатая диаграмма
        axes[i].bar(counts.index, counts.values, color='steelblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(col, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Категория', fontsize=12)
        axes[i].set_ylabel('Количество', fontsize=12)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Добавляем подписи с процентами
        for j, (category, count) in enumerate(counts.items()):
            pct = percentages[category]
            axes[i].text(j, count + 1000, f'{count:,}\n({pct:.1f}%)', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Выводим в консоль
        print(f"\n{col}:")
        for category, count in counts.items():
            pct = percentages[category]
            print(f"  {category:15s}: {count:6,} ({pct:5.1f}%)")
    
    plt.tight_layout()
    
    # Сохраняем
    output_path = figures_dir / '03_categorical_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ График сохранён: {output_path}")
    
    plt.show()

# 3. АНАЛИЗ EXITED ПО КАТЕГОРИЯМ
    print("\n" + "=" * 70)
    print("🎯 3. АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ПО КАТЕГОРИЯМ")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, col in enumerate(categorical_cols):
        # Группируем по категории и считаем процент ушедших
        exit_rate = df.groupby(col)['Exited'].mean() * 100
        
        # Столбчатая диаграмма
        bars = axes[i].bar(exit_rate.index, exit_rate.values, 
                          color=['#e74c3c' if v > 25 else '#3498db' for v in exit_rate.values],
                          edgecolor='black', alpha=0.7)
        
        axes[i].set_title(f'Процент оттока по {col}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Категория', fontsize=12)
        axes[i].set_ylabel('% ушедших клиентов', fontsize=12)
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].axhline(y=21.16, color='red', linestyle='--', linewidth=2, label='Средний отток (21.16%)')
        axes[i].legend()
        
        # Добавляем подписи
        for j, (category, rate) in enumerate(exit_rate.items()):
            axes[i].text(j, rate + 0.5, f'{rate:.1f}%', 
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Выводим в консоль
        print(f"\n{col} - Процент оттока:")
        for category, rate in exit_rate.items():
            status = "❌ Высокий" if rate > 25 else "✅ Нормальный"
            print(f"  {category:15s}: {rate:5.1f}%  {status}")
    
    plt.tight_layout()
    
    # Сохраняем
    output_path = figures_dir / '03_exit_rate_by_category.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ График сохранён: {output_path}")
    
    plt.show()

    # ФИНАЛ
    print("\n" + "=" * 70)
    print("🎉 ЭТАП 3 ЗАВЕРШЁН!")
    print("=" * 70)
    print("\n📊 Ключевые находки:")
    print("  1. Большинство признаков скошены (не нормальные)")
    print("  2. Balance имеет много нулей (клиенты без денег)")
    print("  3. Germany — высокий отток (32% vs 16-17%)")
    print("  4. Gender — отток примерно одинаковый")
    print("\n📁 Результаты:")
    print(f"  - Числовые: results/figures/03_numeric_distributions.png")
    print(f"  - Категориальные: results/figures/03_categorical_distributions.png")
    print(f"  - Отток по категориям: results/figures/03_exit_rate_by_category.png")
    print("\n🚀 Следующий шаг:")
    print("  cd ../04_correlations")
    print("  python main.py")


if __name__ == "__main__":
    main()

    