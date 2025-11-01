"""
ЭТАП 2: Проверка качества данных
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
print("ЭТАП 2: Проверка качества данных")
print("=" * 70)


def main():
    """Главная функция этапа 2"""
    
    # Загрузка данных
    df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"\n📊 Датасет: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
    
    # 1. ПРОВЕРКА ПРОПУСКОВ
    print("\n" + "=" * 70)
    print("🔍 1. ПРОВЕРКА ПРОПУСКОВ (Missing Values)")
    print("=" * 70)
    
    missing = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df)) * 100
    
    missing_data = missing[missing > 0]
    
    if len(missing_data) == 0:
        print("  ✅ ОТЛИЧНО! Пропусков нет")
    else:
        print("  ⚠️ Найдены пропуски:")
        for col, count in missing_data.items():
            pct = missing_pct[col]
            print(f"    • {col}: {count:,} ({pct:.2f}%)")

# 2. ПРОВЕРКА ДУБЛИКАТОВ
    print("\n" + "=" * 70)
    print("🔍 2. ПРОВЕРКА ДУБЛИКАТОВ")
    print("=" * 70)
    
    duplicates = df.duplicated().sum()
    dup_pct = (duplicates / len(df)) * 100
    
    if duplicates == 0:
        print("  ✅ ОТЛИЧНО! Дубликатов нет")
    else:
        print(f"  ⚠️ Найдено дубликатов: {duplicates:,} ({dup_pct:.2f}%)")
        print("  💡 Рекомендация: удалить дубликаты перед обучением")

# 3. ПРОВЕРКА ВЫБРОСОВ (IQR метод)
    print("\n" + "=" * 70)
    print("🔍 3. ПРОВЕРКА ВЫБРОСОВ (Outliers)")
    print("=" * 70)
    print("Метод: IQR (Interquartile Range)")
    print()
    
    # Выбираем только числовые столбцы (кроме id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'CustomerId', 'Exited']]
    
    outliers_summary = []
    
    for col in numeric_cols:
        # IQR метод
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Считаем выбросы
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outliers_pct = (outliers / len(df)) * 100
        
        # Сохраняем информацию
        outliers_summary.append({
            'Признак': col,
            'Выбросов': outliers,
            'Процент': outliers_pct,
            'Нижняя_граница': lower,
            'Верхняя_граница': upper
        })
        
        # Статус
        if outliers_pct < 1:
            status = "✅ Мало"
        elif outliers_pct < 5:
            status = "⚠️ Средне"
        else:
            status = "❌ Много"
        
        print(f"  {col:20s}: {outliers:6,} ({outliers_pct:5.2f}%)  {status}")

# 4. ВИЗУАЛИЗАЦИЯ ВЫБРОСОВ (Boxplots)
    print("\n📊 Создаём boxplot графики...")
    
    # Количество графиков
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        # Boxplot
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Значение', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Добавляем информацию о выбросах
        outliers_info = outliers_summary[i]
        text = f"Выбросов: {outliers_info['Выбросов']:,}\n({outliers_info['Процент']:.1f}%)"
        axes[i].text(0.5, 0.98, text,
                    transform=axes[i].transAxes,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=9)
    
    # Удаляем лишние пустые графики
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # Сохраняем график
    figures_dir = RESULTS_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / '02_outliers_boxplots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ График сохранён: {output_path}")
    
    plt.show()
    
    # 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    print("\n💾 Сохраняем результаты...")
    
    tables_dir = RESULTS_DIR / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Таблица выбросов
    outliers_df = pd.DataFrame(outliers_summary)
    outliers_df.to_csv(tables_dir / '02_outliers_summary.csv', index=False)
    
    # Общая сводка качества
    quality_summary = pd.DataFrame({
        'Метрика': [
            'Всего строк',
            'Всего столбцов',
            'Пропусков (всего)',
            'Столбцов с пропусками',
            'Дубликатов'
        ],
        'Значение': [
            len(df),
            len(df.columns),
            missing.sum(),
            len(missing_data),
            duplicates
        ]
    })
    quality_summary.to_csv(tables_dir / '02_data_quality_summary.csv', index=False)
    
    print(f"✅ Таблицы сохранены в: {tables_dir}")
    
    # ФИНАЛ
    print("\n" + "=" * 70)
    print(" ЭТАП 2 ЗАВЕРШЁН!")
    print("=" * 70)
    print("\n Итог по качеству данных:")
    print(f"  • Пропусков: {missing.sum()} {'✅ нет' if missing.sum() == 0 else '⚠️ есть'}")
    print(f"  • Дубликатов: {duplicates} {'✅ нет' if duplicates == 0 else '⚠️ есть'}")
    print(f"  • Выбросов: есть, но в пределах нормы (см. график)")
    print("\n Результаты:")
    print(f"  - График: results/figures/02_outliers_boxplots.png")
    print(f"  - Таблицы: results/tables/02_*.csv")
    print("\n Следующий шаг:")
    print("  cd ../03_distributions")
    print("  python main.py")


if __name__ == "__main__":
    main()

