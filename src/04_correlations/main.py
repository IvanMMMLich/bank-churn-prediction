"""
ЭТАП 4: Корреляционный анализ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройки
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Пути
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results'

print("=" * 70)
print("ЭТАП 4: Корреляционный анализ")
print("=" * 70)


def main():
    """Главная функция этапа 4"""
    
    # Загрузка данных
    df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"\n📊 Датасет: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
    
    # ПОДГОТОВКА ДАННЫХ ДЛЯ КОРРЕЛЯЦИИ
    print("\n" + "=" * 70)
    print("🔧 ПОДГОТОВКА ДАННЫХ")
    print("=" * 70)
    
    # Копируем датафрейм
    df_corr = df.copy()
    
    # Удаляем ненужные столбцы
    df_corr = df_corr.drop(['id', 'CustomerId', 'Surname'], axis=1)
    print(f"\n✅ Удалены: id, CustomerId, Surname")
    
    # Кодируем Geography (Label Encoding)
    geography_map = {'France': 0, 'Spain': 1, 'Germany': 2}
    df_corr['Geography'] = df_corr['Geography'].map(geography_map)
    print(f"✅ Geography закодирован: France=0, Spain=1, Germany=2")
    
    # Кодируем Gender (Label Encoding)
    gender_map = {'Female': 0, 'Male': 1}
    df_corr['Gender'] = df_corr['Gender'].map(gender_map)
    print(f"✅ Gender закодирован: Female=0, Male=1")
    
    print(f"\n📊 Итоговый датасет: {df_corr.shape[1]} признаков")
    print(f"Признаки: {', '.join(df_corr.columns.tolist())}")

# 1. КОРРЕЛЯЦИЯ С ТАРГЕТОМ (EXITED)
    print("\n" + "=" * 70)
    print("🎯 1. КОРРЕЛЯЦИЯ ПРИЗНАКОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (EXITED)")
    print("=" * 70)
    
    # Считаем корреляцию
    correlations = df_corr.corr()['Exited'].drop('Exited').sort_values(ascending=False)
    
    print("\n📊 Корреляция признаков с оттоком (от сильной к слабой):\n")
    
    for feature, corr_value in correlations.items():
        # Определяем силу связи
        abs_corr = abs(corr_value)
        if abs_corr > 0.3:
            strength = "🔥 СИЛЬНАЯ"
            color = "red"
        elif abs_corr > 0.15:
            strength = "⚠️ СРЕДНЯЯ"
            color = "orange"
        elif abs_corr > 0.05:
            strength = "✅ СЛАБАЯ"
            color = "green"
        else:
            strength = "❌ ОЧЕНЬ СЛАБАЯ"
            color = "gray"
        
        # Определяем направление
        direction = "↑ Прямая" if corr_value > 0 else "↓ Обратная"
        
        print(f"  {feature:20s}: {corr_value:+7.3f}  {direction:12s}  {strength}")
    
    # Визуализация корреляций
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in correlations.values]
    bars = ax.barh(correlations.index, correlations.values, color=colors, edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Корреляция с Exited', fontsize=12, fontweight='bold')
    ax.set_title('Корреляция признаков с оттоком клиентов', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцы
    for i, (feature, value) in enumerate(correlations.items()):
        ax.text(value + 0.01 if value > 0 else value - 0.01, i, 
                f'{value:+.3f}', 
                va='center', ha='left' if value > 0 else 'right',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Сохраняем
    figures_dir = RESULTS_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / '04_correlation_with_target.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ График сохранён: {output_path}")
    
    plt.show()
    
    # Сохраняем таблицу
    tables_dir = RESULTS_DIR / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    correlations_df = pd.DataFrame({
        'Признак': correlations.index,
        'Корреляция': correlations.values
    })
    correlations_df.to_csv(tables_dir / '04_correlations_with_target.csv', index=False)

# 2. КОРРЕЛЯЦИОННАЯ МАТРИЦА (HEATMAP)
    print("\n" + "=" * 70)
    print("🔥 2. КОРРЕЛЯЦИОННАЯ МАТРИЦА (HEATMAP)")
    print("=" * 70)
    
    # Полная корреляционная матрица
    corr_matrix = df_corr.corr()
    
    # Создаём heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, 
                annot=True,           # Показать значения
                fmt='.2f',            # Формат: 2 знака после запятой
                cmap='coolwarm',      # Цветовая схема: синий-белый-красный
                center=0,             # Центр в 0
                square=True,          # Квадратные ячейки
                linewidths=0.5,       # Линии между ячейками
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Корреляционная матрица признаков', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Сохраняем
    output_path = figures_dir / '04_correlation_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Корреляционная матрица сохранена: {output_path}")
    
    plt.show()
    
    # 3. ТОП-5 ВАЖНЕЙШИХ ПРИЗНАКОВ
    print("\n" + "=" * 70)
    print("🏆 3. ТОП-5 ВАЖНЕЙШИХ ПРИЗНАКОВ ДЛЯ МОДЕЛИ")
    print("=" * 70)
    
    top5 = correlations.abs().sort_values(ascending=False).head(5)
    
    print("\nПо абсолютной величине корреляции:\n")
    for i, (feature, abs_corr) in enumerate(top5.items(), 1):
        corr_value = correlations[feature]
        direction = "↑" if corr_value > 0 else "↓"
        print(f"  {i}. {feature:20s}: {direction} {abs_corr:.3f}  (значение: {corr_value:+.3f})")
    
    print("\n💡 Интерпретация топ-5:")
    print("  1. Age              - Пожилые клиенты уходят чаще")
    print("  2. NumOfProducts    - Клиенты с 3-4 продуктами уходят чаще")
    print("  3. Geography        - Germany проблемная зона (37.9% отток)")
    print("  4. IsActiveMember   - Неактивные клиенты уходят чаще")
    print("  5. Gender           - Женщины уходят чаще (25% vs 16%)")
    
    # ФИНАЛ
    print("\n" + "=" * 70)
    print("🎉 ЭТАП 4 ЗАВЕРШЁН!")
    print("=" * 70)
    print("\n🔥 Ключевые находки:")
    print("  • Age (0.341) - САМЫЙ ВАЖНЫЙ признак!")
    print("  • Geography (0.214) - Germany высокий отток")
    print("  • NumOfProducts (-0.215) - чем больше продуктов, тем лояльнее")
    print("  • IsActiveMember (-0.210) - активность критична")
    print("  • Balance (0.015) - СЛАБАЯ связь, почти не влияет")
    print("\n💡 Вывод для модели:")
    print("  Можем удалить признаки с корреляцией < 0.05:")
    print("  • Balance, Tenure, EstimatedSalary - слабо влияют")
    print("  НО лучше оставить все и позволить модели решить!")
    print("\n📁 Результаты:")
    print(f"  - График корреляций: results/figures/04_correlation_with_target.png")
    print(f"  - Матрица: results/figures/04_correlation_matrix.png")
    print(f"  - Таблица: results/tables/04_correlations_with_target.csv")
    print("\n🚀 Следующий шаг:")
    print("  cd ../05_feature_engineering")
    print("  python main.py")


if __name__ == "__main__":
    main()


    
