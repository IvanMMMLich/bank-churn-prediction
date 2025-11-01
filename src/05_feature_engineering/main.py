"""
ЭТАП 5: Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

# Пути
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'

print("=" * 70)
print("ЭТАП 5: Feature Engineering")
print("=" * 70)


def main():
    """Главная функция этапа 5"""
    
    # Загрузка данных
    df_train = pd.read_csv(DATA_DIR / 'train.csv')
    df_test = pd.read_csv(DATA_DIR / 'test.csv')
    
    print(f"\n📊 Train: {df_train.shape[0]:,} строк, {df_train.shape[1]} столбцов")
    print(f"📊 Test:  {df_test.shape[0]:,} строк, {df_test.shape[1]} столбцов")
    
    # Сохраняем id для test (нужен для submission)
    test_ids = df_test['id'].copy()
    
    # 1. УДАЛЕНИЕ НЕНУЖНЫХ ПРИЗНАКОВ
    print("\n" + "=" * 70)
    print(" 1. УДАЛЕНИЕ НЕНУЖНЫХ ПРИЗНАКОВ")
    print("=" * 70)
    
    cols_to_drop = ['id', 'CustomerId', 'Surname']
    
    print(f"\nУдаляем: {', '.join(cols_to_drop)}")
    print("Причина: не несут информации для предсказания")
    
    df_train = df_train.drop(cols_to_drop, axis=1)
    df_test = df_test.drop(cols_to_drop, axis=1)
    
    print(f"\n✅ Train: {df_train.shape[1]} признаков")
    print(f"✅ Test:  {df_test.shape[1]} признаков")
    
    # 2. СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ
    print("\n" + "=" * 70)
    print("✨ 2. СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ")
    print("=" * 70)
    
    # 2.1 Age groups
    print("\n2.1 Группы по возрасту (Age Groups)")
    
    def create_age_group(age):
        if age < 30:
            return 'young'
        elif age < 50:
            return 'middle'
        else:
            return 'senior'
    
    df_train['age_group'] = df_train['Age'].apply(create_age_group)
    df_test['age_group'] = df_test['Age'].apply(create_age_group)
    
    print("  young:  < 30 лет")
    print("  middle: 30-50 лет")
    print("  senior: 50+ лет")
    print(f"  ✅ Создан признак: age_group")
    
    # 2.2 Zero balance
    print("\n2.2 Нулевой баланс (Zero Balance)")
    
    df_train['has_zero_balance'] = (df_train['Balance'] == 0).astype(int)
    df_test['has_zero_balance'] = (df_test['Balance'] == 0).astype(int)
    
    zero_count = df_train['has_zero_balance'].sum()
    zero_pct = (zero_count / len(df_train)) * 100
    print(f"  Клиентов с нулевым балансом: {zero_count:,} ({zero_pct:.1f}%)")
    print(f"  ✅ Создан признак: has_zero_balance")
    
    # 2.3 Balance per product
    print("\n2.3 Баланс на продукт (Balance per Product)")
    
    df_train['balance_per_product'] = df_train['Balance'] / (df_train['NumOfProducts'] + 1)
    df_test['balance_per_product'] = df_test['Balance'] / (df_test['NumOfProducts'] + 1)
    
    print(f"  Средний баланс на продукт: {df_train['balance_per_product'].mean():,.0f}")
    print(f"  ✅ Создан признак: balance_per_product")
    
    # 2.4 Tenure-Age ratio
    print("\n2.4 Стаж относительно возраста (Tenure/Age)")
    
    df_train['tenure_age_ratio'] = df_train['Tenure'] / df_train['Age']
    df_test['tenure_age_ratio'] = df_test['Tenure'] / df_test['Age']
    
    print(f"  Средний ratio: {df_train['tenure_age_ratio'].mean():.3f}")
    print(f"  ✅ Создан признак: tenure_age_ratio")
    
    print(f"\n Создано новых признаков: 4")
    print(f"📊 Train теперь: {df_train.shape[1]} признаков")

# 3. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
    print("\n" + "=" * 70)
    print("🔤 3. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
    print("=" * 70)
    
    # 3.1 Geography - One-Hot Encoding
    print("\n3.1 Geography (One-Hot Encoding)")
    
    geography_dummies_train = pd.get_dummies(df_train['Geography'], prefix='Geography', drop_first=True)
    geography_dummies_test = pd.get_dummies(df_test['Geography'], prefix='Geography', drop_first=True)
    
    df_train = pd.concat([df_train, geography_dummies_train], axis=1)
    df_test = pd.concat([df_test, geography_dummies_test], axis=1)
    
    df_train = df_train.drop('Geography', axis=1)
    df_test = df_test.drop('Geography', axis=1)
    
    print(f"  Создано: {', '.join(geography_dummies_train.columns)}")
    print(f"  ✅ Geography закодирован")
    
    # 3.2 Gender - Label Encoding
    print("\n3.2 Gender (Label Encoding)")
    
    gender_map = {'Female': 0, 'Male': 1}
    df_train['Gender'] = df_train['Gender'].map(gender_map)
    df_test['Gender'] = df_test['Gender'].map(gender_map)
    
    print(f"  Female → 0, Male → 1")
    print(f"  ✅ Gender закодирован")
    
    # 3.3 age_group - One-Hot Encoding
    print("\n3.3 age_group (One-Hot Encoding)")
    
    age_group_dummies_train = pd.get_dummies(df_train['age_group'], prefix='age_group', drop_first=True)
    age_group_dummies_test = pd.get_dummies(df_test['age_group'], prefix='age_group', drop_first=True)
    
    df_train = pd.concat([df_train, age_group_dummies_train], axis=1)
    df_test = pd.concat([df_test, age_group_dummies_test], axis=1)
    
    df_train = df_train.drop('age_group', axis=1)
    df_test = df_test.drop('age_group', axis=1)
    
    print(f"  Создано: {', '.join(age_group_dummies_train.columns)}")
    print(f"  ✅ age_group закодирован")
    
    print(f"\n📊 Train после кодирования: {df_train.shape[1]} признаков")
    print(f"📊 Test после кодирования: {df_test.shape[1]} признаков")

# 4. МАСШТАБИРОВАНИЕ ЧИСЛОВЫХ ПРИЗНАКОВ
    print("\n" + "=" * 70)
    print("⚖️  4. МАСШТАБИРОВАНИЕ ЧИСЛОВЫХ ПРИЗНАКОВ")
    print("=" * 70)
    
    # Отделяем таргет
    y_train = df_train['Exited'].copy()
    X_train = df_train.drop('Exited', axis=1)
    X_test = df_test.copy()
    
    print(f"\n✅ X_train: {X_train.shape}")
    print(f"✅ y_train: {y_train.shape}")
    print(f"✅ X_test:  {X_test.shape}")
    
    # Числовые признаки для масштабирования
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                       'NumOfProducts', 'EstimatedSalary',
                       'balance_per_product', 'tenure_age_ratio']
    
    print(f"\nМасштабируем: {', '.join(numeric_features)}")
    
    # StandardScaler
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    print(f"✅ Применён StandardScaler (mean=0, std=1)")
    
    # 5. СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ
    print("\n" + "=" * 70)
    print("💾 5. СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ")
    print("=" * 70)
    
    # Создаём папку
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем данные
    X_train.to_csv(PROCESSED_DIR / 'X_train.csv', index=False)
    y_train.to_csv(PROCESSED_DIR / 'y_train.csv', index=False)
    X_test.to_csv(PROCESSED_DIR / 'X_test.csv', index=False)
    
    # Сохраняем test_ids для submission
    test_ids.to_csv(PROCESSED_DIR / 'test_ids.csv', index=False)
    
    # Сохраняем scaler
    with open(PROCESSED_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ Сохранено:")
    print(f"  • X_train.csv:  {X_train.shape}")
    print(f"  • y_train.csv:  {y_train.shape}")
    print(f"  • X_test.csv:   {X_test.shape}")
    print(f"  • test_ids.csv: {len(test_ids)} строк")
    print(f"  • scaler.pkl:   StandardScaler объект")
    
    print(f"\n📁 Расположение: {PROCESSED_DIR}")
    
    # ФИНАЛ
    print("\n" + "=" * 70)
    print("🎉 ЭТАП 5 ЗАВЕРШЁН!")
    print("=" * 70)
    print("\n✨ Что сделано:")
    print("  1. Удалены: id, CustomerId, Surname")
    print("  2. Созданы новые признаки: 4")
    print("  3. Закодированы категориальные (One-Hot, Label)")
    print("  4. Масштабированы числовые (StandardScaler)")
    print("  5. Сохранены обработанные данные")
    print(f"\n📊 Итого признаков для модели: {X_train.shape[1]}")
    print(f"   Числовых: {len(numeric_features)}")
    print(f"   Бинарных: {X_train.shape[1] - len(numeric_features)}")
    print("\n🚀 Следующий шаг:")
    print("  cd ../06_modeling")
    print("  python main.py")
    print("\n💡 Там будем обучать модели и делать предсказания!")


if __name__ == "__main__":
    main()
