"""
ЭТАП 6: Обучение и оценка моделей
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                            confusion_matrix, classification_report, roc_curve)

# Настройки
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Пути
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
SUBMISSIONS_DIR = PROJECT_ROOT / 'data' / 'submissions'
RESULTS_DIR = PROJECT_ROOT / 'results'

print("=" * 70)
print("ЭТАП 6: Обучение и оценка моделей")
print("=" * 70)


def main():
    """Главная функция этапа 6"""
    
    # 1. ЗАГРУЗКА ОБРАБОТАННЫХ ДАННЫХ
    print("\n" + "=" * 70)
    print("📂 1. ЗАГРУЗКА ОБРАБОТАННЫХ ДАННЫХ")
    print("=" * 70)
    
    X_train_full = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    y_train_full = pd.read_csv(PROCESSED_DIR / 'y_train.csv').values.ravel()
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
    test_ids = pd.read_csv(PROCESSED_DIR / 'test_ids.csv')
    
    print(f"\n✅ X_train: {X_train_full.shape}")
    print(f"✅ y_train: {y_train_full.shape}")
    print(f"✅ X_test:  {X_test.shape}")
    print(f"✅ test_ids: {len(test_ids)}")
    
    print(f"\nПризнаки ({X_train_full.shape[1]}):")
    print(f"{', '.join(X_train_full.columns.tolist())}")
    
    # Проверка баланса классов
    print(f"\nБаланс классов:")
    unique, counts = np.unique(y_train_full, return_counts=True)
    for label, count in zip(unique, counts):
        pct = (count / len(y_train_full)) * 100
        print(f"  Класс {label}: {count:,} ({pct:.2f}%)")
    
    # 2. РАЗДЕЛЕНИЕ НА TRAIN/VALIDATION
    print("\n" + "=" * 70)
    print("✂️  2. РАЗДЕЛЕНИЕ НА TRAIN/VALIDATION")
    print("=" * 70)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2,           # 20% на валидацию
        random_state=42,         # Для воспроизводимости
        stratify=y_train_full    # Сохраняем пропорцию классов
    )
    
    print(f"\n✅ Train: {X_train.shape[0]:,} строк ({X_train.shape[0]/len(X_train_full)*100:.0f}%)")
    print(f"✅ Val:   {X_val.shape[0]:,} строк ({X_val.shape[0]/len(X_train_full)*100:.0f}%)")
    
    print(f"\nБаланс в Train:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        pct = (count / len(y_train)) * 100
        print(f"  Класс {label}: {count:,} ({pct:.2f}%)")

    # 3. МОДЕЛЬ 1: LOGISTIC REGRESSION (BASELINE)
    print("\n" + "=" * 70)
    print("🤖 3. МОДЕЛЬ 1: LOGISTIC REGRESSION (Baseline)")
    print("=" * 70)
    
    print("\nОбучаем Logistic Regression...")
    
    lr_model = LogisticRegression(
        class_weight='balanced',  # Учитываем дисбаланс классов!
        max_iter=1000,
        random_state=42
    )
    
    lr_model.fit(X_train, y_train)
    print("✅ Обучение завершено")
    
    # Предсказания
    y_pred_lr = lr_model.predict(X_val)
    y_pred_proba_lr = lr_model.predict_proba(X_val)[:, 1]
    
    # Метрики
    print("\n📊 Метрики на Validation:")
    
    accuracy_lr = accuracy_score(y_val, y_pred_lr)
    auc_lr = roc_auc_score(y_val, y_pred_proba_lr)
    f1_lr = f1_score(y_val, y_pred_lr)
    
    print(f"  Accuracy:  {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
    print(f"  AUC-ROC:   {auc_lr:.4f}")
    print(f"  F1-Score:  {f1_lr:.4f}")
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_val, y_pred_lr, target_names=['Остались', 'Ушли']))
    
    # 4. МОДЕЛЬ 2: RANDOM FOREST (МОЩНАЯ)
    print("\n" + "=" * 70)
    print("🌳 4. МОДЕЛЬ 2: RANDOM FOREST")
    print("=" * 70)
    
    print("\nОбучаем Random Forest...")
    print("(может занять 1-2 минуты)")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,         # 100 деревьев
        max_depth=10,             # Глубина деревьев
        min_samples_split=20,     # Минимум для разделения
        min_samples_leaf=10,      # Минимум в листе
        class_weight='balanced',  # Учитываем дисбаланс!
        random_state=42,
        n_jobs=-1                 # Используем все ядра CPU
    )
    
    rf_model.fit(X_train, y_train)
    print("✅ Обучение завершено")
    
    # Предсказания
    y_pred_rf = rf_model.predict(X_val)
    y_pred_proba_rf = rf_model.predict_proba(X_val)[:, 1]
    
    # Метрики
    print("\n📊 Метрики на Validation:")
    
    accuracy_rf = accuracy_score(y_val, y_pred_rf)
    auc_rf = roc_auc_score(y_val, y_pred_proba_rf)
    f1_rf = f1_score(y_val, y_pred_rf)
    
    print(f"  Accuracy:  {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
    print(f"  AUC-ROC:   {auc_rf:.4f}")
    print(f"  F1-Score:  {f1_rf:.4f}")
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_val, y_pred_rf, target_names=['Остались', 'Ушли']))

# 5. СРАВНЕНИЕ МОДЕЛЕЙ
    print("\n" + "=" * 70)
    print("⚖️  5. СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Модель': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [accuracy_lr, accuracy_rf],
        'AUC-ROC': [auc_lr, auc_rf],
        'F1-Score': [f1_lr, f1_rf]
    })
    
    print("\n📊 Сравнительная таблица:\n")
    print(comparison.to_string(index=False))
    
    # Определяем лучшую модель
    best_model_name = 'Random Forest' if auc_rf > auc_lr else 'Logistic Regression'
    best_model = rf_model if auc_rf > auc_lr else lr_model
    best_auc = max(auc_rf, auc_lr)
    
    print(f"\n🏆 Лучшая модель: {best_model_name} (AUC-ROC: {best_auc:.4f})")
    
    # 6. ВИЗУАЛИЗАЦИЯ: ROC-CURVE
    print("\n" + "=" * 70)
    print("📈 6. ROC-КРИВАЯ")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ROC для Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(y_val, y_pred_proba_lr)
    ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})', linewidth=2)
    
    # ROC для Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_val, y_pred_proba_rf)
    ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=2)
    
    # Диагональ (случайная модель)
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Сравнение моделей', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Сохраняем
    figures_dir = RESULTS_DIR / 'figures'
    output_path = figures_dir / '06_roc_curve.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ ROC-кривая сохранена: {output_path}")
    
    plt.show()
    
    # 7. FEATURE IMPORTANCE (для Random Forest)
    print("\n" + "=" * 70)
    print("🔍 7. ВАЖНОСТЬ ПРИЗНАКОВ (Random Forest)")
    print("=" * 70)
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Топ-10 важнейших признаков:\n")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']:25s}: {row['Importance']:.4f}")
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top10 = feature_importance.head(10)
    ax.barh(top10['Feature'], top10['Importance'], color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Важность', fontsize=12)
    ax.set_title('Топ-10 важнейших признаков (Random Forest)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Сохраняем
    output_path = figures_dir / '06_feature_importance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ График сохранён: {output_path}")
    
    plt.show()
    
    # 8. ПРЕДСКАЗАНИЯ НА TEST И СОЗДАНИЕ SUBMISSION
    print("\n" + "=" * 70)
    print("🎯 8. ПРЕДСКАЗАНИЯ НА TEST И СОЗДАНИЕ SUBMISSION")
    print("=" * 70)
    
    print(f"\nИспользуем лучшую модель: {best_model_name}")
    
    # Предсказания на test
    y_test_pred = best_model.predict(X_test)
    
    print(f"✅ Предсказания сделаны для {len(y_test_pred):,} клиентов")
    
    # Распределение предсказаний
    unique, counts = np.unique(y_test_pred, return_counts=True)
    print(f"\nРаспределение предсказаний:")
    for label, count in zip(unique, counts):
        pct = (count / len(y_test_pred)) * 100
        print(f"  Класс {label}: {count:,} ({pct:.2f}%)")
    
    # Создание submission файла
    submission = pd.DataFrame({
        'id': test_ids['id'],
        'Exited': y_test_pred
    })
    
    # Сохранение
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSIONS_DIR / 'submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission файл создан: {submission_path}")
    print(f"   Строк: {len(submission):,}")
    print(f"   Столбцы: id, Exited")
    
    # Сохранение лучшей модели
    models_dir = RESULTS_DIR / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = 'random_forest.pkl' if best_model_name == 'Random Forest' else 'logistic_regression.pkl'
    model_path = models_dir / model_filename
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"✅ Модель сохранена: {model_path}")
    
    # Сохранение таблицы сравнения
    tables_dir = RESULTS_DIR / 'tables'
    comparison.to_csv(tables_dir / '06_model_comparison.csv', index=False)
    feature_importance.to_csv(tables_dir / '06_feature_importance.csv', index=False)
    
    # ФИНАЛ
    print("\n" + "=" * 70)
    print("ЭТАП 6 ЗАВЕРШЁН! ПРОЕКТ ГОТОВ! ")
    print("=" * 70)
    print(f"\n🏆 Результаты:")
    print(f"  • Лучшая модель: {best_model_name}")
    print(f"  • Accuracy:  {accuracy_rf if best_model_name == 'Random Forest' else accuracy_lr:.2%}")
    print(f"  • AUC-ROC:   {best_auc:.4f}")
    print(f"  • F1-Score:  {f1_rf if best_model_name == 'Random Forest' else f1_lr:.4f}")
    print(f"\n📁 Файлы созданы:")
    print(f"  • Submission: {submission_path}")
    print(f"  • Модель: {model_path}")
    print(f"  • Графики: results/figures/06_*.png")
    print(f"  • Таблицы: results/tables/06_*.csv")


if __name__ == "__main__":
    main()

