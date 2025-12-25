"""
机器学习重要曲线可视化工具
结合 sklearn 和 matplotlib 创建专业的学习曲线、损失曲线、ROC曲线等
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import os

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 忽略字体警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 创建输出目录
output_dir = "ml_curves_visualization"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("📊 机器学习重要曲线可视化工具")
print("=" * 60)

# ============================================
# 1. 学习曲线可视化
# ============================================
print("\n📈 1. 创建学习曲线可视化...")


def create_learning_curves():
    """创建多种模型的学习曲线对比"""

    # 生成分类数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42
    )

    # 定义不同的模型
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("学习曲线对比分析", fontsize=16, fontweight="bold")

    # 为每个模型绘制学习曲线
    for idx, (name, model) in enumerate(models.items()):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        # 计算学习曲线
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            n_jobs=-1,
            random_state=42,
        )

        # 绘制曲线
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        ax.plot(train_sizes, train_mean, "o-", color="blue", label="训练集")
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )

        ax.plot(train_sizes, val_mean, "s-", color="red", label="验证集")
        ax.fill_between(
            train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red"
        )

        # 添加诊断信息
        final_train_score = train_mean[-1]
        final_val_score = val_mean[-1]
        gap = final_train_score - final_val_score

        # 诊断文本
        if gap > 0.1:
            diagnosis = "过拟合 (高方差)"
            color = "orange"
        elif final_train_score < 0.7:
            diagnosis = "欠拟合 (高偏差)"
            color = "red"
        else:
            diagnosis = "拟合良好"
            color = "green"

        ax.text(
            0.5,
            0.05,
            diagnosis,
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color=color,
            weight="bold",
        )

        ax.set_xlabel("训练样本数")
        ax.set_ylabel("得分")
        ax.set_title(
            f"{name}\n最终得分: 训练={final_train_score:.3f}, 验证={final_val_score:.3f}"
        )
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    # 隐藏最后一个子图
    axes[1, 1].axis("off")

    # 添加总结信息
    summary_text = """
学习曲线诊断指南:
🔴 高偏差: 训练和验证得分都很低 → 增加模型复杂度
🔵 高方差: 训练得分高，验证得分低 → 增加数据或正则化
🟢 理想: 两者都高且接近 → 模型状态良好
    """
    axes[1, 1].text(
        0.1,
        0.5,
        summary_text,
        fontsize=12,
        verticalalignment="center",
        family="monospace",
    )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "1_learning_curves_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ 保存: {output_file}")


# ============================================
# 2. 损失曲线可视化
# ============================================
print("\n📉 2. 创建损失曲线可视化...")


def create_loss_curves():
    """创建训练过程中的损失曲线"""

    # 生成回归数据
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # 模拟训练过程
    epochs = 100
    train_losses = []
    val_losses = []

    # 使用随机森林模拟训练过程
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 模拟训练损失（实际应用中从训练循环获取）
    np.random.seed(42)
    for epoch in range(epochs):
        # 模拟训练损失下降趋势
        train_loss = 10 * np.exp(-epoch / 20) + 0.1 * np.random.random()
        train_losses.append(train_loss)

        # 模拟验证损失（先下降后可能上升）
        if epoch < 60:
            val_loss = 8 * np.exp(-epoch / 25) + 0.2 * np.random.random()
        else:
            val_loss = 2 + 0.01 * (epoch - 60) + 0.2 * np.random.random()
        val_losses.append(val_loss)

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 损失曲线
    ax1.plot(range(epochs), train_losses, "b-", label="训练损失", linewidth=2)
    ax1.plot(range(epochs), val_losses, "r-", label="验证损失", linewidth=2)

    # 标记最佳点
    best_epoch = np.argmin(val_losses)
    ax1.axvline(
        best_epoch,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"最佳停止点 (Epoch {best_epoch})",
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 损失变化率
    train_grad = np.gradient(train_losses)
    val_grad = np.gradient(val_losses)

    ax2.plot(range(epochs - 1), train_grad, "b-", alpha=0.7, label="训练损失梯度")
    ax2.plot(range(epochs - 1), val_grad, "r-", alpha=0.7, label="验证损失梯度")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("损失变化率")
    ax2.set_title("损失梯度变化")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "2_loss_curves.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ 保存: {output_file}")


# ============================================
# 3. ROC曲线可视化
# ============================================
print("\n🎯 3. 创建ROC曲线可视化...")


def create_roc_curves():
    """创建多种模型的ROC曲线对比"""

    # 生成分类数据
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        random_state=42,
    )

    # 分割数据
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 定义模型
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 为每个模型绘制ROC曲线
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)

        # 预测概率
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.decision_function(X_test)

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    # 绘制随机分类器基线
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="随机分类器 (AUC = 0.5)")

    plt.xlabel("假正例率 (FPR)")
    plt.ylabel("真正例率 (TPR)")
    plt.title("ROC曲线对比")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 添加性能指标文本
    textstr = "性能评估:\n"
    textstr += "• AUC > 0.9: 优秀\n"
    textstr += "• AUC 0.7-0.9: 良好\n"
    textstr += "• AUC 0.5-0.7: 一般\n"
    textstr += "• AUC < 0.5: 差"

    plt.text(
        0.02,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "3_roc_curves.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ 保存: {output_file}")


# ============================================
# 4. 验证曲线可视化
# ============================================
print("\n📊 4. 创建验证曲线可视化...")


def create_validation_curves():
    """创建超参数验证曲线"""

    # 生成数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=42
    )

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("超参数验证曲线", fontsize=16, fontweight="bold")

    # 1. Random Forest max_depth
    ax = axes[0, 0]
    param_range = range(1, 21)
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42),
        X,
        y,
        param_name="max_depth",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    ax.plot(param_range, train_mean, "o-", color="blue", label="训练集")
    ax.plot(param_range, val_mean, "s-", color="red", label="验证集")

    best_depth = param_range[np.argmax(val_mean)]
    ax.axvline(best_depth, color="green", linestyle="--", alpha=0.7)
    ax.text(
        best_depth + 0.5,
        ax.get_ylim()[0] * 0.9,
        f"最佳: {best_depth}",
        fontsize=10,
        color="green",
    )

    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Random Forest: max_depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. SVM C参数
    ax = axes[0, 1]
    param_range = np.logspace(-3, 3, 7)
    train_scores, val_scores = validation_curve(
        SVC(kernel="rbf", random_state=42),
        X,
        y,
        param_name="C",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    ax.semilogx(param_range, train_mean, "o-", color="blue", label="训练集")
    ax.semilogx(param_range, val_mean, "s-", color="red", label="验证集")

    best_c = param_range[np.argmax(val_mean)]
    ax.axvline(best_c, color="green", linestyle="--", alpha=0.7)
    ax.text(
        best_c * 1.5,
        ax.get_ylim()[0] * 0.9,
        f"最佳: {best_c:.3f}",
        fontsize=10,
        color="green",
    )

    ax.set_xlabel("C")
    ax.set_ylabel("Accuracy")
    ax.set_title("SVM (RBF): C参数")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 学习率影响（模拟）
    ax = axes[1, 0]
    learning_rates = np.logspace(-4, 0, 20)
    train_scores = []
    val_scores = []

    for lr in learning_rates:
        # 模拟不同学习率的效果
        np.random.seed(42)
        train_acc = 0.9 - 0.3 * np.exp(-lr * 10) + 0.05 * np.random.random()
        val_acc = 0.85 - 0.4 * np.exp(-lr * 8) + 0.1 * np.random.random()

        train_scores.append(train_acc)
        val_scores.append(val_acc)

    ax.semilogx(learning_rates, train_scores, "o-", color="blue", label="训练集")
    ax.semilogx(learning_rates, val_scores, "s-", color="red", label="验证集")

    best_lr = learning_rates[np.argmax(val_scores)]
    ax.axvline(best_lr, color="green", linestyle="--", alpha=0.7)
    ax.text(
        best_lr * 1.5,
        ax.get_ylim()[0] * 0.9,
        f"最佳: {best_lr:.4f}",
        fontsize=10,
        color="green",
    )

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("学习率影响 (模拟)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 样本量影响
    ax = axes[1, 1]
    sample_sizes = np.linspace(0.1, 1.0, 10)
    train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        X,
        y,
        train_sizes=sample_sizes,
        cv=5,
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    ax.plot(sample_sizes * len(X), train_mean, "o-", color="blue", label="训练集")
    ax.plot(sample_sizes * len(X), val_mean, "s-", color="red", label="验证集")

    # 计算gap
    gap = train_mean - val_mean
    ax2 = ax.twinx()
    ax2.plot(sample_sizes * len(X), gap, "g--", alpha=0.7, label="Gap")
    ax2.set_ylabel("训练-验证 Gap", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    ax.set_xlabel("样本数量")
    ax.set_ylabel("Accuracy")
    ax.set_title("样本量影响")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "4_validation_curves.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ 保存: {output_file}")


# ============================================
# 5. 综合分析仪表板
# ============================================
print("\n🎛️ 5. 创建综合分析仪表板...")


def create_comprehensive_dashboard():
    """创建包含所有曲线的综合分析仪表板"""

    # 生成数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=42
    )

    # 创建大型图形
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("机器学习模型综合分析仪表板", fontsize=18, fontweight="bold")

    # 1. 学习曲线 (左上)
    ax1 = plt.subplot(2, 3, 1)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
    )

    ax1.plot(train_sizes * len(X), np.mean(train_scores, axis=1), "b-", label="训练")
    ax1.plot(train_sizes * len(X), np.mean(val_scores, axis=1), "r-", label="验证")
    ax1.set_xlabel("样本数")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("学习曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. ROC曲线 (中上)
    ax2 = plt.subplot(2, 3, 2)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    ax2.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.set_title("ROC曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 特征重要性 (右上)
    ax3 = plt.subplot(2, 3, 3)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # 前10个重要特征

    ax3.barh(range(len(indices)), importances[indices])
    ax3.set_yticks(range(len(indices)))
    ax3.set_yticklabels([f"Feature_{i}" for i in indices])
    ax3.set_xlabel("重要性")
    ax3.set_title("特征重要性 (Top 10)")
    ax3.grid(True, alpha=0.3)

    # 4. 混淆矩阵 (左下)
    ax4 = plt.subplot(2, 3, 4)
    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    im = ax4.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax4.figure.colorbar(im, ax=ax4)

    ax4.set_xlabel("预测标签")
    ax4.set_ylabel("真实标签")
    ax4.set_title("混淆矩阵")

    # 添加数字
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # 5. 精确率-召回率曲线 (中下)
    ax5 = plt.subplot(2, 3, 5)
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    ax5.plot(recall, precision, label="PR曲线")
    ax5.set_xlabel("Recall")
    ax5.set_ylabel("Precision")
    ax5.set_title("精确率-召回率曲线")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 模型对比 (右下)
    ax6 = plt.subplot(2, 3, 6)
    models_comparison = {
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "LR": LogisticRegression(random_state=42, max_iter=1000),
    }

    scores = []
    names = []
    for name, model in models_comparison.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        scores.append(cv_scores.mean())
        names.append(name)

    bars = ax6.bar(names, scores, color=["blue", "red", "green"])
    ax6.set_ylabel("交叉验证得分")
    ax6.set_title("模型对比")
    ax6.set_ylim(0, 1)

    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "5_comprehensive_dashboard.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ 保存: {output_file}")


# ============================================
# 执行所有可视化
# ============================================
try:
    create_learning_curves()
    print("   ✅ 学习曲线创建成功")
except Exception as e:
    print(f"   ❌ 学习曲线创建失败: {e}")

try:
    create_loss_curves()
    print("   ✅ 损失曲线创建成功")
except Exception as e:
    print(f"   ❌ 损失曲线创建失败: {e}")

try:
    create_roc_curves()
    print("   ✅ ROC曲线创建成功")
except Exception as e:
    print(f"   ❌ ROC曲线创建失败: {e}")

try:
    create_validation_curves()
    print("   ✅ 验证曲线创建成功")
except Exception as e:
    print(f"   ❌ 验证曲线创建失败: {e}")

try:
    create_comprehensive_dashboard()
    print("   ✅ 综合仪表板创建成功")
except Exception as e:
    print(f"   ❌ 综合仪表板创建失败: {e}")

# ============================================
# 打印总结
# ============================================

print("\n" + "=" * 60)
print("✨ 机器学习重要曲线可视化工具创建完成！")
print("=" * 60)

print(f"\n📂 生成的文件位于: {output_dir}/")
print("   1. learning_curves_comparison.png - 学习曲线对比分析")
print("   2. loss_curves.png - 训练损失曲线")
print("   3. roc_curves.png - ROC曲线对比")
print("   4. validation_curves.png - 超参数验证曲线")
print("   5. comprehensive_dashboard.png - 综合分析仪表板")

print("\n💡 使用说明:")
print("   • 所有PNG图片都包含详细的诊断信息")
print("   • 学习曲线帮助判断过拟合/欠拟合")
print("   • 损失曲线监控训练过程和早停点")
print("   • ROC曲线评估分类器性能")
print("   • 验证曲线指导超参数调优")
print("   • 综合仪表板提供全方位分析")

print("\n🎯 核心功能:")
print("   • 多模型对比分析")
print("   • 自动诊断和建议")
print("   • 专业的可视化效果")
print("   • 完整的评估指标")
print("   • 详细的注释说明")

print("\n" + "=" * 60)
print("🎉 现在可以在这些PNG图片中查看专业的机器学习曲线分析！")
print("=" * 60)
