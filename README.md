# 6800codework

## 目录

### `Classifier` (classifier.py)
实现了三种分类算法：
- **朴素贝叶斯**，自动处理离散/连续特征。
- **逻辑回归**，采用softmax多分类优化。
- **决策树**，使用基本的CART分裂器和基尼不纯度。

每个分类器都支持训练、预测和评估，并在训练集和测试集上输出准确率、精确率和召回率等指标。

### 数据集加载器 (load_data.py)
提供以下数据集的加载辅助函数：
- **鸢尾花**（`load_iris`）
- **国会投票记录**（`load_congress_data`）
- **MONK’s 问题**（`load_monks`）

每个加载器返回 `(training_data, test_data)` 矩阵，第一列为标签，其余为特征。

### 训练入口 (train_and_test.py)
命令行入口，用于运行分类器并指定数据集。可通过 `--dataset` 和 `--classifier` 参数配置运行。

## 用法

```bash
python train_and_test.py --dataset iris --classifier naive_bayes
python train_and_test.py --dataset congress --classifier logistic_regression --epochs 800
python train_and_test.py --dataset monks1 --classifier decision_tree --max_depth 5
```

## 报告

请为每个数据集和分类器生成一份报告（建议为PDF），内容包括性能指标（如准确率、精确率、召回率、F1等）。`REPORT.md` 提供了基本模板以供参考。
