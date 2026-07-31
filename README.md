# CP Shift Lab

一个用于学习 Conformal Prediction 的一维异方差回归实验。项目从普通 Split CP 出发，逐步展示异方差、协变量偏移、Oracle WCP 和 Estimated WCP 的关系。

## 实验框架

- `T0`：数据生成、评价指标、单元测试。
- `T1`：普通 Split CP，在 `S0` 无偏移场景验证边际覆盖。
- `T2`：自适应分数 CP，比较固定宽度区间和自适应宽度区间。
- `T3`：构造 `S1` 协变量偏移，观察普通 Split CP 欠覆盖。
- `T4`：使用真实密度比实现 Oracle WCP。
- `T5`：使用域分类器估计密度比，实现 Estimated WCP。

## 代码结构

```text
CP_shift_lab/
  config.json              # 样本量、alpha、模型阶数和随机种子
  run_experiment.py        # 统一运行 T0-T5 并保存结果
  README.md
  requirements.txt

  src/
    data.py                # 数据生成、S0-S4、真实密度比
    models.py              # 多项式回归模型
    conformal.py           # Split CP 和 WCP 阈值规则
    metrics.py             # coverage、length、interval score、ESS
    io_utils.py            # 读取配置和保存 CSV
    plots.py               # 绘图函数
    t0.py                  # T0 测试日志
    t1.py                  # 普通 Split CP
    t2.py                  # 自适应 CP
    t3.py                  # 协变量偏移诊断
    t4.py                  # Oracle WCP
    t5.py                  # Estimated WCP

  tests/
    test_conformal.py
    test_data.py
    test_metrics.py
    test_models.py
    test_t5_domain_classifier.py

  outputs/
    T0/
    T1/
    T2/
    T3/
    T4/
    T5/
```

## 运行

```powershell
python -m pytest -q
python run_experiment.py
```

## 主要输出

每个任务的结果保存在 `outputs/T0` 到 `outputs/T5` 中。主要包括：

- `*_raw_metrics.csv`：每个随机种子的原始指标。
- `*_summary.csv`：200 次重复实验的汇总结果。
- `*.png`：对应任务的诊断图和方法对比图。

当前主线结论：普通 Split CP 在 `S1` 协变量偏移下欠覆盖；Oracle WCP 可以用真实密度比修复；Estimated WCP 使用域分类器估计密度比，在当前实验中接近 Oracle WCP。
