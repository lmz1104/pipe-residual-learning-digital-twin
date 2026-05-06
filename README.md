# Physics-informed Residual Learning for Pipeline Digital Twin Demo

这是一个用于保研/套磁展示的完整代码包，目标是把“热-流耦合管路机理模型 + 残差学习修正 + 结果可视化”整理成一个可运行、可展示、可放 GitHub 的科研型项目。

> 重要说明：当前数据为 pseudo-observed synthetic data，用于展示建模流程和结果表达能力；不能表述为真实工业现场数据结果。

## 1. Project structure

```text
pipe_research_result_package/
├── pipe_research/
│   ├── config.py              # 参数配置
│   ├── physical_model.py      # Darcy-Weisbach 压降机理模型
│   ├── synthetic_data.py      # 合成观测数据与可学习误差构造
│   ├── ml_model.py            # LR / RF 残差学习与 R²/RMSE 评估
│   ├── vibration_model.py     # SDOF 振动代理模型
│   ├── plotting.py            # 论文展示图生成
│   └── evaluation.py          # 结果总结与提升幅度计算
├── outputs/                   # 运行后自动生成结果
├── run_all.py                 # 一键运行全部流程
├── streamlit_app.py           # 可选：交互式 Demo
├── requirements.txt
└── README.md
```

## 2. Installation

```bash
pip install -r requirements.txt
```

## 3. Run all experiments

```bash
python run_all.py
```

运行后会在 `outputs/` 中生成：

- `model_metrics.csv`：MAE / RMSE / R² 对比
- `result_summary.md`：可直接写进 README 或简历的结果描述
- `true_vs_pred.png`：真实值 vs 预测值
- `residual_histogram.png`：残差分布对比
- `metrics_bar_R2.png`：R² 对比图
- `metrics_bar_RMSE.png`：RMSE 对比图
- `feature_importance.png`：RF 特征重要性
- `pressure_drop_curves.png`：机理模型压降曲线
- `vibration_vs_flow.png` / `vibration_vs_pressure.png`：振动代理模型结果

## 4. Example result

在当前默认参数下，`RandomForest correction` 会在测试集上输出类似结果：

```text
RF residual learning reduces RMSE compared with the mechanistic baseline,
and improves R² on the held-out test set.
```

精确数值请以你本地运行后的 `outputs/result_summary.md` 为准。

## 5. Recommended CV wording

中文简历：

> 构建热-流耦合管路机理模型，并基于合成观测数据引入可学习系统误差；采用 LR / RF 残差学习修正机理模型偏差，在测试集上实现 RMSE 明显下降与 R² 提升，形成“机理建模—误差修正—可视化验证”的数字孪生原型。

英文简历：

> Developed a physics-based thermo-fluid pipeline model and integrated LR/RF residual learning to correct systematic model discrepancies. The hybrid model reduced prediction error and improved R² on a held-out synthetic benchmark, forming a prototype workflow for physics-informed digital twin modelling.

## 6. How to present honestly

可以说：

- simulation benchmark
- pseudo-observed data
- controlled residual-learning experiment
- prototype digital twin workflow

不要说：

- industrial field validation
- real pipeline deployment
- 实测工业数据验证

