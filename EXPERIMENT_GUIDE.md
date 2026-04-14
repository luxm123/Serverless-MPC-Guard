# 学术实验完整指南

## 实验目标

在固定并发预算（10个容器）下：
- **首要目标**：SLO Violation Rate ≤ 10%
- **次要目标**：最小化 Cost（AWS Lambda 实际定价）
- **验证假设**：MPC 在违约率和成本间取得更好的权衡（Pareto 最优）

## 实验设计

### 数据集

**来源**：Azure Functions 2019 公开数据集
- 14 天数据，每分钟粒度
- 包含数千个函数的调用序列

### 场景分类

从 Azure 数据中提取 **10 个 30 分钟窗口**：

| 类别 | 窗口数 | 筛选条件 | 特征 |
|------|--------|----------|------|
| **Stable** | 5 | CV < 0.3, mean RPS ∈ [8, 12] | 负载平稳，突发少 |
| **Bursty** | 5 | CV ≥ 0.5, mean RPS ∈ [8, 12] | 负载波动大，突发性强 |

**窗口标识**：`stable_01` ~ `stable_05`，`bursty_01` ~ `bursty_05`

### 策略对比（7个）

| 策略 | 类型 | 说明 | 预期表现 |
|------|------|------|----------|
| `static_0.6` | 静态保守 | 固定 60% CPU | 高违约，低成本 |
| `static_0.8` | 静态基准 | 固定 80% CPU | 中等违约，中等成本 |
| `static_1.0` | 静态充足 | 固定 100% CPU | 低违约，高成本 |
| `aws_tt` | 工业标准 | AWS Target Tracking（目标利用率 75%） | 有超调，不稳定 |
| `hpa_baseline` | PID控制 | Kubernetes HPA 简化版（目标 80%） | 震荡，响应滞后 |
| `mpc` | 模型预测 | 你的 v66.0 MPC（WCP + 鲁棒控制） | **≤10% 违约，成本适中** |
| `oracle** | 离线最优 | 已知未来 30min 负载的全局优化 | 理论上限（Pareto 前沿） |

### 评估指标

**主指标**：
```
SLO Violation Rate = (# requests with latency > base_p90_srv × 1.2) / total_requests
```

**次指标**：
```
Cost = Σ (duration_ms × memory_GB × $0.00001667 / 1000)
     = Σ (duration_ms × (memory_mb/1024) × 0.00001667)
```

**辅助指标**：
- Throughput = successful_requests / experiment_duration
- P50/P90/P99 Latency
- Resource Efficiency = Cost_mpc / Cost_oracle

### 实验配置

```
并发预算：10（固定）
线程池：200
Lambda 内存：128MB（所有策略）
重复次数：每个窗口 × 策略 = 3 次
总实验数：10 窗口 × 7 策略 × 3 次 = 210 次独立运行
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
cd Serverless-MPC-Guard
pip install -r requirements.txt

# 配置 AWS 凭证（用于真实 Lambda 调用）
export AWS_REGION=us-east-1
export MPC_CONTROLLER_NAME=MPC_Controller
export MPC_WORKER_NAME=MPC_BusinessWorker
```

### 2. 生成实验轨迹

```bash
# 步骤 1: 从 Azure 数据中筛选代表性窗口
python experiments/serverless_test/trace_experiment/select_azure_windows.py

# 输出：
# datasets/processed/selected_windows/      (窗口元数据)
# datasets/processed/traces/                (trace_XX.csv)
```

### 3. 运行完整实验

```bash
# 全自动运行（210 次实验）
python experiments/serverless_test/trace_experiment/run_academic_experiment.py \
    --strategies static_0.6 static_0.8 static_1.0 aws_tt hpa_baseline mpc oracle \
    --trials 3 \
    --threads 200 \
    --memory-mb 128
```

**预计耗时**：约 3-6 小时（取决于 AWS Lambda 并发限制）

**实时监控**：
```bash
tail -f experiments/serverless_test/trace_experiment/academic_results_*/interim_results.csv
```

### 4. 自动生成报告

实验完成后自动生成：

```
academic_results_YYYYMMDD_HHMMSS/
├── all_results.csv              # 所有原始数据（210次运行）
├── summary.csv                  # 聚合统计
├── cost_analysis.csv            # 成本分析
├── experiment_config.json       # 实验配置快照
├── figures/
│   ├── violation_rate_comparison.png
│   ├── cost_comparison.png
│   ├── latency_cdf.png
│   └── pareto_frontier.png
└── latex_tables.tex             # LaTeX 表格代码
```

### 5. 推送到 GitHub

```bash
# 自动提交并推送
python scripts/push_to_github.py \
    --message "实验完成：10窗口 × 7策略 × 3次重复，SLO违约率≤10%" \
    --repo /home/ec2-user/Serverless-MPC-Guard
```

## 代码结构

```
Serverless-MPC-Guard/
├── experiments/
│   └── serverless_test/
│       └── trace_experiment/
│           ├── experiment_config.py          # 实验配置常量
│           ├── select_azure_windows.py       # 窗口筛选器
│           ├── run_academic_experiment.py    # 主实验脚本
│           ├── cost_calculator_accurate.py   # 成本计算
│           └── final_results/                # 实验结果（自动生成）
├── src/
│   ├── mpc/
│   │   ├── controller.py    # MPC 控制器
│   │   ├── optimization.py  # v66.0 优化器
│   │   └── middleware.py    # 中间件
│   └── controllers/
│       ├── hpa_baseline_controller.py   # HPA 基线
│       ├── aws_baseline_controller.py   # AWS TT 基线
│       └── oracle_controller.py          # Oracle（新增）
├── lambdas/
│   └── business_worker/
│       └── lambda_function.py            # Worker（支持 oracle）
└── scripts/
    └── push_to_github.py                 # 自动推送
```

## 验证检查清单

- [ ] 所有 7 个策略都已实现并测试
- [ ] 10 个窗口的 trace 文件存在（5 stable + 5 bursty）
- [ ] SLO 动态计算正确（base_p90 × 1.2）
- [ ] Cost 使用真实 AWS 定价公式
- [ ] Oracle 策略使用离线优化（理论上限）
- [ ] 每个策略 × 窗口 × trial 都有独立结果文件
- [ ] LaTeX 表格可直接复制到论文
- [ ] 所有代码已推送到 GitHub

## 常见问题

### Q1: 原始 Azure 数据在哪？
A：论文引用的是 Microsoft 公开数据集，需自行下载到 `datasets/azure_raw/`。如没有，可使用 `select_azure_windows.py` 中内置的模拟数据。

### Q2: 如何只运行某个策略？
```bash
python run_academic_experiment.py --strategies mpc oracle static_1.0
```

### Q3: 实验中断如何恢复？
脚本已支持断点续传：已完成的 trial 会保存在 `interim_results.csv`，下次运行会自动跳过（需手动清理）。

### Q4: Cost 为什么这么低？
AWS Lambda 按毫秒计费。100ms @ 128MB ≈ $0.00000018。百万次请求约 $0.18。

### Q5: 论文中如何呈现结果？
运行 `generate_final_report()` 会自动输出 LaTeX 表格代码，直接复制到 paper.tex 即可。

## 引用

如果使用本实验框架，请引用：

```bibtex
@inproceedings{yourname2026mpc,
  title={Serverless MPC Guard: Model Predictive Control for SLO-Aware Resource Allocation},
  author={Your Name},
  booktitle={IEEE ICWS 2026},
  year={2026}
}
```

## 更新日志

- **2026-04-14**: 初始版本，7策略，10窗口，完整学术实验框架
- Added oracle baseline (offline optimal)
- Fixed Cost calculation to real AWS pricing
- Implemented dynamic SLO with base_p90 tracking
- Auto-generated LaTeX tables
