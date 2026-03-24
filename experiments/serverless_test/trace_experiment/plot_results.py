import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import numpy as np

# 设置绘图风格
sns.set(style="whitegrid")
# 尝试设置中文字体，如果失败则回退到默认
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_merge_data(files, labels):
    """
    读取多个 CSV 文件并合并，添加 Strategy 标签
    """
    merged_df = pd.DataFrame()
    for f, label in zip(files, labels):
        if not os.path.exists(f):
            print(f"[警告] 文件不存在: {f}")
            continue
        try:
            df = pd.read_csv(f)
            df['Strategy'] = label
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            print(f"已加载 {label}: {len(df)} 条记录")
        except Exception as e:
            print(f"[错误] 读取 {f} 失败: {e}")
    return merged_df

def plot_slo_comparison(df, output_dir):
    """
    图1: SLA 违约率对比 (Grouped Bar Chart)
    """
    if 'success' not in df.columns:
        df['success'] = True

    stats = df.groupby(['Strategy', 'qos_class']).apply(
        lambda x: pd.Series({
            'SLO Violation Rate (%)': ((x['slo_violation'] | (~x['success'])).sum() / len(x)) * 100
        })
    ).reset_index()

    plt.figure(figsize=(10, 6))
    
    # 获取最大违约率以动态设置 Y 轴
    max_val = stats['SLO Violation Rate (%)'].max()
    y_limit = max(5.0, max_val * 1.2)
    
    ax = sns.barplot(x='qos_class', y='SLO Violation Rate (%)', hue='Strategy', data=stats, palette='muted', edgecolor='black', alpha=0.9)
    
    plt.title('SLO Violation Rate Comparison (Lower is Better)', fontsize=16, fontweight='bold')
    plt.xlabel('QoS Class', fontsize=14)
    plt.ylabel('SLO Violation Rate (%)', fontsize=14)
    plt.ylim(0, y_limit) 
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 标注数值
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'bottom', 
                        xytext = (0, 3), 
                        textcoords = 'offset points', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_slo_comparison.png'), dpi=300)
    plt.close()
    print("[图表] SLO 违约率对比图已生成")

def plot_q1_cdf(df, output_dir):
    """
    图2: Q1 尾延迟分布 (CDF Plot)
    """
    plt.figure(figsize=(10, 6))
    
    q1_df = df[df['qos_class'] == 'Q1'].copy()
    strategies = sorted(q1_df['Strategy'].unique())
    colors = sns.color_palette('muted', n_colors=len(strategies))
    line_styles = ['-', '--', '-.', ':']
    
    max_p99 = 0
    
    for i, strategy in enumerate(strategies):
        subset = q1_df[q1_df['Strategy'] == strategy]['e2e_latency'].sort_values()
        if subset.empty: continue
            
        y_vals = np.arange(len(subset)) / float(len(subset))
        style = line_styles[i % len(line_styles)]
        
        plt.plot(subset, y_vals, label=f'{strategy}', linewidth=2, color=colors[i], linestyle=style)
        
        p99 = subset.quantile(0.99)
        max_p99 = max(max_p99, p99)
        plt.axvline(x=p99, color=colors[i], linestyle=':', alpha=0.5)

    plt.axvline(x=1000, color='gray', linestyle='--', linewidth=2, label='SLO (1000ms)')
    
    plt.title('Q1 Latency CDF (Tail Latency Analysis)', fontsize=16, fontweight='bold')
    plt.xlabel('End-to-End Latency (ms)', fontsize=14)
    plt.ylabel('CDF (Probability)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    x_limit = max(1500, max_p99 * 1.3)
    plt.xlim(0, x_limit) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_q1_latency_cdf.png'), dpi=300)
    plt.close()
    print("[图表] Q1 CDF 分布图已生成")

def plot_goodput_stacked(df, output_dir):
    """
    图3: 请求结果分布 (Request Outcome Distribution) - 分组柱状图 (Grouped Bar)
    按用户建议修改：
    1. 采用建议3：分组柱状图 (非堆叠)，直观对比各状态的绝对数值。
    2. 保留用户满意的浅色系 (Pastel) 配色。
    """
    df = df.copy()
    
    # 定义请求状态
    condition_fail = (~df['success']) | (df.get('shed_by_worker', False))
    condition_violation = (df['success']) & (df['slo_violation']) & (~df.get('shed_by_worker', False))
    condition_good = (df['success']) & (~df['slo_violation']) & (~df.get('shed_by_worker', False))
    
    df['Outcome'] = 'Unknown'
    df.loc[condition_fail, 'Outcome'] = 'Failed/Shed'
    df.loc[condition_violation, 'Outcome'] = 'SLO Violation (Late)'
    df.loc[condition_good, 'Outcome'] = 'Success (Met SLO)'
    
    # 统计每种策略的各类请求数量
    stats = df.groupby(['Strategy', 'Outcome']).size().unstack(fill_value=0)
    
    # 确保列顺序
    columns_order = ['Success (Met SLO)', 'SLO Violation (Late)', 'Failed/Shed']
    columns_order = [c for c in columns_order if c in stats.columns]
    stats = stats.reindex(columns=columns_order, fill_value=0)
    
    if stats.empty:
        print("[警告] 数据为空，跳过结果分布图")
        return

    # 保留用户满意的浅色系 (Pastel)
    outcome_colors = {
        'Success (Met SLO)': '#CCEBC5',       # Pastel Green
        'SLO Violation (Late)': '#FED9A6',    # Pastel Orange
        'Failed/Shed': '#FBB4AE'              # Pastel Red
    }
    colors = [outcome_colors[c] for c in columns_order]
    
    plt.figure(figsize=(12, 6))
    
    # stacked=False 实现分组柱状图
    ax = stats.plot(kind='bar', stacked=False, figsize=(12, 6), color=colors, edgecolor='white', width=0.8)
    
    plt.title('Request Outcomes by Strategy (Grouped Comparison)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Strategy', fontsize=13)
    plt.ylabel('Number of Requests', fontsize=13)
    plt.xticks(rotation=0, fontsize=11)
    
    # 图例放右上角
    plt.legend(title='', fontsize=11, loc='upper right', frameon=True)
    plt.grid(True, axis='y', alpha=0.2, linestyle='--')

    # 标注数值：为了避免拥挤，只标注 > 100 的数值
    for c in ax.containers:
        labels = [f'{int(v.get_height())}' if v.get_height() > 100 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge', padding=3, color='#555555', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_response_outcome_distribution.png'), dpi=300)
    plt.close()
    print("[图表] 请求结果分布图 (分组柱状版) 已生成")

def plot_allocation_comparison(df, output_dir):
    """
    图4: 平均资源分配对比 (Grouped Bar Chart)
    """
    if 'alloc' not in df.columns:
        df['alloc'] = 1.0
    else:
        df['alloc'] = df['alloc'].fillna(1.0)

    stats = df.groupby(['Strategy', 'qos_class'])['alloc'].mean().reset_index()
    stats['alloc'] = stats['alloc'] * 100.0

    plt.figure(figsize=(10, 6))
    sns.barplot(x='qos_class', y='alloc', hue='Strategy', data=stats, palette='muted')
    
    plt.title('Average CPU Allocation Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('QoS Class', fontsize=14)
    plt.ylabel('Average Allocation (%)', fontsize=14)
    plt.ylim(0, 110)
    
    for p in plt.gca().patches:
        if p.get_height() > 0:
            plt.gca().annotate(f'{p.get_height():.1f}%', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 5), 
                               textcoords = 'offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_allocation_comparison.png'), dpi=300)
    plt.close()
    print("[图表] 资源分配对比图已生成")

def plot_p99_latency_comparison(df, output_dir):
    """
    图5: P99 尾延迟对比 (Grouped Bar Chart)
    """
    stats = df.groupby(['Strategy', 'qos_class'])['e2e_latency'].quantile(0.99).reset_index()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='qos_class', y='e2e_latency', hue='Strategy', data=stats, palette='muted', edgecolor='black')
    
    plt.title('P99 Tail Latency Comparison (Lower is Better)', fontsize=16, fontweight='bold')
    plt.xlabel('QoS Class', fontsize=14)
    plt.ylabel('P99 Latency (ms) [Log Scale]', fontsize=14)
    
    plt.yscale('log')
    
    plt.axhline(y=1000, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(x=-0.4, y=1050, s='SLO Target (1000ms)', color='gray', fontsize=10, fontweight='bold')

    plt.legend(fontsize=12)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'bottom', 
                        xytext = (0, 3), 
                        textcoords = 'offset points', fontsize=10, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_p99_latency_comparison.png'), dpi=300)
    plt.close()
    print("[图表] P99 尾延迟对比图已生成")

def plot_time_series_adaptation(df, output_dir):
    """
    图6: 动态自适应过程
    """
    if 'timestamp' not in df.columns:
        print("[警告] 数据缺少 'timestamp' 列，跳过时序图绘制")
        return

    # Binning data by 0.5s intervals
    df = df.copy()
    df['time_bin'] = (df['timestamp'] // 0.5) * 0.5
    
    strategies = df['Strategy'].unique()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Subplot 1: Request Rate (Load)
    sns.histplot(data=df, x='timestamp', hue='Strategy', bins=50, element="step", ax=axes[0], alpha=0.3)
    axes[0].set_ylabel('Request Rate (req/0.5s)', fontsize=12)
    axes[0].set_title('Load / Request Rate', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')

    # Subplot 2: Latency (Rolling Mean)
    for strategy in strategies:
        subset = df[df['Strategy'] == strategy].sort_values('timestamp')
        # Rolling average of 50 requests
        subset['lat_smooth'] = subset['e2e_latency'].rolling(window=50, min_periods=1).mean()
        axes[1].plot(subset['timestamp'], subset['lat_smooth'], label=strategy, linewidth=2)
    
    axes[1].axhline(y=1000, color='r', linestyle='--', label='SLO (1000ms)')
    axes[1].set_ylabel('E2E Latency (ms)', fontsize=12)
    axes[1].set_title('Latency Adaptation (Rolling Mean)', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 3000) # Cap at 3s for readability
    axes[1].legend(loc='upper right')

    # Subplot 3: Allocation (Rolling Mean)
    for strategy in strategies:
        subset = df[df['Strategy'] == strategy].sort_values('timestamp')
        subset['alloc_smooth'] = subset['alloc'].rolling(window=50, min_periods=1).mean()
        axes[2].plot(subset['timestamp'], subset['alloc_smooth'], label=strategy, linewidth=2)

    axes[2].set_ylabel('CPU Alloc (0.4-1.0)', fontsize=12)
    axes[2].set_title('Dynamic Resource Allocation', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=14)
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_time_series_adaptation.png'), dpi=300)
    plt.close()
    print("[图表] 动态自适应时序图已生成")


def print_summary_table(df):
    """
    打印汇总统计表到控制台
    """
    print("\n=== 实验结果汇总 (Summary) ===")
    
    # 确保 success 列
    if 'success' not in df.columns:
        df['success'] = True
    
    # 确保 alloc 列
    if 'alloc' not in df.columns:
        df['alloc'] = 1.0
    else:
        df['alloc'] = df['alloc'].fillna(1.0)

    # 计算关键指标
    stats = df.groupby(['Strategy', 'qos_class']).apply(
        lambda x: pd.Series({
            'Total Reqs': len(x),
            'SLO Violation (%)': ((x['slo_violation'] | (~x['success'])).sum() / len(x)) * 100,
            'Avg Alloc (%)': x['alloc'].mean() * 100,
            'P99 Latency (ms)': x['e2e_latency'].quantile(0.99)
        })
    ).reset_index()
    
    # 格式化输出
    print(stats.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("==============================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate comparison plots for Serverless MPC Guard')
    parser.add_argument('files', nargs='*', help='CSV files to compare (e.g. baseline.csv mpc.csv)')
    parser.add_argument('--labels', nargs='+', help='Labels for each file (e.g. Baseline MPC)')
    
    args = parser.parse_args()
    
    # 如果没有提供参数，尝试默认行为
    if not args.files:
        # 默认回退逻辑 (方便调试)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        
        default_files = []
        default_labels = []
        
        # 定义要查找的文件模式和对应的标签
        file_patterns = [
            ('results_baseline.csv', 'Baseline'),
            ('results_static.csv', 'Static'),
            ('results_mpc.csv', 'MPC'),
            ('results_ablation_no_fidelity.csv', 'No-Fidelity'),
            ('results_ablation_no_shedding.csv', 'No-Shedding')
        ]
        
        for filename, label in file_patterns:
            f_path = os.path.join(results_dir, filename)
            if os.path.exists(f_path):
                default_files.append(f_path)
                default_labels.append(label)
        
        if default_files:
            args.files = default_files
            args.labels = default_labels
            print(f"[Info] Automatically found result files: {default_labels}")
        else:
            print("Usage: python plot_results.py file1.csv file2.csv --labels Baseline MPC")
            sys.exit(1)
            
    # 如果没有提供标签，默认使用文件名
    if not args.labels:
        args.labels = [os.path.basename(f).split('.')[0] for f in args.files]
        
    if len(args.files) != len(args.labels):
        print("[错误] 文件数量与标签数量不一致")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(args.files[0]), 'comparison_plots')
    ensure_dir(output_dir)
    
    print(f"=== 开始生成对比图表 ===")
    merged_df = load_and_merge_data(args.files, args.labels)
    
    if not merged_df.empty:
        print_summary_table(merged_df)  # 打印汇总表
        plot_slo_comparison(merged_df, output_dir)
        plot_q1_cdf(merged_df, output_dir)
        plot_goodput_stacked(merged_df, output_dir)
        plot_allocation_comparison(merged_df, output_dir)
        plot_p99_latency_comparison(merged_df, output_dir)
        plot_time_series_adaptation(merged_df, output_dir)
        print(f"=== 所有图表生成完毕: {output_dir} ===")
    else:
        print("没有数据可绘图")