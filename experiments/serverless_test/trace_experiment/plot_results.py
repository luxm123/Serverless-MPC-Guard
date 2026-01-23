import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import numpy as np

# 设置绘图风格
# 使用 seaborn-ticks 风格，更接近论文发表质量
sns.set_theme(style="ticks", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# === 全局配色方案 (High Contrast) ===
STRATEGY_COLORS = {
    'Baseline': '#d62728',      # Red (Danger/Default)
    'Static': '#ff7f0e',        # Orange (Warning)
    'MPC': '#2ca02c',           # Green (Good/Safe)
    'No-Fidelity': '#9467bd',   # Purple
    'No-Shedding': '#8c564b'    # Brown
}

def get_strategy_color(strategy_name):
    """获取策略对应的颜色，如果未定义则返回灰色"""
    return STRATEGY_COLORS.get(strategy_name, '#7f7f7f')

def save_plot(filename, output_dir):
    """统一保存图表，确保去白边和高 DPI"""
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图表] 已保存: {filename}")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_palette_for_df(df):
    """根据 DataFrame 中的 Strategy 列生成调色板"""
    strategies = sorted(df['Strategy'].unique())
    return [get_strategy_color(s) for s in strategies]

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
    改进：动态调整 Y 轴高度，避免过多留白；增加纹理
    """
    if 'success' not in df.columns:
        df['success'] = True

    stats = df.groupby(['Strategy', 'qos_class']).apply(
        lambda x: pd.Series({
            'SLO Violation Rate (%)': ((x['slo_violation'] | (~x['success'])).sum() / len(x)) * 100
        })
    ).reset_index()

    plt.figure(figsize=(12, 6))
    
    # 动态 Y 轴
    max_val = stats['SLO Violation Rate (%)'].max()
    y_limit = max(5.0, max_val * 1.2)
    
    palette = get_palette_for_df(stats)
    ax = sns.barplot(x='qos_class', y='SLO Violation Rate (%)', hue='Strategy', data=stats, palette=palette, edgecolor='black', alpha=0.9)
    
    # 添加纹理 (黑白打印友好)
    hatches = ['/', '\\', 'x', '.', 'o']
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(hatches[i % len(hatches)])

    plt.title('SLO Violation Rate Comparison (Lower is Better)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('QoS Class', fontsize=16)
    plt.ylabel('SLO Violation Rate (%)', fontsize=16)
    plt.ylim(0, y_limit) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Strategy', fontsize=14, title_fontsize=14, loc='upper right', frameon=True, shadow=True)
    
    # 标注数值
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'bottom', 
                        xytext = (0, 3), 
                        textcoords = 'offset points', fontsize=12, fontweight='bold')

    save_plot('1_slo_comparison.png', output_dir)

def plot_q1_cdf(df, output_dir):
    """
    图2: Q1 尾延迟分布 (CDF Plot)
    改进：X轴动态范围；线条加粗；SLO线淡化
    """
    plt.figure(figsize=(12, 7))
    
    q1_df = df[df['qos_class'] == 'Q1'].copy()
    strategies = sorted(q1_df['Strategy'].unique())
    line_styles = ['-', '--', '-.', ':']
    
    max_p99 = 0
    
    for i, strategy in enumerate(strategies):
        subset = q1_df[q1_df['Strategy'] == strategy]['e2e_latency'].sort_values()
        if subset.empty: continue
            
        y_vals = np.arange(len(subset)) / float(len(subset))
        color = get_strategy_color(strategy)
        style = line_styles[i % len(line_styles)]
        
        plt.plot(subset, y_vals, label=f'{strategy}', linewidth=3, color=color, linestyle=style, alpha=0.9)
        
        p99 = subset.quantile(0.99)
        max_p99 = max(max_p99, p99)
        plt.axvline(x=p99, color=color, linestyle=':', alpha=0.4, linewidth=1.5)

    # SLO 线 (灰色虚线)
    plt.axvline(x=1000, color='gray', linestyle='--', linewidth=2.0, label='SLO (1000ms)')
    
    plt.title('Q1 Latency CDF (Tail Latency Analysis)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('End-to-End Latency (ms)', fontsize=16)
    plt.ylabel('CDF (Probability)', fontsize=16)
    plt.legend(loc='lower right', fontsize=14, frameon=True, shadow=True)
    plt.grid(True, alpha=0.4, linestyle='--')
    
    # 动态 X 轴
    x_limit = max(1200, max_p99 * 1.2)
    plt.xlim(0, x_limit) 
    
    save_plot('2_q1_latency_cdf.png', output_dir)

def plot_goodput_stacked(df, output_dir):
    """
    图3: 有效吞吐量堆叠图 (Stacked Bar Chart)
    改进：高对比度配色；网格线
    """
    df['is_success'] = (~df['slo_violation']) & (~df['shed_by_worker'])
    stats = df[df['is_success']].groupby(['Strategy', 'qos_class']).size().unstack(fill_value=0)
    
    qos_order = ['Q1', 'Q2', 'Q3']
    stats = stats.reindex(columns=qos_order, fill_value=0)
    
    if stats.empty:
        print("[警告] 没有有效请求，跳过吞吐量图")
        return

    # Q1=Purple, Q2=Teal, Q3=Yellow
    qos_colors = ['#4a1486', '#008080', '#fdb462'] 
    
    ax = stats.plot(kind='bar', stacked=True, figsize=(12, 7), color=qos_colors, edgecolor='black', width=0.6)
    
    plt.title('Effective Throughput (Goodput) by Strategy', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Strategy', fontsize=16)
    plt.ylabel('Total Successful Requests', fontsize=16)
    plt.xticks(rotation=0, fontsize=14)
    plt.legend(title='QoS Class', fontsize=14, title_fontsize=14, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')

    for c in ax.containers:
        labels = [f'{v.get_height():.0f}' if v.get_height() > 50 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=11, fontweight='bold')

    save_plot('3_goodput_stacked.png', output_dir)

def plot_fidelity_comparison(df, output_dir):
    """
    图4: 平均保真度对比 (Grouped Bar Chart)
    """
    if 'fidelity' not in df.columns:
        df['fidelity'] = 1.0
    else:
        df['fidelity'] = df['fidelity'].fillna(1.0)

    stats = df.groupby(['Strategy', 'qos_class'])['fidelity'].mean().reset_index()
    stats['fidelity'] = stats['fidelity'] * 100.0

    plt.figure(figsize=(12, 6))
    palette = get_palette_for_df(stats)
    
    ax = sns.barplot(x='qos_class', y='fidelity', hue='Strategy', data=stats, palette=palette, edgecolor='black', alpha=0.9)
    
    # 添加纹理
    hatches = ['/', '\\', 'x', '.', 'o']
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(hatches[i % len(hatches)])
    
    plt.title('Average Fidelity Comparison (Trade-off Analysis)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('QoS Class', fontsize=16)
    plt.ylabel('Average Fidelity (%)', fontsize=16)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Strategy', fontsize=14, title_fontsize=14, loc='lower right', frameon=True, shadow=True)
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.1f}%', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', 
                        xytext = (0, 3), 
                        textcoords = 'offset points', fontsize=11, fontweight='bold')

    save_plot('4_fidelity_comparison.png', output_dir)

def plot_p99_latency_comparison(df, output_dir):
    """
    图5: P99 尾延迟对比 (Grouped Bar Chart)
    改进：SLO 线改为灰色虚线
    """
    stats = df.groupby(['Strategy', 'qos_class'])['e2e_latency'].quantile(0.99).reset_index()
    
    plt.figure(figsize=(12, 6))
    palette = get_palette_for_df(stats)
    
    ax = sns.barplot(x='qos_class', y='e2e_latency', hue='Strategy', data=stats, palette=palette, edgecolor='black')
    
    # 添加纹理
    hatches = ['/', '\\', 'x', '.', 'o']
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(hatches[i % len(hatches)])
    
    plt.title('P99 Tail Latency Comparison (Lower is Better)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('QoS Class', fontsize=16)
    plt.ylabel('P99 Latency (ms) [Log Scale]', fontsize=16)
    
    plt.yscale('log')
    
    # SLO 线 (灰色)
    plt.axhline(y=1000, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(x=-0.4, y=1100, s='SLO Target (1000ms)', color='gray', fontsize=12, fontweight='bold')

    plt.legend(fontsize=14, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, which="both", ls="--", alpha=0.3)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'bottom', 
                        xytext = (0, 3), 
                        textcoords = 'offset points', fontsize=11, fontweight='bold')
            
    save_plot('5_p99_latency_comparison.png', output_dir)

def plot_time_series_adaptation(df, output_dir):
    """
    图6: 动态自适应过程
    改进：统一配色
    """
    if 'timestamp' not in df.columns:
        print("[警告] 数据缺少 'timestamp' 列，跳过时序图绘制")
        return

    df = df.copy()
    df['time_bin'] = (df['timestamp'] // 0.5) * 0.5
    
    strategies = sorted(df['Strategy'].unique())
    colors = [get_strategy_color(s) for s in strategies]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # 1. Request Rate
    sns.histplot(data=df, x='timestamp', hue='Strategy', bins=50, element="step", ax=axes[0], palette=STRATEGY_COLORS, alpha=0.3)
    axes[0].set_ylabel('Request Rate (req/0.5s)', fontsize=14)
    axes[0].set_title('Load / Request Rate', fontsize=16, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 2. Latency
    for i, strategy in enumerate(strategies):
        subset = df[df['Strategy'] == strategy].sort_values('timestamp')
        subset['lat_smooth'] = subset['e2e_latency'].rolling(window=50, min_periods=1).mean()
        axes[1].plot(subset['timestamp'], subset['lat_smooth'], label=strategy, linewidth=2.5, color=get_strategy_color(strategy))
    
    axes[1].axhline(y=1000, color='gray', linestyle='--', label='SLO (1000ms)')
    axes[1].set_ylabel('E2E Latency (ms)', fontsize=14)
    axes[1].set_title('Latency Adaptation (Rolling Mean)', fontsize=16, fontweight='bold')
    axes[1].set_ylim(0, 3000)
    axes[1].legend(loc='upper right', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # 3. Fidelity
    for i, strategy in enumerate(strategies):
        subset = df[df['Strategy'] == strategy].sort_values('timestamp')
        subset['fid_smooth'] = subset['fidelity'].rolling(window=50, min_periods=1).mean()
        axes[2].plot(subset['timestamp'], subset['fid_smooth'], label=strategy, linewidth=2.5, color=get_strategy_color(strategy))

    axes[2].set_ylabel('Fidelity (0-1)', fontsize=14)
    axes[2].set_title('Fidelity Scaling', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=16)
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(loc='lower right', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    save_plot('6_time_series_adaptation.png', output_dir)

def print_summary_table(df):
    """
    打印汇总统计表到控制台
    """
    print("\n=== 实验结果汇总 (Summary) ===")
    
    # 确保 success 列
    if 'success' not in df.columns:
        df['success'] = True
    
    # 确保 fidelity 列
    if 'fidelity' not in df.columns:
        df['fidelity'] = 1.0
    else:
        df['fidelity'] = df['fidelity'].fillna(1.0)

    # 计算关键指标
    stats = df.groupby(['Strategy', 'qos_class']).apply(
        lambda x: pd.Series({
            'Total Reqs': len(x),
            'SLO Violation (%)': ((x['slo_violation'] | (~x['success'])).sum() / len(x)) * 100,
            'Avg Fidelity (%)': x['fidelity'].mean() * 100,
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
        plot_fidelity_comparison(merged_df, output_dir)
        plot_p99_latency_comparison(merged_df, output_dir)
        plot_time_series_adaptation(merged_df, output_dir)
        print(f"=== 所有图表生成完毕: {output_dir} ===")
    else:
        print("没有数据可绘图")