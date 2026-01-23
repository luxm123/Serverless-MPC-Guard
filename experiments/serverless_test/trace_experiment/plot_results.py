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
    # 计算每个策略每个 QoS 的违约率
    stats = df.groupby(['Strategy', 'qos_class']).apply(
        lambda x: pd.Series({
            'SLO Violation Rate (%)': (x['slo_violation'].sum() / len(x)) * 100
        })
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='qos_class', y='SLO Violation Rate (%)', hue='Strategy', data=stats, palette='muted')
    
    plt.title('SLO Violation Rate Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('QoS Class', fontsize=14)
    plt.ylabel('SLO Violation Rate (%)', fontsize=14)
    plt.ylim(0, 110)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 标注数值
    for p in plt.gca().patches:
        if p.get_height() > 0:
            plt.gca().annotate(f'{p.get_height():.1f}%', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 5), 
                               textcoords = 'offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_slo_comparison.png'), dpi=300)
    plt.close()
    print("[图表] SLO 违约率对比图已生成")

def plot_q1_cdf(df, output_dir):
    """
    图2: Q1 尾延迟分布 (CDF Plot)
    """
    plt.figure(figsize=(10, 6))
    
    # 只筛选 Q1 数据
    q1_df = df[df['qos_class'] == 'Q1'].copy()
    
    strategies = q1_df['Strategy'].unique()
    colors = sns.color_palette('muted', n_colors=len(strategies))
    
    for i, strategy in enumerate(strategies):
        subset = q1_df[q1_df['Strategy'] == strategy]['e2e_latency'].sort_values()
        if subset.empty:
            continue
            
        # 计算 CDF
        y_vals = np.arange(len(subset)) / float(len(subset))
        plt.plot(subset, y_vals, label=f'{strategy} (Q1)', linewidth=2, color=colors[i])
        
        # 标记 P99
        p99 = subset.quantile(0.99)
        plt.axvline(x=p99, color=colors[i], linestyle=':', alpha=0.5)
        plt.text(p99, 0.5, f' P99={p99:.0f}ms', color=colors[i], fontsize=9, rotation=90)

    plt.axvline(x=1000, color='red', linestyle='--', label='SLO (1000ms)')
    
    plt.title('Q1 Latency CDF (Tail Latency Analysis)', fontsize=16, fontweight='bold')
    plt.xlabel('End-to-End Latency (ms)', fontsize=14)
    plt.ylabel('CDF (Probability)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 限制 X 轴范围以聚焦有效区域 (0-3000ms)
    # 超过 3000ms 的长尾对分析 SLO (1000ms) 意义不大，且会压缩有效部分
    plt.xlim(0, 3000) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_q1_latency_cdf.png'), dpi=300)
    plt.close()
    print("[图表] Q1 CDF 分布图已生成")

def plot_goodput_stacked(df, output_dir):
    """
    图3: 有效吞吐量堆叠图 (Stacked Bar Chart)
    """
    # 有效请求：没有违约 且 没有被丢弃
    df['is_success'] = (~df['slo_violation']) & (~df['shed_by_worker'])
    
    # 统计每种策略、每个 QoS 的成功请求总数
    stats = df[df['is_success']].groupby(['Strategy', 'qos_class']).size().unstack(fill_value=0)
    
    # 重新排序 QoS 列，保证堆叠顺序 Q1 在最下
    qos_order = ['Q1', 'Q2', 'Q3']
    stats = stats.reindex(columns=qos_order, fill_value=0)
    
    if stats.empty:
        print("[警告] 没有有效请求，跳过吞吐量图")
        return

    # 绘制堆叠图
    stats.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    
    plt.title('Effective Throughput (Goodput) by Strategy', fontsize=16, fontweight='bold')
    plt.xlabel('Strategy', fontsize=14)
    plt.ylabel('Total Successful Requests', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='QoS Class', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_goodput_stacked.png'), dpi=300)
    plt.close()
    print("[图表] 有效吞吐量堆叠图已生成")

def plot_fidelity_comparison(df, output_dir):
    """
    图4: 平均保真度对比 (Grouped Bar Chart)
    """
    # 确保有 fidelity 列，如果没有则默认为 1.0 (Baseline/Static)
    if 'fidelity' not in df.columns:
        df['fidelity'] = 1.0
    else:
        df['fidelity'] = df['fidelity'].fillna(1.0)

    # 计算平均保真度
    stats = df.groupby(['Strategy', 'qos_class'])['fidelity'].mean().reset_index()
    stats['fidelity'] = stats['fidelity'] * 100.0  # 转百分比

    plt.figure(figsize=(10, 6))
    sns.barplot(x='qos_class', y='fidelity', hue='Strategy', data=stats, palette='muted')
    
    plt.title('Average Fidelity Comparison (Trade-off Analysis)', fontsize=16, fontweight='bold')
    plt.xlabel('QoS Class', fontsize=14)
    plt.ylabel('Average Fidelity (%)', fontsize=14)
    plt.ylim(0, 110)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    for p in plt.gca().patches:
        if p.get_height() > 0:
            plt.gca().annotate(f'{p.get_height():.1f}%', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 5), 
                               textcoords = 'offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_fidelity_comparison.png'), dpi=300)
    plt.close()
    print("[图表] 保真度对比图已生成")

def plot_p99_latency_comparison(df, output_dir):
    """
    图5: P99 尾延迟对比 (Grouped Bar Chart)
    """
    stats = df.groupby(['Strategy', 'qos_class'])['e2e_latency'].quantile(0.99).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='qos_class', y='e2e_latency', hue='Strategy', data=stats, palette='muted')
    
    plt.title('P99 Tail Latency Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('QoS Class', fontsize=14)
    plt.ylabel('P99 Latency (ms) [Log Scale]', fontsize=14)
    
    # 使用对数坐标，解决 Baseline (5000ms) 和 MPC (200ms) 比例悬殊问题
    plt.yscale('log')
    
    # 增加 SLO 线
    plt.axhline(y=1000, color='red', linestyle='--', label='SLO (1000ms)', linewidth=2)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)

    for p in plt.gca().patches:
        if p.get_height() > 0:
            plt.gca().annotate(f'{int(p.get_height())}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 5), 
                               textcoords = 'offset points', fontsize=9)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_p99_latency_comparison.png'), dpi=300)
    plt.close()
    print("[图表] P99 尾延迟对比图已生成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate comparison plots for Serverless MPC Guard')
    parser.add_argument('files', nargs='+', help='CSV files to compare (e.g. baseline.csv mpc.csv)')
    parser.add_argument('--labels', nargs='+', help='Labels for each file (e.g. Baseline MPC)')
    
    args = parser.parse_args()
    
    # 如果没有提供参数，尝试默认行为
    if not args.files:
        # 默认回退逻辑 (方便调试)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        
        default_files = []
        default_labels = []
        
        for strategy in ['baseline', 'static', 'mpc']:
            f_path = os.path.join(results_dir, f'results_{strategy}.csv')
            if os.path.exists(f_path):
                default_files.append(f_path)
                default_labels.append(strategy.capitalize())
        
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
        plot_slo_comparison(merged_df, output_dir)
        plot_q1_cdf(merged_df, output_dir)
        plot_goodput_stacked(merged_df, output_dir)
        plot_fidelity_comparison(merged_df, output_dir)
        plot_p99_latency_comparison(merged_df, output_dir)
        print(f"=== 所有图表生成完毕: {output_dir} ===")
    else:
        print("没有数据可绘图")