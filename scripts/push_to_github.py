#!/usr/bin/env python3
"""
GitHub 自动推送脚本

功能：
1. 自动提交实验代码和结果
2. 生成提交信息
3. 推送到 GitHub 仓库

使用方法：
python push_to_github.py --message "实验完成：10窗口 × 7策略 × 3次重复"
"""
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse


def run_command(cmd: str, cwd: str = None) -> tuple[int, str, str]:
    """
    执行 shell 命令

    Returns:
        (returncode, stdout, stderr)
    """
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr


def check_git_status(repo_path: str) -> bool:
    """检查仓库状态"""
    code, out, err = run_command("git status --porcelain", cwd=repo_path)
    if code != 0:
        print(f"[Error] Git status failed: {err}")
        return False
    return bool(out.strip())


def git_push(repo_path: str, commit_message: str, remote: str = "origin", branch: str = "main") -> bool:
    """
    执行完整的 git 推送流程

    Returns:
        True 成功, False 失败
    """
    print(f"\n{'='*60}")
    print(f"Pushing to GitHub: {remote}/{branch}")
    print(f"Commit message: {commit_message}")
    print(f"{'='*60}\n")

    # 1. 检查状态
    if not check_git_status(repo_path):
        print("[Info] No changes to commit.")
        return True

    # 2. 显示变更
    code, out, err = run_command("git status", cwd=repo_path)
    print(out)

    # 3. 添加所有变更
    print("[Step 1/5] git add .")
    code, out, err = run_command("git add .", cwd=repo_path)
    if code != 0:
        print(f"[Error] git add failed: {err}")
        return False

    # 4. 提交
    print("[Step 2/5] git commit")
    code, out, err = run_command(f'git commit -m "{commit_message}"', cwd=repo_path)
    if code != 0:
        print(f"[Error] git commit failed: {err}")
        # 可能��有变更
        if "nothing to commit" in err.lower():
            return True
        return False

    print(f"[Success] Committed: {out.splitlines()[0] if out else 'OK'}")

    # 5. 推送到远程
    print(f"[Step 3/5] git push {remote} {branch}")
    code, out, err = run_command(f"git push {remote} {branch}", cwd=repo_path)

    if code == 0:
        print(f"[Success] Pushed to {remote}/{branch}")
        print(out)
        return True
    else:
        print(f"[Error] Git push failed: {err}")
        print("\nTroubleshooting:")
        print("  1. Check your GitHub authentication (git config user.name/email)")
        print("  2. Ensure you have push permissions")
        print("  3. Check if remote exists: git remote -v")
        return False


def main():
    parser = argparse.ArgumentParser(description='Automated GitHub push for experiment results')
    parser.add_argument('--message', '-m', type=str, default=None,
                       help='Commit message (auto-generated if not provided)')
    parser.add_argument('--repo', type=str, default='.',
                       help='Repository path (default: current directory)')
    parser.add_argument('--remote', type=str, default='origin',
                       help='Remote name (default: origin)')
    parser.add_argument('--branch', type=str, default='main',
                       help='Branch name (default: main)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without doing it')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip running experiments, only push existing results')

    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    print(f"Repository: {repo_path}")

    # 验证仓库
    if not (repo_path / '.git').exists():
        print(f"[Error] Not a git repository: {repo_path}")
        sys.exit(1)

    # 生成提交信息
    if args.message:
        commit_msg = args.message
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"Experiment results: {timestamp}\n\n"
        if not args.skip_experiments:
            commit_msg += "Added: Academic experiment suite with 10 windows × 7 strategies × 3 trials\n"
            commit_msg += "Includes: oracle baseline, accurate cost calculation, SLO violation analysis\n"
            commit_msg += "Results in experiments/serverless_test/trace_experiment/final_results/\n"

    # Dry run 模式
    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        print(f"  Commit message: {commit_msg[:100]}...")
        print(f"  Remote: {args.remote}/{args.branch}")
        return

    # 确认
    print(f"\nProceed with push? (y/n): ", end='')
    resp = input().strip().lower()
    if resp != 'y':
        print("Aborted.")
        sys.exit(0)

    # 执行推送
    success = git_push(str(repo_path), commit_msg, args.remote, args.branch)

    if success:
        print("\n✓ GitHub push completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ GitHub push failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
