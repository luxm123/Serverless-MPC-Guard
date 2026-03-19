import boto3
import zipfile
import os
import io
import time

# Configuration
REGION = 'us-east-1'
WORKER_FUNC_NAME = 'MPC_BusinessWorker'

lmb = boto3.client('lambda', region_name=REGION)

def zip_function(folder_path, extra_dirs=None, ignore_patterns=None):
    if ignore_patterns is None:
        ignore_patterns = [
            '.git', '__pycache__', 'node_modules', '.ipynb_checkpoints',
            'dataset/amzn_fine_food_reviews', 'dataset/file', 'dataset/video',
            'azure', 'google', 'openwhisk', 'docs', 'clusterdata'
        ]
        
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        if os.path.isdir(folder_path):
            base = folder_path
            for root, dirs, files in os.walk(folder_path):
                dirs[:] = [d for d in dirs if not any(p in os.path.join(root, d) for p in ignore_patterns)]
                for file in files:
                    if file != 'function.zip' and not any(p in file for p in ignore_patterns):
                        full = os.path.join(root, file)
                        arc = os.path.relpath(full, base)
                        z.write(full, arc)
                        
        if extra_dirs:
            for d in extra_dirs:
                if os.path.isdir(d):
                    base = d
                    for root, dirs, files in os.walk(d):
                        dirs[:] = [d for d in dirs if not any(p in os.path.join(root, d) for p in ignore_patterns)]
                        for file in files:
                            if not any(p in file for p in ignore_patterns):
                                full = os.path.join(root, file)
                                arc = os.path.join(os.path.basename(d), os.path.relpath(full, base))
                                z.write(full, arc)
    buf.seek(0)
    return buf.read()

def wait_for_function_update(func_name):
    print(f"    Waiting for {func_name} update to complete...")
    for i in range(30):
        try:
            resp = lmb.get_function(FunctionName=func_name)
            status = resp['Configuration']['LastUpdateStatus']
            if status == 'Successful':
                return
            elif status == 'Failed':
                raise Exception(f"Function update failed: {resp['Configuration']['LastUpdateStatusReason']}")
        except Exception as e:
            print(f"    Error checking status: {e}")
        time.sleep(1)

if __name__ == '__main__':
    cwd = os.getcwd()
    folder = os.path.join(cwd, 'lambdas', 'business_worker')
    extra_dirs = [
        os.path.join(cwd, 'src'),
        os.path.join(cwd, 'benchmarks', 'function_bench')
    ]
    
    print(f"Updating code for {WORKER_FUNC_NAME}...")
    zip_content = zip_function(folder, extra_dirs=extra_dirs)
    
    lmb.update_function_code(
        FunctionName=WORKER_FUNC_NAME,
        ZipFile=zip_content
    )
    wait_for_function_update(WORKER_FUNC_NAME)
    print(f"Successfully updated {WORKER_FUNC_NAME}!")
