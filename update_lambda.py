import boto3
import os
import zipfile
import io
import time

# CRITICAL FIX: Do not hardcode region. Use environment or config.
# If running on EC2 eu-north-1, this must be eu-north-1.
session = boto3.Session()
REGION = session.region_name or 'us-east-1'
print(f"Deploying to Region: {REGION}")

MPC_FUNC_NAME = 'MPC_Controller'
WORKER_FUNC_NAME = 'MPC_BusinessWorker'

lmb = boto3.client('lambda', region_name=REGION)

def zip_function(folder, extra_dirs=[]):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        # Add function files
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.py') or file.endswith('.json'):
                    full = os.path.join(root, file)
                    arc = os.path.relpath(full, folder)
                    z.write(full, arc)
        
        # Add extra directories (e.g., src/)
        for d in extra_dirs:
            base = os.path.dirname(d)
            for root, dirs, files in os.walk(d):
                for file in files:
                    if file.endswith('.py') or file.endswith('.json'):
                        full = os.path.join(root, file)
                        # Archive path should include the directory name (e.g., src/mpc/...)
                        # If d is .../src, we want arc to be src/...
                        # os.path.relpath(full, base) will give src/...
                        arc = os.path.relpath(full, base)
                        z.write(full, arc)
    buf.seek(0)
    return buf.read()

def update_function(func_name, lambda_folder):
    print(f"Updating Lambda Code: {func_name}...")
    
    # Path to lambda handler
    lambda_dir = os.path.join(os.getcwd(), 'lambdas', lambda_folder)
    # Path to src
    src_dir = os.path.join(os.getcwd(), 'src')
    
    zip_content = zip_function(lambda_dir, extra_dirs=[src_dir])
    
    try:
        lmb.update_function_code(
            FunctionName=func_name,
            ZipFile=zip_content
        )
        print(f"Update for {func_name} initiated. Waiting for status...")
        
        # Wait for update
        for i in range(30):
            resp = lmb.get_function(FunctionName=func_name)
            status = resp['Configuration']['LastUpdateStatus']
            if status == 'Successful':
                print(f"Update for {func_name} Successful!")
                return
            elif status == 'Failed':
                print(f"Update for {func_name} Failed: {resp['Configuration']['LastUpdateStatusReason']}")
                return
            time.sleep(1)
            
    except Exception as e:
        print(f"Error updating lambda {func_name}: {e}")

def update_function_config(func_name, memory_size=1024):
    print(f"Updating Lambda Config: {func_name} (Memory: {memory_size}MB)...")
    try:
        lmb.update_function_configuration(
            FunctionName=func_name,
            MemorySize=memory_size,
            Timeout=60  # Ensure timeout is sufficient
        )
        print(f"Config update for {func_name} initiated.")
    except Exception as e:
        print(f"Error updating config for {func_name}: {e}")

def update_lambda():
    update_function(MPC_FUNC_NAME, 'mpc_controller')
    update_function_config(MPC_FUNC_NAME, memory_size=1024)
    
    update_function(WORKER_FUNC_NAME, 'business_worker')
    update_function_config(WORKER_FUNC_NAME, memory_size=1024)

if __name__ == '__main__':
    update_lambda()
