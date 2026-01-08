import boto3
import os
import zipfile
import io
import time

REGION = 'us-east-1'
MPC_FUNC_NAME = 'MPC_Controller'

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

def update_lambda():
    print(f"Updating Lambda Code: {MPC_FUNC_NAME}...")
    
    # Path to lambda handler
    lambda_dir = os.path.join(os.getcwd(), 'lambdas', 'mpc_controller')
    # Path to src
    src_dir = os.path.join(os.getcwd(), 'src')
    
    zip_content = zip_function(lambda_dir, extra_dirs=[src_dir])
    
    try:
        lmb.update_function_code(
            FunctionName=MPC_FUNC_NAME,
            ZipFile=zip_content
        )
        print("Update initiated. Waiting for status...")
        
        # Wait for update
        for i in range(30):
            resp = lmb.get_function(FunctionName=MPC_FUNC_NAME)
            status = resp['Configuration']['LastUpdateStatus']
            if status == 'Successful':
                print("Update Successful!")
                return
            elif status == 'Failed':
                print(f"Update Failed: {resp['Configuration']['LastUpdateStatusReason']}")
                return
            time.sleep(1)
            
    except Exception as e:
        print(f"Error updating lambda: {e}")

if __name__ == '__main__':
    update_lambda()
