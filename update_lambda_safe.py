
import boto3
import os
import zipfile
import io
import time

# Configuration
REGION = 'us-east-1'
MPC_FUNC_NAME = 'MPC_Controller'

lmb = boto3.client('lambda', region_name=REGION)

def zip_function(folder_path, extra_dirs=None):
    print("Preparing deployment package in memory...")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        # 1. Add Lambda Handler Code
        if os.path.isdir(folder_path):
            base = folder_path
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file != 'function.zip':
                        full = os.path.join(root, file)
                        arc = os.path.relpath(full, base)
                        z.write(full, arc)
                        # print(f"  - Added {arc}")
        
        # 2. Add Shared Source Code (src/)
        if extra_dirs:
            for d in extra_dirs:
                if os.path.isdir(d):
                    base = d
                    for root, dirs, files in os.walk(d):
                        for file in files:
                            # Skip __pycache__
                            if '__pycache__' in root: continue
                            
                            full = os.path.join(root, file)
                            # Ensure it goes into 'src/' folder in zip
                            arc = os.path.join('src', os.path.relpath(full, base))
                            z.write(full, arc)
                            # print(f"  - Added {arc}")
    
    buf.seek(0)
    print(f"Package created. Size: {len(buf.getvalue()) / 1024:.2f} KB")
    return buf.read()

def wait_for_update(func_name):
    print(f"Waiting for {func_name} update to complete...")
    for i in range(30):
        try:
            resp = lmb.get_function(FunctionName=func_name)
            status = resp['Configuration']['LastUpdateStatus']
            if status == 'Successful':
                print("Update Successful.")
                return
            elif status == 'Failed':
                reason = resp['Configuration'].get('LastUpdateStatusReason', 'Unknown')
                raise Exception(f"Update Failed: {reason}")
        except Exception as e:
            print(f"  Check status error: {e}")
        time.sleep(1)

def update_mpc_lambda():
    print(f"Updating Lambda: {MPC_FUNC_NAME}...")
    
    # Paths
    lambda_dir = os.path.join(os.getcwd(), 'lambdas', 'mpc_controller')
    src_dir = os.path.join(os.getcwd(), 'src')
    
    # Create Zip
    zip_content = zip_function(lambda_dir, extra_dirs=[src_dir])
    
    # Update Function Code
    print("Uploading to AWS...")
    try:
        lmb.update_function_code(
            FunctionName=MPC_FUNC_NAME,
            ZipFile=zip_content
        )
        wait_for_update(MPC_FUNC_NAME)
        print("Done!")
    except Exception as e:
        print(f"Update failed: {e}")

if __name__ == "__main__":
    update_mpc_lambda()
