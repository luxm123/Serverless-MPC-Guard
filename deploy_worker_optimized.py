import os
import shutil
import zipfile
import boto3

def deploy_worker():
    print("Creating deployment package for Optimized Worker...")
    
    # Paths
    base_dir = os.getcwd()
    worker_src = os.path.join(base_dir, 'lambdas', 'business_worker', 'lambda_function.py')
    src_dir = os.path.join(base_dir, 'src')
    package_dir = os.path.join(base_dir, 'worker_package_tmp')
    zip_path = os.path.join(base_dir, 'worker_optimized.zip')
    
    # Clean up previous
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    os.makedirs(package_dir)
    
    # 1. Copy lambda_function.py
    shutil.copy(worker_src, package_dir)
    
    # 2. Copy src/ directory
    # Destination: package_dir/src
    shutil.copytree(src_dir, os.path.join(package_dir, 'src'))
    
    # 3. Zip it
    print("Zipping package...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Archive name should be relative to package_dir
                arcname = os.path.relpath(file_path, package_dir)
                zipf.write(file_path, arcname)
                
    # 4. Upload to AWS
    print("Uploading to AWS Lambda (MPC_BusinessWorker)...")
    lambda_client = boto3.client('lambda', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
    
    try:
        with open(zip_path, 'rb') as f:
            zipped_code = f.read()
            
        resp = lambda_client.update_function_code(
            FunctionName='MPC_BusinessWorker',
            ZipFile=zipped_code
        )
        print(f"Update Success: {resp['FunctionArn']}")
        
    except Exception as e:
        print(f"Update Failed: {e}")
        
    # Cleanup
    # shutil.rmtree(package_dir)
    # os.remove(zip_path)

if __name__ == "__main__":
    deploy_worker()
