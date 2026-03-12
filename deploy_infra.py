import boto3
import json
import zipfile
import os
import io
import time

# Configuration
REGION = 'us-east-1'
ACCOUNT_ID = boto3.client('sts').get_caller_identity().get('Account')
ROLE_NAME = 'MPCLambdaRole'

# Lambda Names
MPC_FUNC_NAME = 'MPC_Controller'
WORKER_FUNC_NAME = 'MPC_BusinessWorker'
RECOVERY_WORKER_NAME = 'MPC_RecoveryWorker'

# Infrastructure Names
SFN_NAME = 'MPC_Layered_Defense_Workflow_Complex'
DYNAMODB_STATE_TABLE = 'MPC_State'
DYNAMODB_DATA_TABLE = 'MPC_CriticalData'

iam = boto3.client('iam', region_name=REGION)
lmb = boto3.client('lambda', region_name=REGION)
sfn = boto3.client('stepfunctions', region_name=REGION)
events = boto3.client('events', region_name=REGION)
dynamodb = boto3.client('dynamodb', region_name=REGION)

sqs = boto3.client('sqs', region_name=REGION)

def create_queue(queue_name, dlq_name=None):
    print(f"Checking SQS Queue: {queue_name}...")
    try:
        # Create DLQ first if requested
        dlq_arn = None
        if dlq_name:
            print(f"Checking DLQ: {dlq_name}...")
            try:
                sqs.create_queue(QueueName=dlq_name)
                resp = sqs.get_queue_url(QueueName=dlq_name)
                dlq_url = resp['QueueUrl']
                attrs = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=['QueueArn'])
                dlq_arn = attrs['Attributes']['QueueArn']
                print(f"  - DLQ Ready: {dlq_url}")
            except Exception as e:
                print(f"  - Error creating DLQ: {e}")

        attributes = {}
        if dlq_arn:
            attributes['RedrivePolicy'] = json.dumps({
                'deadLetterTargetArn': dlq_arn,
                'maxReceiveCount': '3'
            })

        sqs.create_queue(QueueName=queue_name, Attributes=attributes)
        resp = sqs.get_queue_url(QueueName=queue_name)
        print(f"  - Queue ready: {resp['QueueUrl']}")
        return resp['QueueUrl']
    except Exception as e:
        print(f"Error creating queue: {e}")
        # If queue exists with different attributes, try to update it
        if "QueueAlreadyExists" in str(e):
            try:
                resp = sqs.get_queue_url(QueueName=queue_name)
                url = resp['QueueUrl']
                print(f"  - Queue exists, updating attributes: {url}")
                if attributes:
                    sqs.set_queue_attributes(QueueUrl=url, Attributes=attributes)
                return url
            except Exception as e2:
                print(f"  - Error updating queue attributes: {e2}")
                return None
        return None

def create_role():
    print(f"Creating/Getting IAM Role: {ROLE_NAME}...")
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust_policy))
    except iam.exceptions.EntityAlreadyExistsException:
        pass

    # Custom Policy for Least Privilege
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:DescribeTable"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sqs:SendMessage",
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes",
                    "sqs:GetQueueUrl"
                ],
                "Resource": f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                    "xray:GetSamplingStatisticSummaries"
                ],
                "Resource": "*"
            }
            ,
            {
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutMetricData"
                ],
                "Resource": "*"
            }
        ]
    }

    # Remove AdministratorAccess if it exists (Cleanup)
    try:
        iam.detach_role_policy(RoleName=ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AdministratorAccess")
    except:
        pass

    # Put inline policy
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName='MPCLeastPrivilegePolicy',
        PolicyDocument=json.dumps(policy_doc)
    )
    
    time.sleep(20) # Wait for propagation
    return f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"

def create_table(table_name, key_name):
    print(f"Checking DynamoDB Table: {table_name}...")
    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{'AttributeName': key_name, 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': key_name, 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST'
        )
        print(f"  - Creating {table_name}...")
        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(TableName=table_name)
        print(f"  - {table_name} is active.")
    except dynamodb.exceptions.ResourceInUseException:
        print(f"  - {table_name} already exists.")

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
                # 过滤目录
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
                        # 过滤目录
                        dirs[:] = [d for d in dirs if not any(p in os.path.join(root, d) for p in ignore_patterns)]
                        
                        for file in files:
                            if not any(p in file for p in ignore_patterns):
                                full = os.path.join(root, file)
                                # 保持 extra_dirs 的顶层目录名
                                arc = os.path.join(os.path.basename(d), os.path.relpath(full, base))
                                z.write(full, arc)
    buf.seek(0)
    return buf.read()

def wait_for_function_update(func_name):
    """Waits for function update to complete"""
    print(f"    Waiting for {func_name} update to complete...")
    max_retries = 30
    for i in range(max_retries):
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
    print("    Warning: Timeout waiting for update.")

def deploy_lambda(func_name, folder, role_arn, runtime='python3.9', handler='lambda_function.lambda_handler', env_vars={}, extra_dirs=None):
    print(f"Deploying Lambda: {func_name} ({runtime})...")
    
    if extra_dirs is None:
        extra_dirs = [os.path.join(os.getcwd(), 'src')]
        
    zip_content = zip_function(folder, extra_dirs=extra_dirs)
    env_config = {'Variables': env_vars}
    tracing_config = {'Mode': 'Active'} # Enable X-Ray
    
    try:
        lmb.create_function(
            FunctionName=func_name,
            Runtime=runtime,
            Role=role_arn,
            Handler=handler,
            Code={'ZipFile': zip_content},
            Timeout=15,
            MemorySize=128,
            Environment=env_config,
            TracingConfig=tracing_config
        )
        print(f"  - Created {func_name}")
    except lmb.exceptions.ResourceConflictException:
        # Update Code
        lmb.update_function_code(
            FunctionName=func_name,
            ZipFile=zip_content
        )
        wait_for_function_update(func_name)
        
        # Update Config
        try:
            lmb.update_function_configuration(
                FunctionName=func_name,
                Environment=env_config,
                Runtime=runtime,
                Handler=handler,
                TracingConfig=tracing_config
            )
            wait_for_function_update(func_name)
            print(f"  - Updated {func_name}")
        except lmb.exceptions.ResourceConflictException as e:
             # Retry once if still conflicting
             print(f"    Conflict updating config, retrying... ({e})")
             time.sleep(5)
             lmb.update_function_configuration(
                FunctionName=func_name,
                Environment=env_config,
                Runtime=runtime,
                Handler=handler,
                TracingConfig=tracing_config
            )
             wait_for_function_update(func_name)
             print(f"  - Updated {func_name} (Retry)")

    return f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{func_name}"

def deploy_sfn(asl_file, replacements, role_arn):
    print(f"Deploying Step Function: {SFN_NAME}...")
    
    with open(asl_file, 'r') as f:
        definition_str = f.read()
    
    # Replace placeholders
    for key, value in replacements.items():
        # Support both ${KEY} and {{KEY}} formats
        definition_str = definition_str.replace(f"${{{key}}}", value)
        definition_str = definition_str.replace(f"{{{{{key}}}}}", value)
        
    # Validate JSON
    try:
        definition = json.loads(definition_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing ASL JSON: {e}")
        return

    # SFN Role
    sfn_role_name = 'MPCSfnRole'
    try:
        iam.create_role(
            RoleName=sfn_role_name,
            AssumeRolePolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Principal": {"Service": "states.amazonaws.com"}, "Action": "sts:AssumeRole"}]
            })
        )
    except iam.exceptions.EntityAlreadyExistsException:
        pass
        
    # SFN Policy
    sfn_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "lambda:InvokeFunction",
                "Resource": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:MPC_*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sqs:SendMessage",
                    "sqs:GetQueueUrl"
                ],
                "Resource": f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogDelivery",
                    "logs:GetLogDelivery",
                    "logs:UpdateLogDelivery",
                    "logs:DeleteLogDelivery",
                    "logs:ListLogDeliveries",
                    "logs:PutResourcePolicy",
                    "logs:DescribeResourcePolicies",
                    "logs:DescribeLogGroups"
                ],
                "Resource": "*"
            }
        ]
    }
    
    # Clean up old Admin access if present
    try:
        iam.detach_role_policy(RoleName=sfn_role_name, PolicyArn="arn:aws:iam::aws:policy/AdministratorAccess")
    except:
        pass
        
    iam.put_role_policy(
        RoleName=sfn_role_name,
        PolicyName='MPCSfnPolicy',
        PolicyDocument=json.dumps(sfn_policy)
    )
    
    time.sleep(2)
    sfn_role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{sfn_role_name}"

    try:
        sfn.create_state_machine(
            name=SFN_NAME,
            definition=json.dumps(definition),
            roleArn=sfn_role_arn,
            type='STANDARD',
            tracingConfiguration={'enabled': True}
        )
        print("  - Created State Machine")
    except sfn.exceptions.StateMachineAlreadyExists:
        # Get ARN
        resp = sfn.list_state_machines()
        sfn_arn = next(item['stateMachineArn'] for item in resp['stateMachines'] if item['name'] == SFN_NAME)
        
        sfn.update_state_machine(
            stateMachineArn=sfn_arn,
            definition=json.dumps(definition),
            roleArn=sfn_role_arn,
            tracingConfiguration={'enabled': True}
        )
        print("  - Updated State Machine")

def deploy_event_rule(sfn_arn):
    rule_name = 'MPC_Recovery_Schedule'
    print(f"Deploying EventBridge Rule: {rule_name}...")
    
    try:
        events.put_rule(
            Name=rule_name,
            ScheduleExpression='rate(5 minutes)',
            State='ENABLED'
        )
        
        # Target: Recovery Worker Lambda directly (simulating cron job)
        # But wait, in the architecture, recovery worker is a Lambda, usually triggered by EventBridge.
        # Let's target the Lambda directly.
        target_arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{RECOVERY_WORKER_NAME}"
        
        events.put_targets(
            Rule=rule_name,
            Targets=[{
                'Id': 'RecoveryWorkerTarget',
                'Arn': target_arn
            }]
        )
        
        # Permission for EventBridge to invoke Lambda
        try:
            lmb.add_permission(
                FunctionName=RECOVERY_WORKER_NAME,
                StatementId='EventBridgeInvoke',
                Action='lambda:InvokeFunction',
                Principal='events.amazonaws.com',
                SourceArn=f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{rule_name}"
            )
        except lmb.exceptions.ResourceConflictException:
            pass
            
    except Exception as e:
        print(f"Error deploying EventBridge rule: {e}")

if __name__ == '__main__':
    # 1. IAM & Infra
    role_arn = create_role()
    create_table(DYNAMODB_STATE_TABLE, 'id')
    create_table(DYNAMODB_DATA_TABLE, 'requestId')
    recovery_queue_url = create_queue('RecoveryQueue', dlq_name='RecoveryDLQ')
    
    # 2. Deploy Lambdas
    cwd = os.getcwd()
    
    # MPC Controller 只需核心逻辑
    mpc_arn = deploy_lambda(MPC_FUNC_NAME, os.path.join(cwd, 'lambdas', 'mpc_controller'), role_arn)
    
    # Business Worker 需要 benchmarks 进行真实负载运行
    worker_extra = [
        os.path.join(cwd, 'src'),
        os.path.join(cwd, 'benchmarks', 'function_bench')
    ]
    worker_arn = deploy_lambda(WORKER_FUNC_NAME, os.path.join(cwd, 'lambdas', 'business_worker'), role_arn, extra_dirs=worker_extra)
    
    # Recovery Worker 只需核心逻辑
    recovery_arn = deploy_lambda(RECOVERY_WORKER_NAME, os.path.join(cwd, 'lambdas', 'recovery_worker'), role_arn, 
                                 env_vars={'QUEUE_URL': recovery_queue_url, 'TABLE_NAME': DYNAMODB_DATA_TABLE})
    
    # 3. Deploy Step Function
    replacements = {
        'MPC_CONTROLLER_ARN': mpc_arn,
        'BUSINESS_WORKER_ARN': worker_arn,
        'DYNAMODB_TABLE_NAME': DYNAMODB_DATA_TABLE,
        'RECOVERY_QUEUE_URL': recovery_queue_url,
        'RECOVERY_WORKER_ARN': recovery_arn
    }
    
    # Using the Integrated DAG (normalized path)
    asl_path = os.path.join(cwd, 'serverless', 'workflows', 'integrated.asl.json')
    deploy_sfn(asl_path, replacements, role_arn)
    
    # 4. EventBridge
    # We are not deploying the rule pointing to SFN, but to the worker directly for now as per previous logic
    deploy_event_rule(None) 
    
    print("Deployment Complete!")
    print(f"SFN ARN: arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{SFN_NAME}")
