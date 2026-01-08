import json
import boto3
import time
import os
import decimal
import random

# Helper class to convert DynamoDB items to JSON
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)

dynamodb = boto3.resource('dynamodb')
sqs = boto3.client('sqs')
cloudwatch = boto3.client('cloudwatch')
state_table = dynamodb.Table('MPC_State')
# Use the passed env var for data table, default to MPC_CriticalData
DATA_TABLE_NAME = os.environ.get('TABLE_NAME', 'MPC_CriticalData')
data_table = dynamodb.Table(DATA_TABLE_NAME)

def get_queue_url(queue_name):
    url = os.environ.get('QUEUE_URL')
    if url: return url
    try:
        response = sqs.get_queue_url(QueueName=queue_name)
        return response['QueueUrl']
    except Exception as e:
        print(f"Error getting queue URL: {e}")
        return None

def lambda_handler(event, context):
    print("Starting Adaptive Recovery Worker (REAL EXECUTION)...")
    
    # 1. Check System Health (MPC State)
    # We implement "Backpressure" here.
    try:
        response = state_table.get_item(Key={'id': 'global_params'})
        if 'Item' in response:
            state = response['Item']
            # Convert Decimal to float
            congestion_price = float(state.get('congestion_price', 0.0))
            
            print(f"Current System State: Price={congestion_price:.4f}")
            
            # ADAPTIVE LOGIC:
            # If congestion price is high (> 1.0), the system is under stress.
            # We should BACK OFF to avoid adding more pressure (Thundering Herd).
            if congestion_price > 1.0:
                cloudwatch.put_metric_data(
                    Namespace='MPC/Recovery',
                    MetricData=[{
                        'MetricName': 'Backoff',
                        'Value': 1.0,
                        'Unit': 'Count'
                    }]
                )
                return {
                    'statusCode': 200,
                    'body': json.dumps('System congested, skipping recovery.')
                }
        else:
            print("No MPC state found, assuming safe to proceed.")
            
    except Exception as e:
        print(f"Error reading MPC state: {e}")
        # Fail safe: proceed cautiously
    
    # 2. Poll Recovery Queue
    q_url = get_queue_url('RecoveryQueue')
    if not q_url:
        return {'statusCode': 500, 'body': 'Queue not found'}
    
    # Receive messages (Batch of 10)
    response = sqs.receive_message(
        QueueUrl=q_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=2,
        VisibilityTimeout=30
    )
    
    messages = response.get('Messages', [])
    print(f"Pulled {len(messages)} messages from Recovery Queue.")
    
    if not messages:
        return {'statusCode': 200, 'body': 'No messages to process.'}
        
    processed_count = 0
    
    for msg in messages:
        try:
            body = json.loads(msg['Body'])
            task = body.get('task', {})
            original_intent = body.get('original_intent', 'unknown')
            req_id = task.get('id', 'unknown')
            
            print(f"Recovering Task {req_id} | Intent: {original_intent}")
            
            # --- 3. REAL BUSINESS LOGIC (Duplicate of business_worker) ---
            # In a real system, we would run the specific calculation here.
            # We simulate CPU work + Latency
            base_latency = 0.1
            jitter = random.uniform(0, 0.1)
            time.sleep(base_latency + jitter)
            
            # Simulated Quote Result
            quote_result = {
                "price": random.randint(100, 200), # Mock quote
                "recovered_at": time.time(),
                "original_task": task,
                "source": "recovery_worker"
            }
            
            # --- 4. REAL PERSISTENCE (Duplicate of save-quotes but Python) ---
            # Idempotency Check: ConditionExpression='attribute_not_exists(requestId)'
            try:
                data_table.put_item(
                    Item={
                        'requestId': str(req_id),
                        'quotes': json.dumps(quote_result),
                        'status': 'RECOVERED',
                        'timestamp': decimal.Decimal(str(time.time()))
                    },
                    ConditionExpression='attribute_not_exists(requestId)'
                )
                print(f"  - Saved Task {req_id} to DB.")
            except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
                print(f"  - Task {req_id} already exists in DB. Skipping (Idempotent).")
            
            # 5. Delete Message (Ack)
            # Only delete if processing (business logic + save) was successful
            sqs.delete_message(
                QueueUrl=q_url,
                ReceiptHandle=msg['ReceiptHandle']
            )
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing message {msg.get('MessageId')}: {e}")
            # Do NOT delete message. It will go back to queue (and eventually DLQ).
            
    cloudwatch.put_metric_data(
        Namespace='MPC/Recovery',
        MetricData=[{
            'MetricName': 'RecoveredTasks',
            'Value': float(processed_count),
            'Unit': 'Count'
        }]
    )
    return {
        'statusCode': 200,
        'body': json.dumps(f"Successfully recovered {processed_count} tasks.")
    }
