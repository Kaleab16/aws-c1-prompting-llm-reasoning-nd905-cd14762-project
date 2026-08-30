import boto3
import time

REGION = "us-east-1"

BUCKET = "udacity-agentic-engineer-c1-eval-888494046502"
DATASET_KEY = "output_eval_dataset.jsonl"

EVAL_ROLE_ARN = (
    "arn:aws:iam::888494046502:role/bedrock-eval-role"
)

bedrock = boto3.client(
    "bedrock",
    region_name=REGION
)


def main():
    print("Starting Bedrock evaluation...")

    response = bedrock.create_evaluation_job(
        jobName=f"support-chatbot-evaluation-{int(time.time())}",

        jobDescription=(
            "LLM-as-a-judge evaluation for customer support chatbot"
        ),

        roleArn=EVAL_ROLE_ARN,

        applicationType="ModelEvaluation",

        evaluationConfig={
            "automated": {
                "datasetMetricConfigs": [
                    {
                        "taskType": "General",

                        "dataset": {
                            "name": "custom-eval-dataset",

                            "datasetLocation": {
                                "s3Uri": (
                                    f"s3://{BUCKET}/{DATASET_KEY}"
                                )
                            }
                        },

                        "metricNames": [
                            "Builtin.Correctness",
                            "Builtin.Completeness",
                            "Builtin.Helpfulness",
                            "Builtin.FollowingInstructions"
                        ]
                    }
                ],

                "evaluatorModelConfig": {
                    "bedrockEvaluatorModels": [
                        {
                            "modelIdentifier": (
                                "amazon.nova-pro-v1:0"
                            )
                        }
                    ]
                }
            }
        },

        inferenceConfig={
            "models": [
                {
                    "precomputedInferenceSource": {
                        "inferenceSourceIdentifier": (
                            "my-support-chatbot"
                        )
                    }
                }
            ]
        },

        outputDataConfig={
            "s3Uri": (
                f"s3://{BUCKET}/evaluation-results/"
            )
        }
    )

    job_arn = response["jobArn"]

    print("\nEvaluation job created successfully.")
    print(f"Job ARN: {job_arn}")

    while True:
        result = bedrock.get_evaluation_job(
            jobIdentifier=job_arn
        )

        status = result["status"]

        print(f"Status: {status}")

        if status in (
            "Completed",
            "Failed",
            "Stopped"
        ):
            break

        time.sleep(15)

    if status != "Completed":
        raise RuntimeError(
            f"Evaluation ended with status: {status}"
        )

    print("\nEvaluation completed successfully.")

    print(
        f"Results: "
        f"s3://{BUCKET}/evaluation-results/"
    )


if __name__ == "__main__":
    main()