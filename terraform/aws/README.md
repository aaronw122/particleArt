# prtkl — AWS Terraform Config

Alternative deployment target. Provisions a t3.micro EC2 instance with Docker pre-installed.

## Prerequisites

- AWS CLI configured (`aws configure`)
- EC2 key pair created in the target region

## Deploy

```bash
cd terraform/aws
terraform init
terraform apply -var="key_pair_name=your-key-name"
```

## After provisioning

SSH in and run `deploy.sh`, or rsync + docker compose manually:

```bash
ssh ec2-user@$(terraform output -raw instance_public_ip)
```

Estimated cost: ~$8.50/month (t3.micro on-demand, us-east-1).
