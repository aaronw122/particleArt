variable "project_name" {
  description = "Project name for resource naming and tagging."
  type        = string
  default     = "prtkl"
}

variable "key_pair_name" {
  description = "Existing EC2 key pair name for SSH access."
  type        = string
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type."
  type        = string
  default     = "t3.micro"
}

variable "app_port" {
  description = "Port the application listens on."
  type        = number
  default     = 3002
}
