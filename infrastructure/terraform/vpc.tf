module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.7.1"

  name = "k3s-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["eu-central-1a", "eu-central-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  default_security_group_ingress = [
    {
      from_port   = 0,
      to_port     = 0,
      protocol    = "-1",
      cidr_blocks = "10.0.0.0/16",
      description = "Allow all inbound traffic from within the VPC"
    }
  ]

  default_security_group_egress = [
    {
      from_port   = 0,
      to_port     = 0,
      protocol    = "-1",
      cidr_blocks = "0.0.0.0/0",
      description = "Allow all outbound traffic"
    }
  ]

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }

  default_security_group_name = "k3s-default-vpc-sg"
  default_security_group_tags = {
    Name = "k3s-default-vpc-sg"
  }
}

resource "aws_security_group" "allow_ssh_and_ports" {
  name        = "allow_ssh_and_ports"
  description = "Allow SSH and specific port range inbound traffic"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # ingress {
  #   from_port   = 80
  #   to_port     = 443
  #   protocol    = "tcp"
  #   cidr_blocks = ["0.0.0.0/0"]
  # }

  # kube-api server port
  ingress {
    from_port   = 6443
    to_port     = 6443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # K8s ingress rule for ports 30000-40000
  ingress {
    from_port   = 30000
    to_port     = 40000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow outbound traffic for all ports
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "allow_ssh_and_ports"
  }
}
