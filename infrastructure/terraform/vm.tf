module "ec2_instances" {
  source  = "terraform-aws-modules/ec2-instance/aws"
  version = "5.6.1" # Specify the version that suits your needs

  for_each = toset(["master"]) # can also add 'worker'
  name     = "k3s-${each.key}"

  ami                    = var.ami_id # Valid AMI for your region
  instance_type          = "t3.medium"
  key_name               = "k3s-machines"               # The key created on AWS in EC2 service section; chmod 400 ~/.ssh/<KEY>.pem
  subnet_id              = module.vpc.public_subnets[0] # Using the first public subnet
  vpc_security_group_ids = [aws_security_group.allow_ssh_and_ports.id, module.vpc.default_security_group_id]

  associate_public_ip_address = true

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}
