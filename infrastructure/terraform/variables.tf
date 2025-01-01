variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "ami_id" {
  type    = string
  default = "ami-0cf42ae9a371140c8" # 22.04 eu-central-1; https://cloud-images.ubuntu.com/locator/ec2/
}
