provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Name    = "k3s"
      purpose = "k3s-sandbox-cluster"
    }
  }
}
