import gdown

url = "https://drive.google.com/u/0/uc?id=19S3Nk2O6X2MiuZEWuG4Onvtq53mpWLdp&export=download"
output = "data.zip"
gdown.download(url, output)
