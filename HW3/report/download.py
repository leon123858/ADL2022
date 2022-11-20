import gdown

data_url = "https://drive.google.com/u/0/uc?id=19b-RnXuAFhXhEHW7ah0M-pQsxHr1cPsZ&export=download"
data_output = "data.zip"

gdown.download(data_url, data_output)

model_url = "https://drive.google.com/u/0/uc?id=19bFqQrNq0e2a4REZC1ryMvCfnPpNBh9y&export=download"
model_output = "model.zip"

gdown.download(model_url, model_output)
