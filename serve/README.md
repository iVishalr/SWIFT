# SWIFT Serve

This folder consists of all the files required for serving model for inference.

## Steps to setup SWIFT inference with TorchServe

_Note that all the below steps needs to be executed from the project's root directory._

1. Install TorchServe and its dependencies

Using python-pip,

```bash
pip install torchserve torch-model-archiver torch-workflow-archiver
```

Using conda,

```bash
conda install -c pytorch torchserve torch-model-archiver torch-workflow-archiver
```

2. Creating `swift.mar` file

Type the following commands in terminal.

```bash
# create models.zip file
zip -r serve/models.zip model/ model_zoo/

# create model_store in serve/
mkdir serve/model_store

# create swift.mar using torch-model-archiver
torch-model-archiver --model-name swift --version 1.0 --model-file model/SWIFT.py --handler serve/handler.py --extra-files serve/models.zip

# move swift.mar to serve/model_store/
mv swift.mar serve/model_store/
```

3. Serving SWIFT through TorchServe

```bash
# Start TorchServe
torchserve --start --model-store serve/model_store/ --models swift=swift.mar --ts-config serve/config/config.properties --ncs

# Stop TorchServe
torchserve --stop
```

4. Making predictions (In development)

```bash
python3 serve/test_serve.py
```

`serve/test_serve.py` reads an image from a file, packages it to adhere to TorchServe API and makes a request to TorchServe for predictions.
