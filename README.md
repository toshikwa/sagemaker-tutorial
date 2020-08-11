# A tutorial on Amazon SageMaker

This is a tutorial on Amazon SageMaker to easily train and deploy your ML models. It's based on the work in the 3days internship at AWS Japan, which is in collaboration with [@uidilr](https://github.com/uidilr). We've also written an article in Japanese [here](https://qiita.com/ku2482/items/af187b427d941dee7ca0), so please refer to it.

## Installation
```bash
# Install dependencies.
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# Download the pretrained model of Sentence-BERT.
wget https://watanabe-storage.s3-us-west-2.amazonaws.com/dataset/modules.pickle -P source_dir
```

## Usage
To train and deploy an example of ML models, see [deploy_example.ipynb](https://github.com/ku2482/sagemaker-tutorial/blob/master/deploy_example.ipynb).
