# A tutorial on Amazon SageMaker
This is a tutorial on Amazon SageMaker to easily train and deploy your ML models. It's based on the work in the 3days internship at AWS, which is in collaboration with [@uidilr](https://github.com/uidilr).

We suppose readers have experiences of ML and are unfamiliar with Amazon SageMaker. We also write [an article](https://qiita.com/ku2482/private/67f894b7778dbdd84b6a) for Japanese. Please reffer to it.

## Installation
```
# Install dependencies.
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# Download the pretrained model of Sentence-BERT.
wget https://watanabe-storage.s3-us-west-2.amazonaws.com/dataset/modules.pickle -P source_dir
```

## Usage
To train and deploy an example of ML models, see `deploy_example.ipynb`.
