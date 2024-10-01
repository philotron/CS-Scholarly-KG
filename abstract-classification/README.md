The data is taken from the following publication: 
Gonçalves, S., Cortez, P. & Moro, S. (2020): A deep learning classifier for sentence classification in biomedical and computer science abstracts.

- The model used is SciBERT, a BERT model pre-trained on academic texts. Link: https://huggingface.co/allenai/scibert_scivocab_uncased
- The model takes sentences of academic abstracts (field: Computer Science) as input and classifies them to one of the following 5 categories: Objective, Background, Methods, Results, Conclusions 
- There is manually labeled training data available from the publication: 3287 train, 824 val, 619 test samples
- An extensive hyperparameter grid search was conducted with the follwong options:
    - learning rate in [2e-5, 3e-5, 4e-5]
    - weight decay in [0, 0.08, 0.18]
    - warmup steps in [0, 100]
    - batch size in [16, 32, 64]
    - dropout in [0.1, 0.25]
- Finally the best hyperparameter combination was:
    - batch size: 32
    - dropout: 0.25
    - warmup steps: 100
    - weight decay: 0.08
    - learning rate: 3e-05
    - test accuracy: 0.7592892
    - test f1 score: 0.7538552
    - --> Exceeded the F1 score of the original publication about 1% (from 74.60%)