from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from nltk.tokenize import sent_tokenize


# tokenizing function
def preprocess_function_batch(tokenizer, examples):
    return tokenizer(
        examples["sentence"], 
        truncation=True,
        padding=True,
        max_length=512,
        #add_special_tokens=True,
        return_tensors="pt"
    )

# inference function
def classify_abstract_sentences(id: str, abstract: str) -> dict:
    
    #load tokenizer and fine-tuned model
    model_checkpoint = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model_checkpoint = "scibert-finetuned-abstract-classification/best_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=5)
    model.to("cuda")

    #generate single sentences from abstract
    abstract_sentences = sent_tokenize(abstract)
    
    #generate tokenized dataset from abstract
    test_data = [
        {
            "sentence": sentence
        }
        for sentence in abstract_sentences
    ]
    test_dataset = Dataset.from_list(test_data)
    encoded_test_data = preprocess_function_batch(tokenizer, test_dataset).to("cuda")

    #infer from model
    output = model(**encoded_test_data)
    predictions = output.logits.argmax(-1)

    #concatenate sentences of each class 
    classified_sents = ["", "", "", "", ""]
    for ix, sentence in enumerate(abstract_sentences):
        classified_sents[predictions[ix]] += sentence + " "

    #add "none" to empty classes & remove final space
    for ix, abstract_class in enumerate(classified_sents):
        if len(abstract_class) == 0:
            classified_sents[ix] = "none"
        else:
            classified_sents[ix] = abstract_class[:-1]

    #generate output dict
    output_dict = {
        "id": id,
        "background": classified_sents[0],
        "objective": classified_sents[1],
        "methods": classified_sents[2],
        "results": classified_sents[3],
        "conclusions": classified_sents[4]
    }
    return output_dict

if __name__ == "__main__":
    #testing the inference method
    abstract1 = """Recent deep learning models can efficiently combine inputs from different modalities (e.g., images and text) and learn to align their latent representations, or to translate signals from one domain to another (as in image captioning, or text-to-image generation). However, current approaches mainly rely on brute-force supervised training over large multimodal datasets. In contrast, humans (and other animals) can learn useful multimodal representations from only sparse experience with matched cross-modal data. Here we evaluate the capabilities of a neural network architecture inspired by the cognitive notion of a "Global Workspace": a shared representation for two (or more) input modalities. Each modality is processed by a specialized system (pretrained on unimodal data, and subsequently frozen). The corresponding latent representations are then encoded to and decoded from a single shared workspace. Importantly, this architecture is amenable to self-supervised training via cycle-consistency: encoding-decoding sequences should approximate the identity function. For various pairings of vision-language modalities and across two datasets of varying complexity, we show that such an architecture can be trained to align and translate between two modalities with very little need for matched data (from 4 to 7 times less than a fully supervised approach). The global workspace representation can be used advantageously for downstream classification tasks and for robust transfer learning. Ablation studies reveal that both the shared workspace and the self-supervised cycle-consistency training are critical to the system's performance."""
    abstract2 = """Multimodal Sentiment Analysis leverages multimodal signals to detect the sentiment of a speaker. Previous approaches concentrate on performing multimodal fusion and representation learning based on general knowledge obtained from pretrained models, which neglects the effect of domain-specific knowledge. In this paper, we propose Contrastive Knowledge Injection (ConKI) for multimodal sentiment analysis, where specific-knowledge representations for each modality can be learned together with general knowledge representations via knowledge injection based on an adapter architecture. In addition, ConKI uses a hierarchical contrastive learning procedure performed between knowledge types within every single modality, across modalities within each sample, and across samples to facilitate the effective learning of the proposed representations, hence improving multimodal sentiment predictions. The experiments on three popular multimodal sentiment analysis benchmarks show that ConKI outperforms all prior methods on a variety of performance metrics."""
    output_dict1 = classify_abstract_sentences("id_12418934577", abstract1)
    output_dict2 = classify_abstract_sentences("id_12418973423", abstract2)
    print(output_dict1)
    print(output_dict2)