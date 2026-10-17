import torch
from evaluate import load
from collections import defaultdict


def ngrams(tokens, n):
    return zip(*(tokens[i:] for i in range(n)))
import numpy as np
from collections import Counter

def compute_perplexity(all_texts_list, model_id='gpt2-large', device='cuda'):
    torch.cuda.empty_cache() 
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=all_texts_list, model_id=model_id, device=device)
    return results['mean_perplexity']

def compute_wordcount(all_texts_list):
    wordcount = load("word_count")
    wordcount = wordcount.compute(data=all_texts_list)
    return wordcount['unique_words']

def compute_diversity(all_texts_list):
    ngram_range = [2,3,4]

    try:
        import spacy
        tokenizer = spacy.load("en_core_web_sm").tokenizer
        tokenize = lambda text: [str(token) for token in tokenizer(text)]
    except (ImportError, OSError):
        tokenize = lambda text: text.split()
    token_list = []
    for sentence in all_texts_list:
        token_list.append(tokenize(sentence))
    ngram_sets = {}
    ngram_counts = defaultdict(int)
    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = (1-len(ngram_sets[n])/ngram_counts[n]) if ngram_counts[n] else 0.0
    diversity = 1
    for val in metrics.values():
        diversity *= (1-val)
    metrics['diversity'] = diversity
    return metrics

def compute_topk_ngrams(all_texts_list, k=10):
    ngram_range = [2,3,4]

    try:
        import spacy
        tokenizer = spacy.load("en_core_web_sm").tokenizer
        tokenize = lambda text: [str(token) for token in tokenizer(text)]
    except (ImportError, OSError):
        tokenize = lambda text: text.split()
    token_list = []
    for sentence in all_texts_list:
        token_list.append(tokenize(sentence))

    # --- start of code (uses nltk.util.ngrams) ---
    from collections import Counter

    topk_by_n = {}
    for n in ngram_range:
        counts = Counter()
        for tokens in token_list:
            if len(tokens) >= n:
                counts.update(ngrams(tokens, n))
        topk_by_n[n] = counts.most_common(k)

    return topk_by_n

def compute_memorization(all_texts_list, human_references, n=4):

    import spacy
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    unique_four_grams = set()
    for sentence in human_references:
        unique_four_grams.update(ngrams([str(token) for token in tokenizer(sentence)], n))

    total = 0
    duplicate = 0
    for sentence in all_texts_list:
        four_grams = list(ngrams([str(token) for token in tokenizer(sentence)], n))
        total += len(four_grams)
        for four_gram in four_grams:
            if four_gram in unique_four_grams:
                duplicate += 1

    return duplicate/total

def compute_mauve(all_texts_list, human_references, model_id):
    torch.cuda.empty_cache() 
    # assert model_id == 'gpt2-large'
    mauve = load("mauve")

    results = mauve.compute(predictions=all_texts_list, references=human_references, featurize_model_name=model_id, max_text_length=256, device_id=0)

    assert len(all_texts_list) == len(human_references)

    return results.mauve, results.divergence_curve

def compute_bleu(all_texts_list, human_references):
    bleu = load("bleu")

    human_references = [[ref] for ref in human_references]
    results = bleu.compute(predictions=all_texts_list, references=human_references)

    return results['bleu']

def compute_sacrebleu(all_texts_list, human_references, tokenize, use_effective_order=False):
    sacrebleu = load("sacrebleu")

    human_references = [[ref] for ref in human_references]
    results = sacrebleu.compute(predictions=all_texts_list, references=human_references, tokenize=tokenize, use_effective_order=use_effective_order)

    return results['score']

def compute_debertascore(all_texts_list, human_references):
    bert = load("bertscore")

    human_references = [[ref] for ref in human_references]
    results = bert.compute(predictions=all_texts_list, references=human_references, lang="en", model_type='microsoft/deberta-xlarge-mnli')

    del results['hashcode']
    for key, value in results.items():
        results[key] = np.asarray(value).mean()

    return results


def compute_bertscore(all_texts_list, human_references):
    bert = load("bertscore")

    human_references = [[ref] for ref in human_references]
    results = bert.compute(predictions=all_texts_list, references=human_references, lang="en", rescale_with_baseline=True)

    del results['hashcode']
    for key, value in results.items():
        results[key] = np.asarray(value).mean()

    return results

def compute_rouge(all_texts_list, human_references, use_aggregator=True, use_stemmer=False):
    rouge = load("rouge")

    human_references = [[ref] for ref in human_references]
    results = rouge.compute(predictions=all_texts_list, references=human_references, use_aggregator=use_aggregator, use_stemmer=use_stemmer)

    return results

def load_samples(filename):
    """Load generated text from a pickle file or newline-delimited text file."""
    import pickle
    from pathlib import Path
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Samples file not found: {path}")
    if path.suffix == ".pkl":
        with path.open("rb") as stream:
            samples = pickle.load(stream)
    else:
        samples = path.read_text(encoding="utf-8").splitlines()
    if not isinstance(samples, (list, tuple)) or not all(isinstance(x, str) for x in samples):
        raise TypeError("Samples must be a list of strings or a text file")
    return [x.strip("<|endoftext|>").strip() for x in samples]


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Evaluate generated text samples.")
    parser.add_argument("--filename", required=True, help=".pkl or newline-delimited text samples")
    parser.add_argument("--ppl_model", default=None,
                        help="Perplexity model name/path; omit to skip PPL")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer for length truncation; defaults to --ppl_model")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--output_json", default=None, help="Optional metrics JSON output")
    parser.add_argument("--device", default="cuda", help="Perplexity device (cuda or cpu)")
    args = parser.parse_args()
    texts = load_samples(args.filename)
    if args.tokenizer or args.ppl_model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.ppl_model)
        clipped = []
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            clipped.append(tokenizer.decode(ids[:args.max_tokens]))
        texts = clipped
    metrics = {"filename": args.filename, "num_samples": len(texts),
               "diversity": compute_diversity(texts)["diversity"]}
    if args.ppl_model:
        metrics["generative_ppl"] = compute_perplexity(texts, args.ppl_model, args.device)
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        from pathlib import Path
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
