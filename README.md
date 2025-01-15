# fp-tbo
  Parsing   is the process of analyzing and breaking down a string or text into structured components based on predefined rules, often used in compilers, interpreters, or data processing tasks.

This script provides tools for analyzing sentences, extracting grammatical elements, and determining sentence structure and tense using Conditional Random Fields (CRF). Below is a high-level overview of its key components:

# Features

1. Data Loading:
   - load_data_from_csv(file_path): Reads sentences and tags from a CSV file.

2. Feature Extraction:
   - word2features(sentence, i): Extracts word-level features like casing, position, and neighboring words.
   - sent2features(sentence): Converts an entire sentence into feature dictionaries for CRF.

3. Tag Analysis:
   -  analyze_sentence(sentence, model) : Uses a CRF model to predict grammatical tags and analyze a sentence's structure.
   -  handle_negation(words, tags) : Identifies and handles negation in sentences.

4.   Grammatical Elements  :
   - Extracts subjects, predicates, and objects using  extract_subject_predicate_object(tags, words) .

5.   Sentence Type and Tense  :
   - Detects sentence type ( declarative ,  interrogative , or  exclamatory ) using  detect_sentence_type(sentence) .
   - Matches tenses with grammatical rules in  match_tense_rule(tags, words) .

6.   CRF Model Integration  :
   - Supports CRF for training and predicting grammatical tags.

# Requirements

- Python 3.x
- Libraries:
  -  pandas 
  -  sklearn-crfsuite 
  -  scikit-learn 

# Usage

1.   Prepare Dataset  :
   - Input CSV with columns:  sentence_id ,  word , and  tag .

2.   Train a CRF Model   (example not included in the script):
   - Use  sklearn_crfsuite  to train a CRF model with features generated using  sent2features .

3.   Analyze Sentences  :
   - Pass a trained model and a sentence to  analyze_sentence  for a detailed analysis of its elements and structure.

4.   Customization  :
   - Extend feature extraction in  word2features  to improve tagging accuracy.

# Notes

- Ensure consistent formatting of input data.
- Use a well-trained CRF model for optimal results.
