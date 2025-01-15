import pandas as pd
from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import train_test_split

def load_data_from_csv(file_path):
    """
    Load dataset from a CSV file and format it into a list of sentences.
    Each sentence is a list of (word, tag) tuples.
    """
    data = []
    try:
        df = pd.read_csv(file_path)
        for _, group in df.groupby('sentence_id'):
            words = list(group['word'])
            tags = list(group['tag'])
            # Check consistency between words and tags
            if len(words) != len(tags):
                raise ValueError(f"Inconsistent word-tag pairs in sentence_id {_}")
            sentence = list(zip(words, tags))
            data.append(sentence)
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")
    return data


def word2features(sentence, i):
    """Extract features for a given word in a sentence."""
    word = sentence[i][0]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'BOS': i == 0,  # Beginning of sentence
        'EOS': i == len(sentence) - 1,  # End of sentence
    }
    if i > 0:
        prev_word = sentence[i - 1][0]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.isupper()': prev_word.isupper(),
            '-1:word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True

    if i < len(sentence) - 1:
        next_word = sentence[i + 1][0]
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.isupper()': next_word.isupper(),
            '+1:word.istitle()': next_word.istitle(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sentence):
    """Convert a sentence into a list of feature dictionaries."""
    features = [word2features(sentence, i) for i in range(len(sentence))]
    return features

def sent2labels(sentence):
    """Extract labels (tags) from a sentence."""
    labels = [label for _, label in sentence]
    return labels

def detect_sentence_type(sentence):
    """Deteksi jenis kalimat berdasarkan tanda baca atau struktur."""
    sentence = sentence.strip()
    if sentence.endswith("?"):
        return "interrogative"
    elif sentence.endswith("!"):
        return "exclamatory"
    else:
        return "declarative"

def extract_subject_predicate_object(tags, words):
    """Extract subject, predicate, and object with support for interrogative sentences."""
    subject, predicate, obj = [], [], []
    
    # Initialize other elements
    auxiliary_verbs = []
    adverbs = []
    determiners = []
    adjectives = []
    possessive_adjectives = []
    possessive_pronouns = []
    conjunctions = []
    prepositions = []
    
    # Detect if it's a question
    is_question = words[-1].endswith('?') or words[0].lower() in ['what', 'where', 'when', 'who', 'why', 'how', 'do', 'does', 'did', 'is', 'are', 'was', 'were']
    
    # Find all verb positions
    verb_indices = [i for i, tag in enumerate(tags) if tag in ['Verb', 'Verb_past', 'Verb_ing', 'verb_past_participle']]
    aux_indices = [i for i, tag in enumerate(tags) if tag == 'Auxiliary_verb']
    
    if not verb_indices and not aux_indices:
        return {
            "subject": [],
            "predicate": [],
            "object": [],
            "auxiliary_verbs": [],
            "adverbs": [],
            "determiners": [],
            "adjectives": [],
            "possessive_adjectives": [],
            "possessive_pronouns": [],
            "conjunctions": [],
            "prepositions": []
        }
    
    # Define tags that can be part of subject phrase
    subject_tags = [
        'Pronoun_Subject', 
        'Noun', 
        'Noun_plural', 
        'Determiner', 
        'Adjective', 
        'Possessive_adjective'
    ]
    
    if is_question:
        # For questions, look for subject after auxiliary verb or WH-word
        start_idx = 1  # Skip the first word if it's a question word
        if aux_indices:
            start_idx = aux_indices[0] + 1
            
        # Extract subject from position after auxiliary/question word
        i = start_idx
        while i < len(words):
            if tags[i] in subject_tags:
                phrase, last_idx = build_phrase(words, tags, i, subject_tags)
                if phrase:
                    subject.append(phrase)
                    break
            i += 1
    else:
        # Original logic for declarative sentences
        i = 0
        verb_pos = min(verb_indices) if verb_indices else min(aux_indices)
        while i < verb_pos:
            if tags[i] in subject_tags:
                phrase, last_idx = build_phrase(words, tags, i, subject_tags)
                if phrase:
                    subject.append(phrase)
                i = last_idx + 1
            else:
                i += 1
    
    # Extract predicate (verbs and auxiliary verbs)
    predicate = [words[i] for i in aux_indices] if aux_indices else []
    if verb_indices:
        predicate.append(words[verb_indices[0]])
    
    # Define object tags
    object_tags = [
        'Noun', 
        'Noun_plural', 
        'Pronoun_object', 
        'Determiner', 
        'Adjective',
        'Preposition',
        'Adverb'
    ]
    
    # Extract object (after verb/auxiliary)
    last_verb_pos = max(verb_indices) if verb_indices else max(aux_indices)
    i = last_verb_pos + 1
    while i < len(words):
        if tags[i] in object_tags:
            phrase, last_idx = build_phrase(words, tags, i, object_tags)
            if phrase:
                obj.append(phrase)
            i = last_idx + 1
        else:
            i += 1
    
    # Collect other elements
    for i, (word, tag) in enumerate(zip(words, tags)):
        if tag == 'Auxiliary_verb': auxiliary_verbs.append(word)
        elif tag == 'Adverb': adverbs.append(word)
        elif tag == 'Determiner': determiners.append(word)
        elif tag == 'Adjective': adjectives.append(word)
        elif tag == 'Possessive_adjective': possessive_adjectives.append(word)
        elif tag == 'Possessive_pronoun': possessive_pronouns.append(word)
        elif tag == 'Conjunction': conjunctions.append(word)
        elif tag == 'Preposition': prepositions.append(word)
    
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "auxiliary_verbs": auxiliary_verbs,
        "adverbs": adverbs,
        "determiners": determiners,
        "adjectives": adjectives,
        "possessive_adjectives": possessive_adjectives,
        "possessive_pronouns": possessive_pronouns,
        "conjunctions": conjunctions,
        "prepositions": prepositions
    }


def analyze_sentence(sentence, model):
    """Analyze a given sentence and detect its tense and structure."""
    # Tokenize kalimat
    words = sentence.rstrip('.').split()
    print("Words:", words)

    # Tokenize dan prediksi dengan CRF model
    features = []
    for i, word in enumerate(words):
        word_features = {
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'BOS': i == 0,
            'EOS': i == len(words) - 1,
        }
        
        # Tambah fitur kata sebelumnya
        if i > 0:
            prev_word = words[i - 1]
            word_features.update({
                '-1:word.lower()': prev_word.lower(),
                '-1:word.isupper()': prev_word.isupper(),
                '-1:word.istitle()': prev_word.istitle(),
            })
        else:
            word_features['BOS'] = True
            
        # Tambah fitur kata setelahnya
        if i < len(words) - 1:
            next_word = words[i + 1]
            word_features.update({
                '+1:word.lower()': next_word.lower(),
                '+1:word.isupper()': next_word.isupper(),
                '+1:word.istitle()': next_word.istitle(),
            })
        else:
            word_features['EOS'] = True
            
        features.append(word_features)
    
    try:
        tags = model.predict([features])[0]
        print("Predicted tags:", tags)
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

    # Koreksi tag prediksi jika diperlukan
    corrected_tags = []
    for i, (word, tag) in enumerate(zip(words, tags)):
        # Cek apakah kata sebelumnya adalah subjek
        is_after_subject = (i > 0 and corrected_tags[i-1] == 'Pronoun_Subject')
        
        if word.lower() in ['is', 'are', 'was', 'were', 'have', 'has', 'am', 'do','does', 'had', 'will', 'been', 'shall', 'be', 'did']:
            corrected_tags.append('Auxiliary_verb')
        # Jika kata setelah subjek dan berakhiran 's'
        elif (is_after_subject and 
            word.endswith('s') and 
            not word.lower() in ['is', 'was', 'has']):
            corrected_tags.append('Verb')
        else:
            corrected_tags.append(tag)


    # Tentukan tense berdasarkan tense rules
    sentence_tense = match_tense_rule(corrected_tags, words)
    #print("Detected tense:", sentence_tense)

    # Analisis elemen kalimat
    elements = {key: [] for key in [
        "Pronoun_Subject", "Pronoun_object", "Possessive_adjective", "Possessive_pronoun",
        "Verb", "Verb_ing", "Verb_past_participle", "Verb_past", "Auxiliary_verb",
        "Noun", "Noun_plural", "Adverb", "Determiner", "Adjective", "Conjunction", "Preposition"
    ]}
    for word, tag in zip(words, corrected_tags):
        if tag in elements:
            elements[tag].append(word)

    spo = extract_subject_predicate_object(corrected_tags, words)
    sentence_type = detect_sentence_type(sentence)

    return {
        "elements": elements,
        "subject": spo["subject"],
        "predicate": spo["predicate"],
        "object": spo["object"],
        "sentence_type": sentence_type,
        "sentence_tense": sentence_tense
    }


def match_tense_rule(tags, words):
    """Match the sentence to a tense rule based on word tags."""
    # Convert words to lowercase for easier matching
    words = [word.lower() for word in words]
    
    # Helper function to check if any auxiliary verb is in the sentence
    def contains_any(word_list):
        return any(aux in words for aux in word_list)

    # Define auxiliary verbs with variations (positive and negative)
    will_variants = ["will", "won't", "will not", "wont", "will be not"]
    shall_variants = ["shall", "shan't", "shall not", "shant"]
    has_variants = ["has", "hasn't", "has not", "hasnt"]
    have_variants = ["have", "haven't", "have not", "havent"]
    do_variants = ["do", "don't", "do not", "dont"]
    does_variants = ["does", "doesn't", "does not", "doesnt"]
    did_variants = ["did", "didn't", "did not", "didnt"]
    had_variants = ["had", "hadn't", "had not", "hadnt"]
    is_variants = ["is", "isn't", "is not", "isnt"]
    are_variants = ["are", "aren't", "are not", "arent"]
    was_variants = ["was", "wasn't", "was not", "wasnt"]
    were_variants = ["were", "weren't", "were not", "werent"]
    am_variants = ["am", "am not"]

    Verb_ing = 'Verb_ing'
    Verb_past = 'Verb_past'
    Verb = 'Verb'
    Verb_past_participle = 'Verb_past_participle'

     # PAST TENSE
    if contains_any(was_variants) or contains_any(were_variants):
        for word, tag in zip(words, tags):
            if tag == 'Verb_ing':  # Check for Verb_ing tag
                return 'past_continuous'


    if contains_any(had_variants):
        for word, tag in zip(words, tags):
            if 'been' in words and tag == 'Verb_ing':  # Check for Verb_ing tag
                return 'past_perfect_continuous'
            if tag == 'Verb_past_participle':
                return 'past_perfect'

    if contains_any(did_variants):
        for word, tag in zip(words, tags):
            if tag == 'Verb' and not contains_any(do_variants) and not contains_any(does_variants):  # Check for Verb_past tag
                return 'simple_past'

    for word, tag in zip(words, tags):
        if tag == 'Verb_past' and not contains_any(do_variants) and not contains_any(does_variants):  # Check for Verb_past tag
            return 'simple_past'

    # FUTURE TENSE
    if contains_any(will_variants) or contains_any(shall_variants):
        if contains_any(have_variants):
            for word, tag in zip(words, tags):
                if 'been' in words:
                        if tag == 'Verb_ing':  # Check for Verb_ing tag
                            return 'future_perfect_continuous'
                if tag == 'Verb_past_participle':
                    return 'future_perfect'
            
        for word, tag in zip(words, tags):
            if 'be' in words and tag == 'Verb_ing':  # Check for Verb_ing tag
                return 'future_continuous'
        return 'simple_future'

    # PRESENT TENSE
    if contains_any(is_variants) or contains_any(are_variants) or contains_any(am_variants):
        for word, tag in zip(words, tags):
            if tag == 'Verb_ing':  # Check for Verb_ing tag
                return 'present_continuous'

    if contains_any(do_variants) or contains_any(does_variants):
        return 'simple_present'

    if len(words) >= 2 and words[0].lower() in ['she', 'he', 'it'] and words[1].endswith['s', 'es'] and words[1] not in ['has', 'does']:
        return 'simple_present'

    if contains_any(has_variants) or contains_any(have_variants):
        for word, tag in zip(words, tags):
            if 'been' in words:
                    if tag == 'Verb_ing':  # Check for Verb_ing tag
                        return 'present_perfect_continuous'
                    
            if tag == 'Verb_past_participle':
                return 'present_perfect'

    return 'unknown'


def build_phrase(words, tags, start_idx, valid_tags):
    """Build a phrase starting from given index using valid tags."""
    phrase = []
    i = start_idx
    while i < len(tags) and tags[i] in valid_tags:
        phrase.append(words[i])
        i += 1
    return ' '.join(phrase), i - 1


def read_sentences_from_file(file_path):
    """Membaca kalimat dari file teks"""
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def main():
    dataset_path = "C:\\Users\\rahel\\OneDrive\\Dokumen\\unud\\Semester 3 - Informatika\\Teori Bahasa Dan Otomata\\code-ing\\fp-tbo\\parsing-code\\dataset2.csv"
    sentences_file_path = "C:\\Users\\rahel\\OneDrive\\Dokumen\\unud\\Semester 3 - Informatika\\Teori Bahasa Dan Otomata\code-ing\\fp-tbo\\parsing-code\\dataKalimat.txt"
    
    # Load dataset dan train model
    try:
        data = load_data_from_csv(dataset_path)
        if not data:
            raise ValueError("Dataset is empty or not formatted correctly!")
    except ValueError as e:
        print(e)
        exit()

    # Split data dan train model
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=100)
    X_train = [sent2features(sentence) for sentence in train_data]
    y_train = [sent2labels(sentence) for sentence in train_data]
    X_test = [sent2features(sentence) for sentence in test_data]
    y_test = [sent2labels(sentence) for sentence in test_data]

    # Train CRF model
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    try:
        print("Training model...")
        crf.fit(X_train, y_train)
        print("Model training completed!")
    except Exception as e:
        print(f"Error training CRF model: {e}")
        exit()

    # Evaluasi model
    y_pred = crf.predict(X_test)
    print("\nModel Accuracy:", metrics.flat_accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(metrics.flat_classification_report(y_test, y_pred, labels=crf.classes_, zero_division=0))

    # Loop untuk input kalimat dinamis
    while True:
        print("\n" + "="*50)
        print("Sentence Analyzer")
        print("="*50)
        print("1. Analyze a single sentence")
        print("2. Analyze sentences from file")
        print("3. Exit")
        choice = input("Choose option (1/2/3): ")

        if choice == '1':
            sentence = input("\nEnter a sentence to analyze (e.g., 'She was reading a book'): ")
            if sentence.strip():
                try:
                    print("\nAnalyzing sentence:", sentence)
                    analysis = analyze_sentence(sentence, crf)
                    
                    print("\nAnalysis Results:")
                    print("-"*30)
                    print(f"Sentence Type: {analysis['sentence_type']}")
                    print(f"Tense: {analysis['sentence_tense']}")
                    print("\nSentence Elements:")
                    print(f"Subject: {', '.join(analysis['subject']) if analysis['subject'] else 'Not found'}")
                    print(f"Predicate: {', '.join(analysis['predicate']) if analysis['predicate'] else 'Not found'}")
                    print(f"Object: {', '.join(analysis['object']) if analysis['object'] else 'Not found'}")
                    
                    print("\nDetailed Elements:")
                    for element_type, words in analysis['elements'].items():
                        if words:
                            print(f"{element_type}: {', '.join(words)}")
                except Exception as e:
                    print(f"Error analyzing sentence: {e}")
            else:
                print("Please enter a valid sentence!")

        elif choice == '2':
            sentences = read_sentences_from_file(sentences_file_path)
            if sentences:
                print(f"\nFound {len(sentences)} sentences in the file.")
                for i, sentence in enumerate(sentences, 1):
                    try:
                        print(f"\n{'-'*50}")
                        print(f"Analyzing sentence {i}: {sentence}")
                        analysis = analyze_sentence(sentence, crf)
                        
                        print("\nAnalysis Results:")
                        print("-"*30)
                        print(f"Sentence Type: {analysis['sentence_type']}")
                        print(f"Tense: {analysis['sentence_tense']}")
                        print("\nSentence Elements:")
                        print(f"Subject: {', '.join(analysis['subject']) if analysis['subject'] else 'Not found'}")
                        print(f"Predicate: {', '.join(analysis['predicate']) if analysis['predicate'] else 'Not found'}")
                        print(f"Object: {', '.join(analysis['object']) if analysis['object'] else 'Not found'}")
                        
                        print("\nDetailed Elements:")
                        for element_type, words in analysis['elements'].items():
                            if words:
                                print(f"{element_type}: {', '.join(words)}")
                    except Exception as e:
                        print(f"Error analyzing sentence {i}: {e}")
                        continue
            else:
                print("No sentences found in the file or error reading the file.")
        
        elif choice == '3':
            print("\nThank you for using Sentence Analyzer!")
            break
        else:
            print("\nInvalid option! Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()