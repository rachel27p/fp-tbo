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
        'BOS': i == 0,
        'EOS': i == len(sentence) - 1,
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
    return [word2features(sentence, i) for i in range(len(sentence))]

def sent2labels(sentence):
    """Extract labels (tags) from a sentence."""
    return [label for _, label in sentence]

def detect_sentence_type(sentence):
    """Detect sentence type based on punctuation or structure."""
    sentence = sentence.strip()
    if sentence.endswith("?"):
        return "interrogative"
    elif sentence.endswith("!"):
        return "exclamatory"
    else:
        return "declarative"

def handle_negation(words, tags):
    """Handle negation phrases like 'doesn't' and 'don't'."""
    corrected_tags = []
    for i, (word, tag) in enumerate(zip(words, tags)):
        if word.lower() in ["doesn't", "don't", "didn't"]:
            corrected_tags.append('Auxiliary_verb')
            corrected_tags.append('Negation')  # Add separate tag for 'not'
        else:
            corrected_tags.append(tag)
    return corrected_tags

def analyze_sentence(sentence, model):
    """Analyze a given sentence and detect its tense and structure."""
    words = sentence.rstrip('.?!').split()
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

        if i > 0:
            prev_word = words[i - 1]
            word_features.update({
                '-1:word.lower()': prev_word.lower(),
                '-1:word.isupper()': prev_word.isupper(),
                '-1:word.istitle()': prev_word.istitle(),
            })
        else:
            word_features['BOS'] = True

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

    corrected_tags = handle_negation(words, tags)

    # Tentukan tense berdasarkan aturan
    sentence_tense = match_tense_rule(corrected_tags, words)

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

def extract_subject_predicate_object(tags, words):
    """Extract subject, predicate, and object with phrase support."""
    subject, predicate, obj = [], [], []
    
    # Find the main verb position
    verb_indices = [i for i, tag in enumerate(tags) if tag in ['Verb', 'Verb_past', 'Verb_ing']]
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
    
    # Get the position of the first verb (main verb or auxiliary)
    verb_pos = min(verb_indices) if verb_indices else min(aux_indices)
    
    # Define tags that can be part of subject phrase
    subject_tags = [
        'Pronoun_Subject', 
        'Noun', 
        'Noun_plural', 
        'Determiner', 
        'Adjective', 
        'Possessive_adjective'
    ]
    
    # Extract subject phrases
    i = 0
    while i < verb_pos:
        if tags[i] in subject_tags:
            phrase, last_idx = build_phrase(words, tags, i, subject_tags)
            if phrase:
                subject.append(phrase)
            i = last_idx + 1
        else:
            i += 1
    
    # The verb and any auxiliary verbs are predicate
    predicate = [words[i] for i in aux_indices] if aux_indices else []
    if verb_indices:
        predicate.append(words[verb_indices[0]])
    
    # Define tags that can be part of object phrase
    object_tags = [
        'Noun', 
        'Noun_plural', 
        'Pronoun_object', 
        'Determiner', 
        'Adjective',
        'Preposition'
    ]
    
    # Extract object phrases
    i = verb_pos + 1
    while i < len(words):
        if tags[i] in object_tags:
            phrase, last_idx = build_phrase(words, tags, i, object_tags)
            if phrase:
                obj.append(phrase)
            i = last_idx + 1
        else:
            i += 1
    
    # Collect other elements
    auxiliary_verbs = [words[i] for i in range(len(words)) if tags[i] == 'Auxiliary_verb']
    adverbs = [words[i] for i in range(len(words)) if tags[i] == 'Adverb']
    determiners = [words[i] for i in range(len(words)) if tags[i] == 'Determiner']
    adjectives = [words[i] for i in range(len(words)) if tags[i] == 'Adjective']
    possessive_adjectives = [words[i] for i in range(len(words)) if tags[i] == 'Possessive_adjective']
    possessive_pronouns = [words[i] for i in range(len(words)) if tags[i] == 'Possessive_pronoun']
    conjunctions = [words[i] for i in range(len(words)) if tags[i] == 'Conjunction']
    prepositions = [words[i] for i in range(len(words)) if tags[i] == 'Preposition']
    
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

def match_tense_rule(tags, words):
    """
    Match the sentence to a tense rule based on word tags and patterns.
    Handles interrogative, negative, and positive sentences with more precise checks.
    """
    # Get indices of different verb types and auxiliaries
    verb_indices = [i for i, tag in enumerate(tags) if tag == 'Verb']
    verb_past_indices = [i for i, tag in enumerate(tags) if tag == 'Verb_past']
    verb_ing_indices = [i for i, tag in enumerate(tags) if tag == 'Verb_ing']
    aux_verb_indices = [i for i, tag in enumerate(tags) if tag == 'Auxiliary_verb']
    negation_indices = [i for i, tag in enumerate(tags) if tag == 'Negation']

    # Check if sentence is negative
    is_negative = len(negation_indices) > 0
    
    # Get sentence type
    sentence_type = detect_sentence_type(' '.join(words))
    
    # Handle interrogative sentences
    if sentence_type == 'interrogative':
        if aux_verb_indices:
            aux_verb = words[aux_verb_indices[0]].lower()
            # Present tense interrogatives
            if aux_verb in ['do', 'does']:
                if is_negative:
                    return 'simple_present_negative_interrogative'
                return 'simple_present_interrogative'
            # Past tense interrogatives
            elif aux_verb == 'did':
                if is_negative:
                    return 'simple_past_negative_interrogative'
                return 'simple_past_interrogative'
            # Present continuous interrogatives
            elif aux_verb in ['is', 'are', 'am'] and verb_ing_indices:
                if is_negative:
                    return 'present_continuous_negative_interrogative'
                return 'present_continuous_interrogative'
            # Past continuous interrogatives
            elif aux_verb in ['was', 'were'] and verb_ing_indices:
                if is_negative:
                    return 'past_continuous_negative_interrogative'
                return 'past_continuous_interrogative'
        return 'unknown_interrogative'

    # Handle declarative sentences
    # First, check for continuous tenses
    if aux_verb_indices and verb_ing_indices:
        aux_verb = words[aux_verb_indices[0]].lower()
        # Present continuous
        if aux_verb in ['am', 'is', 'are']:
            if is_negative:
                return 'present_continuous_negative'
            return 'present_continuous'
        # Past continuous
        elif aux_verb in ['was', 'were']:
            if is_negative:
                return 'past_continuous_negative'
            return 'past_continuous'

    # Check for simple past
    if verb_past_indices:
        if aux_verb_indices and is_negative:
            return 'simple_past_negative'
        return 'simple_past'

    # Check for simple present
    if verb_indices:
        main_verb = words[verb_indices[0]].lower()
        verb_pos = verb_indices[0]

        # Find subject type before the verb
        subject_info = get_subject_type(tags[:verb_pos], words[:verb_pos])
        
        # Handle simple present based on subject type
        if subject_info['found']:
            if subject_info['is_plural']:
                # Plural subjects (I, you, they, we) use base form
                if not main_verb.endswith(('s', 'es')) or main_verb in ['was', 'has', 'does']:
                    if is_negative:
                        return 'simple_present_negative'
                    return 'simple_present'
            else:
                # Singular subjects (he, she, it) use s/es form
                if main_verb.endswith(('s', 'es')) and main_verb not in ['was', 'has', 'does']:
                    if is_negative:
                        return 'simple_present_negative'
                    return 'simple_present'

    return 'unknown'

def get_subject_type(tags, words):
    """
    Helper function to determine subject type (plural/singular).
    Returns dict with 'found' boolean and 'is_plural' boolean.
    """
    result = {'found': False, 'is_plural': False}
    
    # Check for pronouns first
    for i, tag in enumerate(tags):
        if tag == 'Pronoun_Subject':
            word = words[i].lower()
            result['found'] = True
            if word in ['i', 'you', 'they', 'we']:
                result['is_plural'] = True
            return result
    
    # Check for nouns and noun phrases
    noun_indices = [i for i, tag in enumerate(tags) 
                   if tag in ['Noun', 'Noun_plural']]
    
    if noun_indices:
        result['found'] = True
        last_noun_index = noun_indices[-1]
        # Check if the last noun is plural
        if tags[last_noun_index] == 'Noun_plural':
            result['is_plural'] = True
        
        # Check for coordinating conjunctions joining multiple subjects
        if 'Conjunction' in tags:
            result['is_plural'] = True
    
    return result


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
    dataset_path = "C:\\Users\\rahel\\OneDrive\\Dokumen\\unud\\Semester 3 - Informatika\\Teori Bahasa Dan Otomata\\code-ing\\fp-tbo\\dataset2.csv"
    sentences_file_path = "C:\\Users\\rahel\\OneDrive\\Dokumen\\unud\\Semester 3 - Informatika\\Teori Bahasa Dan Otomata\\code-ing\\fp-tbo\\dataKalimat.txt"
    
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