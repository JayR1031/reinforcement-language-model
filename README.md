# 🧠 Reinforcement Learning with Language Models

> **Character n-gram language model with reinforcement learning from human feedback**

This project demonstrates a complete implementation of a character-level n-gram language model enhanced with reinforcement learning principles. The model learns from a corpus, generates text based on learned patterns, and improves through human feedback.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
  - [1. Corpus Construction](#1-corpus-construction)
  - [2. Character N-Gram Modeling](#2-character-n-gram-modeling)
  - [3. Text Generation](#3-text-generation)
  - [4. Reinforcement Learning](#4-reinforcement-learning)
  - [5. The Role of EOS (End of Sequence)](#5-the-role-of-eos-end-of-sequence)
  - [6. Testing and Iteration](#6-testing-and-iteration)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Practical Insights](#practical-insights)
- [Getting Started](#getting-started)

---

## 🎯 Overview

This project builds a **character-level n-gram language model** that:
1. Learns patterns from a text corpus
2. Generates new text based on those patterns
3. Improves through reinforcement learning (reward/penalty system)
4. Adapts to human preferences over time

---

## 🔑 Key Concepts

### 1. Corpus Construction

**What it is:** The corpus is the training data - a collection of text that the model learns from.

**How it works:**
- Reads text data from a CSV file containing quotes/sentences
- Cleans and preprocesses the text
- Adds special `<EOS>` (End of Sequence) markers to indicate where sequences end
- Builds the foundational dataset the model will learn patterns from

**Why it matters:** The quality and diversity of your corpus directly affects the model's ability to generate coherent, varied text. A rich corpus leads to better pattern recognition.

**Code concept:**
```python
import pandas as pd

def build_corpus(csv_path, column_name='text'):
    """Load text data and prepare it for training"""
    df = pd.read_data(csv_path)
    corpus = ' '.join(df[column_name].tolist())
    corpus += '<EOS>'  # Add end marker
    return corpus
```

---

### 2. Character N-Gram Modeling

**What it is:** N-gram modeling looks at sequences of N characters to predict what comes next.

**How it works:**
- Slides a window of size N through the corpus
- For each N-character sequence (called a "context"), records what character came next
- Builds a probability distribution: "given these N characters, what's likely to follow?"
- Stores these patterns in a transition dictionary

**Example with N=3:**
- Text: "hello"
- Context "hel" → next char is "l"
- Context "ell" → next char is "o"
- Context "llo" → next char is "<EOS>"

**Why it matters:** This is the core learning mechanism. The model learns language patterns by observing character sequences and their continuations.

**Code concept:**
```python
from collections import defaultdict, Counter

def build_ngram_model(corpus, n=3):
    """Build transition probabilities from corpus"""
    transitions = defaultdict(Counter)
    
    for i in range(len(corpus) - n):
        context = corpus[i:i+n]        # N characters
        next_char = corpus[i+n]        # What follows
        transitions[context][next_char] += 1
    
    return transitions
```

---

### 3. Text Generation

**What it is:** Using the learned patterns to create new text.

**How it works:**
1. Start with a seed context (N characters)
2. Look up what characters typically follow this context
3. Randomly sample a next character based on probabilities
4. Append that character and shift the context window
5. Repeat until reaching `<EOS>` or max length

**Why it matters:** This is where the model demonstrates what it has learned. Generation quality shows how well the model captured language patterns.

**Code concept:**
```python
import random

def generate_text(model, seed, max_length=100):
    """Generate text using the n-gram model"""
    generated = seed
    context = seed[-n:]  # Last N characters
    
    while len(generated) < max_length:
        if context not in model:
            break
            
        # Get possible next characters and their counts
        possible_chars = model[context]
        
        # Sample based on frequencies
        chars, counts = zip(*possible_chars.items())
        next_char = random.choices(chars, weights=counts)[0]
        
        if next_char == '<EOS>':
            break
            
        generated += next_char
        context = generated[-n:]  # Update context
    
    return generated
```

---

### 4. Reinforcement Learning

**What it is:** A learning paradigm where the model improves based on feedback (rewards and penalties).

**How it works:**
1. **Generate** text using current model
2. **Evaluate** the output (human rates it as good/bad)
3. **Update** the model:
   - **Reward (positive feedback):** Increase probabilities of the character sequences used
   - **Penalty (negative feedback):** Decrease probabilities of the character sequences used
4. **Iterate:** Model gradually learns which patterns produce better outputs

**Why it matters:** This allows the model to align with human preferences, going beyond statistical patterns to optimize for quality.

**Key components:**
- **Reward function:** How we quantify "good" output
- **Update rule:** How we adjust probabilities based on feedback
- **Exploration vs. Exploitation:** Balance between trying new patterns and using known good ones

**Code concept:**
```python
def apply_feedback(model, generated_text, reward, learning_rate=0.1):
    """Update model based on human feedback"""
    n = len(seed)
    
    for i in range(len(generated_text) - n):
        context = generated_text[i:i+n]
        next_char = generated_text[i+n]
        
        if context in model:
            # Adjust the count/probability
            current_count = model[context][next_char]
            
            if reward > 0:  # Good output
                model[context][next_char] += learning_rate * reward
            else:  # Bad output
                model[context][next_char] = max(1, current_count + learning_rate * reward)
```

---

### 5. The Role of EOS (End of Sequence)

**What it is:** `<EOS>` is a special token marking where text sequences end.

**Why it's crucial:**
- **Natural stopping:** Tells the generator when to stop producing text
- **Sentence boundaries:** Helps model learn where thoughts/sentences complete
- **Pattern recognition:** Allows model to learn "this context typically ends here"
- **Prevents infinite generation:** Without it, model might generate indefinitely

**Where it appears:**
- End of each training example in the corpus
- As a possible next character during generation
- In transition probabilities alongside regular characters

**Example:**
```
Training: "hello world<EOS>"
Model learns: context "rld" → '<EOS>' (ends here)
Generation: When model samples '<EOS>', it stops
```

---

### 6. Testing and Iteration

**What it is:** The continuous process of generating, evaluating, and improving.

**The feedback loop:**
```
1. Generate text with current model
   ↓
2. Human evaluates output
   ↓
3. Apply reward/penalty
   ↓
4. Model updates probabilities
   ↓
5. Generate again (improved)
   ↓
(repeat)
```

**Why it matters:** Each iteration makes the model slightly better at producing human-preferred outputs. Over time, this compounds into significant improvement.

**Testing strategies:**
- Generate multiple samples and compare quality
- Test with different seed contexts
- Track metrics like coherence, relevance, creativity
- Monitor how probabilities shift over time

---

## 🔧 Implementation Details

### Core Data Structures

1. **Transition Dictionary**
   - Maps context → Counter of next characters
   - Example: `{'hel': Counter({'l': 10, 'p': 2})}`

2. **Corpus**
   - String containing all training text
   - Preprocessed and marked with `<EOS>`

3. **Hyperparameters**
   - `n`: Context size (typically 2-5)
   - `learning_rate`: How much to adjust per feedback
   - `max_length`: Generation limit

### Algorithm Flow

```
┌─────────────────────┐
│  Load CSV Corpus    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Build N-Gram Model  │
│ (Count transitions) │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Generate Text      │
│  (Sample from       │
│   probabilities)    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Human Feedback     │
│  (Reward/Penalty)   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Update Model       │
│  (Adjust probs)     │
└──────────┬──────────┘
           ↓
    (Loop back to Generate)
```

---

## 💡 Practical Insights

### What Makes This Approach Powerful

1. **Simplicity:** Character-level models are conceptually simple and easy to understand
2. **No vocabulary limits:** Can generate any character combination
3. **Learns from examples:** Directly mimics patterns in training data
4. **Human-in-the-loop:** RL allows continuous alignment with preferences

### Common Challenges

1. **Short context (small N):** May generate incoherent text
2. **Long context (large N):** Requires more data, less creative
3. **Sparse data:** Some contexts rarely appear, limiting learning
4. **Reward shaping:** Defining good rewards is an art

### Best Practices

1. **Start with quality corpus:** Diverse, clean, representative data
2. **Choose N wisely:** Balance between coherence and creativity (3-4 often works well)
3. **Iterate frequently:** Many small updates beat few large ones
4. **Monitor diversity:** Ensure model doesn't converge to repetitive outputs
5. **Save checkpoints:** Keep versions to compare progress

---

## 🚀 Usage

### Running the Notebook

1. Open the Colab notebook
2. Upload your CSV dataset (with a text column)
3. Run cells sequentially:
   - Build corpus
   - Train n-gram model
   - Generate samples
   - Provide feedback
   - Observe improvements

### Customization

- **Change N:** Modify context size for different pattern complexity
- **Adjust learning rate:** Control how quickly model adapts to feedback
- **Vary corpus:** Try different text styles/domains
- **Experiment with rewards:** Test different feedback strategies

---

## 🎓 Getting Started

### Prerequisites
- Python 3.x
- pandas (for CSV handling)
- collections (defaultdict, Counter)
- random (for sampling)

### Quick Start

1. Clone this repository
2. Open the Colab notebook
3. Follow the step-by-step cells
4. Experiment with your own data!

---

## 📖 Learning Resources

### Key Takeaways

- **N-grams capture local patterns** in text
- **Probabilistic generation** creates varied outputs
- **Reinforcement learning** aligns models with human values
- **EOS tokens** provide natural boundaries
- **Iteration** is key to improvement

### Connections to Broader Concepts

- **Markov Chains:** N-grams are Markov models (next state depends only on last N states)
- **Language Modeling:** Foundation for modern NLP (GPT, BERT build on these ideas)
- **RLHF:** This project demonstrates reinforcement learning from human feedback at a basic level
- **Generative Models:** Understanding generation from probability distributions

---

## 🤝 Contributing

Feel free to fork, experiment, and share improvements! This is a learning project meant to demystify language models and RL.

---

## 📝 License

MIT License - feel free to use for learning and experimentation.

---

**Happy Learning! 🎉**

Remember: The best way to understand these concepts is to run the code, experiment with parameters, and observe how changes affect output. Each iteration teaches you something new about how language models work.
