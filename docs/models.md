# Models

Every model lives in its own subdirectory under `models/` and inherits from
`core.adversarial_model.AdversarialModel`. Each subdirectory is self-contained:
it owns its weights, corpora, and any cached artifacts.

## Adversarial DGA models

**TLD convention.** Eight models are defined by their paper on the second-level
label only: WGAN-GP DeepDGA, CDGA, GWDGA, WGAN-based DGA, PKDGA, WordDGA, TLVDGA
and SADGA. Because the detectors are trained on full domains and the lexical
metrics change with the TLD, these models append a TLD drawn from the empirical
public-suffix distribution of the D1 Tranco slice (`core/tld.py`, top 50
suffixes weighted by frequency). Models whose paper or seed data fixes the TLD
(static `.com`, a paper list, or the seed's own TLD) keep their rule. The
SLD-only analysis mode (`docs/analysis.md`) removes the TLD again for a
generator-only comparison.

Every adversarial model reads its benign training input from the D1 benign
split (`TRANCO_D1_BENIGN` in `core/data_splits.py`, 100,000 domains). The
models that also need real AGDs draw them from the D1 malicious split through
`load_malicious_seeds()`. See [`datasets.md`](datasets.md).

### DeepDGA (`models/deepdga/`)

GAN-based adversarial DGA from
*"DeepDGA: Adversarially-Tuned Domain Generation and Detection"*
(Anderson et al., 2016). Trained on benign domains to mimic their lexical
distribution.

- Internal `fit()` loads benign training data from the D1 benign split
  (`TRANCO_D1_BENIGN`, `dataset/splits/D1_benign.csv`).
- Short-circuits when its trained artifact already exists on disk.
- `fit()` records its training time and sets `trained_from_scratch` (`false`
  on a cached-weights load, `true` on a real training run).

### CharBot (`models/charbot/`)

From *"CharBot: A Simple and Effective Method for Evading DGA Classifiers"*
(Peck et al., 2019). Adapted from the authors' source.

### Deception (`models/deception/`)

From *"Detection of Algorithmically Generated Domain Names used by Botnets: A
Dual Arms Race"* (Spooren et al., 2019). Adapted from the authors' source.

### MaskDGA (`models/maskdga/`)

From *"MaskDGA: An Evasion Attack Against DGA Classifiers and Adversarial
Defenses"* (Sidi et al., 2020). Adapted from the authors' source.

- `fit()` trains a substitute classifier and caches its weights, short-circuiting
  when they exist. It records its training time and `trained_from_scratch` the
  same way DeepDGA does.

### PCFGBasedDGA (`models/pcfg_based_dga/`)

PCFG-based adversarial DGA from *"Stealthy Domain Generation Algorithms"* (Fu et al., 2017). Generates stealthy domains by modeling the syllable structure 
of benign domains using a Probabilistic Context-Free Grammar.

- `fit()` implements a native regex-based heuristic to extract syllables from the D1 Tranco slice, splitting them into terminal sets (A and C) and injecting numerics.
- Caches these learned terminal sets as "weights" in a local `.h5` file.
- Short-circuits when its `.h5` artifact already exists on disk to bypass redundant dataset processing.

### HMMBasedDGA (`models/hmm_based_dga/`)

HMM-based adversarial DGA from *"Stealthy Domain Generation Algorithms"* (Fu et al., 2017). Generates stealthy domains character-by-character utilizing 
Markov transition probabilities (N-grams) learned from benign training data.

- `fit()` calculates N-gram frequencies, context counts, and empirical length percentiles from the D1 Tranco slice.
- Serializes and caches its complex dictionary structures into an `.h5` file.
- Short-circuits on a cached-weights load and correctly tracks the `trained_from_scratch` boolean flag.

### FGSMBasedDGA (`models/fgsm_based_dga/`)
Gradient-based adversarial DGA from *"Improving DGA-Based Malicious Domain Classifiers for Malware Defense with Adversarial Machine Learning"* (Yilmaz et al., 2020). Employs a Fast Gradient Sign Method (FGSM) attack against a target LSTM classifier, mutating real malicious domains via continuous noise injection and restricted cosine similarity projection to evade detection.

- `fit()` trains an internal LSTM substitute classifier using a 50/50 balanced dataset composed of benign domains from the D1 Tranco slice and real AGDs loaded from DGArchive CSVs.
- Caches the complete set of malicious seed domains and the classifier weights within two different `.h5` files using a hybrid persistence mechanism.

### GADGA (`models/gadga/`)
Genetic Algorithm-based adversarial DGA from *"Detection of algorithmically-generated domains: An adversarial machine learning approach"* (Alaeiyan et al., 2020). Generates evasive domains by evolving a population to balance pronunciation scores (via a bigram model) and lexical randomness (normalized chi-squared). 
- `fit()` builds a bigram language model from the D1 Tranco slice and initializes a seed population randomly sampled from malicious domains. Sets `trained_from_scratch` the same way MaskDGA does.
- Executes the genetic algorithm to evolve the population, then caches the final elite domains and the language model probabilities in a local `.h5` file.  
- Short-circuits when its `.h5` artifact already exists on disk.   

### WGANGPDeepDGA (`models/wgan_gp_deepdga/`)
WGAN-GP DeepDGA implementation based on *"Domain Generation Algorithm Detection Utilizing Model Hardening Through GAN-Generated Adversarial Examples"* (Gould et al., 2020). Generates stealthy domain names by training a Wasserstein GAN with Gradient Penalty, utilizing an architecture that features Embedding, LSTM, and parallel Conv1D branches.  
- `fit()` performs adversarial training over one-hot encoded domains from the D1 Tranco slice, utilizing @tf.function graph execution for efficiency.  
- Serializes and caches both its generator and critic models into local `.h5` files.  
- Short-circuits when these weights already exist on disk, accurately tracks the `trained_from_scratch` flag, and applies rejection sampling during generation to ensure valid DNS lengths and structures.

### Khaos (`models/khaos/`)
From *"Khaos: An Adversarial Neural Network DGA With High Anti-Detection Ability"* (Yun et al., 2020). Generates stealthy domains by using a Wasserstein GAN with Gradient Penalty approach, combined with n-grams.  
- `fit()` loads benign training data from the D1 Tranco slice to build a dictionary of the top 5,000 most frequent n-grams and train the discriminator and generator networks. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches the generator weights, discriminator weights, and the learned n-gram dictionary in local .`h5` files.  Short-circuits when its trained artifacts already exist on disk.  

### DnGAN (`models/dngan/`)
From *"Adversarial DGA Domain Examples Generation and Detection"* (Cao et al., 2020).GAN-based adversarial DGA that generates stealthy domains by projecting character probability distributions into a continuous embedding space to enable differentiable adversarial training.  
- `fit()` loads benign training data from the D1 Tranco slice and malicious samples to pre-train the discriminator before the adversarial phase. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches its generator and discriminator weights locally as `.h5` files.  Short-circuits when its trained artifacts already exist on disk.

### ShadowDGA (`models/shadowdga/`)
From *"ShadowDGA: Toward Evading DGA Detectors with GANs"* (Zheng et al., 2021). GAN-based adversarial DGA that uses 1D CNNs, Residual Blocks, and Wasserstein GAN with Gradient Penalty (WGAN-GP) to model and generate benign-looking domain sequences.

- `fit()` loads benign training data from the D1 Tranco slice, extracting valid Second-Level Domains (SLDs) and encoding them as one-hot tensors. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches the learned generator and discriminator weights locally as `.h5` artifacts. Short-circuits when its `.h5` artifacts already exist on disk to bypass redundant dataset processing and training.

### NDG (`models/ndg/`)
From *"Neural networks based domain name generation"* (Wang and Guo, 2021). VAE-based adversarial DGA that generates stealthy domains using a Variational Autoencoder (VAE) combined with Gated Convolutional Neural Networks (GCNN) to model the lexical sequence of benign domains.  
- `fit()` loads benign training data from the D1 Tranco slice. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches its learned model weights in a local `.h5` file.  Short-circuits when its trained artifact already exists on disk. 

### CLETer (`models/cleter/`)
From *"CLETer: A Character-level Evasion Technique Against Deep Learning DGA Classifiers"* (Liu, Zhang, et al., 2021). Character-level adversarial evasion technique that performs black-box character substitution guided by influence scores (HIS, TIS, CIS/OIS) to evade deep learning DGA classifiers.
- `fit()` extracts malicious base domains from DGArchive CSV files to serve as modification targets.
- Automatically instantiates and fits the default `CNNDetector` if no target classifier is explicitly provided.
- Caches extracted base domains (SLDs and TLDs) and the CNN detector weights in two different local `.h5` files. Short-circuits when its `.h5` cache already exists on disk.
- The target classifier is queried in batches: one call for all the prefixes and suffixes of the influence scores and one per substitution round, which reproduces the sequential search of Algorithm 1 exactly at a fraction of the cost.

### GeometricPerturbationDGA (`models/geometric_perturbation_dga/`)
From *"A Novel DGA Domain Adversarial Sample Generation Method By Geometric Perturbation"* (Liu, Yu, et al., 2021). Adversarial DGA based on the geometric perturbation algorithm that generates adversarial samples by adding geometric vectors from benign domains to DGA domains.
- `fit()` implements a dual-network architecture (an LSTM-based Generator and a CNN-based target classifier), training them through MLE pre-training and joint adversarial phases using a Straight-Through Estimator to maintain differentiability. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches the optimized Generator and Object networks weights in two different local `.h5` files, short-circuiting when its trained artifacts already exist on disk.
- Generation keeps the adversarial legal sample $X'$ at the length of $X$ (Eq. 18), applies the modulo perturbation of Eq. 15 to a DGA SLD and reattaches its TLD; pairs whose label breaks RFC 1034/1035 are redrawn a bounded number of times.

### CDGA (`models/cdga/`)
From *"CDGA: A GAN-based Controllable Domain Generation Algorithm"* (Zhai et al., 2022). GAN-based controllable DGA that generates stealthy domain names that comply with RFC domain rules while ensuring zero repetition by combining a Wasserstein GAN with Gradient Penalty (WGAN-GP), a Sequence Autoencoder, and Controllable Text Generation (CTG).  
- `fit()` trains a SentencePiece tokenizer (unigram) and a Word2Vec embedding model (skip-gram) with SLD labels from benign domains. Trains an LSTM-based Autoencoder, which is then used to adversarially train the WGAN-GP's MLP generator and ResNet critic. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Generation utilizes an autoregressive inference decoder with state reuse, explicitly setting probabilities to 0 on invalid tokens to enforce structural validity and bypass duplicates. 
- Caches trained artifacts into `.h5` files and short-circuits when they already exist on disk.

### GWDGA (`models/gwdga/`)
From *"GWDGA: An Effective Adversarial DGA"* (Shu et al., 2022). A VAE and GCNN-based adversarial DGA that generates domains by modeling the statistical distribution of benign subdomains using a vocabulary composed by the most frequent n-grams and words.  
- `fit()` loads benign training data from the D1 Tranco slice. It dynamically builds a vocabulary (GWDict) using frequent n-grams and a local Lean Domain Search (LDS) list (`dataset/fixes.md`, obtained from https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103), implementing a fallback to dataset frequencies if the LDS file is missing. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches the learned vocabulary and the trained neural network weights in local `.h5` files.  Short-circuits when both of these cached artifacts already exist on disk.

### WGANBasedDGA (`models/wgan_based_dga/`)
From *"A WGAN-Based Method for Generating Malicious Domain Training Data"* (Zhang et al., 2022). WGAN-based malicious domain generator that generates domain hostnames by modeling character distributions via a Wasserstein Generative Adversarial Network paired with a custom ASCII-based character encoder/decoder mapping.  
- `fit()` loads malicious domain samples, isolates hostnames, encodes them into 30-dimensional normalized numeric vectors and trains the neuronal network. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches the generator weights in a local `.h5` file, short-circuiting when cached weights exist.

### ReplaceDGA (`models/replacedga/`)
From *"ReplaceDGA: BiLSTM-Based Adversarial DGA With High Anti-Detection Ability"* (Hu et al., 2023).
Character-level and BiLSTM-based adversarial DGA that generates domains by modifying two characters from benign base domains using a neural language model.
- `fit()` loads data from TRANCO_D1_BENIGN, filters valid characters, and splits data into training and generation sets. Trains a BiLSTM network (Adam, up to 50 epochs). It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches model weights to a `.pth` file, short-circuiting if cached.
- Generates domains performing 2 model-guided character replacements on a base domain and appends a common TLD.

### PKDGA (`models/pkdga/`)
From *"PKDGA: A Partial Knowledge-Based Domain Generation Algorithm for Botnets"* (Nie et al., 2023). Partial Knowledge-Based DGA that uses Reinforcement Learning to generate adversarial SLDs that evade a target detector by learning from feedback.  
- `fit()` loads benign training data from the D1 Tranco slice, extracting and filtering SLDs based on a supported 39-character vocabulary. Executes a two-phase training process: Maximum Likelihood Estimation (MLE) pre-training, followed by 60 batches of adversarial RL training using Monte Carlo rollouts. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- If no target detector is explicitly provided, it automatically instantiates a default CNN-based detector to act as the reward oracle.    
- Caches the learned generator weights in a local `.trc` file, short-circuiting when the artifact already exists on disk.   
- `generator.py` and `rollout.py` were extracted from the repository the authors provided (https://github.com/abcdefdf/PKDGA) and slightly modified to adapt the code to the framework.

### WordDGA (`models/worddga/`)

From *"WordDGA: Hybrid Knowledge-Based Word-Level Domain Names Against DGA Classifiers and Adversarial DGAs"* (Selvaraj and Panjanathan, 2024). cWGAN-based adversarial DGA that combines a Conditional Wasserstein GAN with Gumbel-Softmax differentiable sequence generation, RCNN-BiLSTM feature extraction, and NLP/statistical heuristics.
- `fit()` loads benign SLDs from the D1 Tranco slice (`TRANCO_D1_BENIGN`) and malicious SLDs from DGArchive, building a character $n$-gram model ($n=3..7$) for statistical reputation scoring. It records its training time and `trained_from_scratch` the same way DeepDGA does.
- Caches generator weights and known DGA domains in local `.h5` files, short-circuiting when artifacts exist on disk.
- Instantiates a default `CNNDetector` target if none is provided to guide domain generation via black-box iterative evasion and collision avoidance.
- Generation follows Algorithm 1 with a different base DGA SLD per output and the 50 candidate words drawn from the generator and scored by the target classifier in a single call each.

### TLVDGA (`models/tlvdga/`)
From *"Adversarial Sample of Domain Generation Algorithm Based on Variational Autoencoder"* (Nangong and Wu, 2025). TLVDGA model that generates domains utilizing a Variational Autoencoder (VAE) architecture that combines a Transformer Encoder, a LSTM Decoder, and Word2Vec embeddings.  
- `fit()` loads benign training data from the D1 Tranco slice and extracts the SLDs to process the sequences.  It builds an n-gram dictionary using the NLTK library, segments the domains using Bi-direction Maximum Matching (BiMM), and trains both the Word2Vec embeddings and the VAE model. It records its training time and `trained_from_scratch` the same way DeepDGA does.  
- Caches the learned vocabulary and VAE weights in different local `.h5` files.  Short-circuits when these artifacts already exist on disk.

### TitanDGA (`models/titan_dga/`)
From *"TITAN DGA: Enhancing DGA Evasiveness through a Transformer-based Autoencoder and Adversarial Self-Augmentation"* (Pregardier et al., 2025). GAN-based adversarial DGA that combines a Transformer Autoencoder and a GAN to generate stealthy domains trained on benign distributions.
- `fit()` loads benign training data from the Tranco slice, limited to a sample of 10,000 domains. Extracts SLDs, trains a SentencePiece unigram tokenizer, and executes a two-phase training process combining initial transformer-GAN training and targeted self-augmentation against the framework's CNN detector (`detectors/cnn/`). It records its training time and `trained_from_scratch` the same way DeepDGA does.  
- Caches trained weights into a `.h5` file and SentencePiece artifacts, short-circuiting when they already exist on disk.
- Uses a fixed list of the 50 most common traditional TLDs registered globally, as well as the 50 most frequent new gTLDs, based on statistics provided by https://domainnamestat.com/.

### SADGA (`models/sadga/`)
From *"SADGA: A Self Attention GAN-Based Adversarial DGA with High Anti-detection Ability"*  (Luo et    al., 2026). SAGAN-based adversarial DGA that integrates a self-attention mechanism into the CNN-based WGAN framework.

- `fit()` loads the first 10,000 benign domains from the D1 Tranco slice. It dynamically builds a vocabulary (SADict) using frequent n-grams and a local Lean Domain Search (LDS) list (`dataset/fixes.md`, obtained from https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103), implementing a fallback to dataset frequencies if the LDS file is missing. It records its training time and `trained_from_scratch` the same way DeepDGA does. 
- Tokenizes domains using Bidirectional Maximum Matching (BiMM), choosing the sequence with fewer tokens and favoring Forward Maximum Matching (FMM) in case of a tie.  
- Caches its generator weights and vocabulary in local `.h5` files, short-circuiting when these artifacts already exist on disk.

## Control models

Not adversarial DGAs in the paper's sense. They give the evaluation pipeline
a defined positive and negative reference.

### `MaliciousDGAModel` (`models/malicious_dga/`)

Loads real-world DGA family samples from the D3 DGArchive slice
(`DGARCHIVE_D3_MALICIOUS`, rows `[50_000, end)` of each
`dataset/dgarchive/<family>.csv`). One instance per family; `main.py`
currently uses **Dyre**, **Suppobox**, **Qakbot**, **Rovnix**, **Conficker**,
**Cryptolocker**, **Banjori**, **Symmi**, **Gozi** and **Bamital**.

- Constructor takes `name` and `family` (CSV file name under the DGArchive
  directory).
- Used as the malicious control group, i.e. the per-family detection
  baseline.
- `generate_domains(n)` draws without replacement (`random.sample`) from the
  slice. If the slice holds fewer than `n` domains, it warns and returns the
  whole slice shuffled.

### `BenignDomainsModel` (`models/benign_domains/`)

Loads benign domains from the D3 benign split (`TRANCO_D3_BENIGN`,
`dataset/splits/D3_benign.csv`, 100,000 domains). Used as the benign control
group, i.e. the baseline for detector false-positive rate.

Sets `is_malicious = False` so the detection analysis scores it against benign
ground truth (see `docs/analysis.md`).

## Dummy models

Three dummy baselines, `RandomDummyModel`, `HashDummyModel`,
`DictionaryDummyModel`, live under `models/random_dummy/`,
`models/hash_dummy/`, `models/dictionary_dummy/`. They are floor references
and are not wired into `main.py`.

## Adding a new model

1. Create `models/<your_model>/<your_model>.py`.
2. Inherit from `AdversarialModel` and implement `fit`, `generate_domain`,
   `generate_domains`.
3. Make `fit()` fully self-contained: read its own dataset slice, write its own
   weights into the module's directory.
4. Import and register it in `main.py`.
