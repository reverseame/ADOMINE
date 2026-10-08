# Assumptions

## Fu et al., 2017

### HMM-based DGA

#### Information Gaps

* **Stopping mechanism and domain length:** The paper mentions the use of domains between 3 and 10 letters in its experiments (to mimic the IPv4 space), but omits how length limits are dynamically handled in other contexts, and does not specify whether an explicit End-of-String state exists in the Markov chain.

* **Hidden inference algorithm:** The authors mention the use of a *zero-knowledge HMM inference algorithm*, relying on external references, but they do not provide mathematical details, the exact formulation of the network, or explain how the hidden states are mapped.

* **Zero probabilities and edge cases:** The paper does not specify what happens if the model reaches a combination of characters that never appeared in the training set and therefore lacks registered transitions.

* **Context initialization:** It is not documented how the initial context is supplied when generating the first characters of a domain when there are not yet enough previous characters (history length *L*) available in the chain.

* **TLD (Top-Level Domain) management:** The study focuses on the properties of the domain name (SLD), but does not detail the TLD selection process (e.g., `.com`, `.net`) required for the generated domain to be resolvable and functional within an evaluation framework.

---

#### Assumptions

* **Domain length:** Since the paper only indicates that the length used in its experiments for each domain is between 3 and 10 letters, this range is assumed to be the default but configurable through parameters (for example, from the `main.py` file). The length is then selected randomly within this range. If no range is provided (by passing `None` as parameters), instead of using a simple average, the model dynamically assumes the empirical percentiles of the benign dataset (5th percentile for the minimum and 95th percentile for the maximum, bounded between 3 and 20 characters). In this way, the HMM is forced to traverse as many states as the calculated domain length, without generating artificial biases in detection.

* **Training (architecture):** Since the key parameter in the paper is *L* (symbol history), the architecture has been simplified by using a visible Markov model based on explicit N-grams, which is equivalent to the zero-knowledge inference described. If `L = 1` is requested, the model uses the probabilities of the bigrams extracted from the benign domains. If `L = 2` is requested, trigrams are used. In this way, character transitions are simulated accurately.

* **Zero probabilities:** Since the paper does not specify how to deal with unseen events, Laplace Smoothing has been applied (adding an artificial +1 to all possible transition counts in the observed vocabulary). In this way, no transition probability will be exactly zero. This prevents division-by-zero errors and code failures, allowing the DGA to retain a certain degree of randomness.

* **Context initialization:** In order to compute the probability of the first letter, the use of a padding token (`^`) repeated *L* times at the beginning of each domain during both the training and generation phases has been assumed (e.g., `^^^google`). This ensures that the history always has the correct length.

* **TLDs:** It is assumed that random TLDs are used. In the code (`fit`), TLDs and dots are removed from benign domains in order not to contaminate the Markov matrices. During the generation phase, a static list of 22 common TLDs (such as `.com`, `.org`, `.ru`, etc.) is assumed, and one is randomly selected and concatenated to the end of the generated text string.

* **Training data:** Although the paper details the use of lists with different characteristics to train the model, the Tranco dataset has been used for consistency with the framework.

### PCFG-based DGA

#### Information Gaps

* **Exact production rules and probability vector (ፀ):** The paper provides a very simple example based on parenthesis matching (Figure 3), stating that terminals originate from two disjoint syllable lists (*A* and *C*). However, it does not provide the actual grammar rules or the probability vector (ፀ) used in the final experiments.

* **Hyphenation algorithm:** The authors mention that they extract syllables from benign domains using the algorithm of a previous research DGA called *Kwyjibo* (Reference [48]). They do not explain how this algorithm works or how they deal with unpronounceable domains (e.g., `xnxxt`, or those containing numbers), which are impossible to syllabify using traditional English rules.

* **Depth control (branching rate):** The authors indicate that "the branching rate must be less than 1" so that domains have a finite length, but they do not specify how forced termination is implemented if the tree starts growing uncontrollably, generating excessively long domains. Furthermore, they do not indicate which domain length is used for this DGA (for the HMM-based DGA they mention between 3 and 10 letters, and for the Kwyjibo algorithm, which is also syllable-based, a limit of 2 to 4 syllables is used).

* **Insertion of numbers and special characters:** The authors mention the use of dictionaries such as `pcfg_ipv4_num` (syllables + numbers from 0 to 2000), but they do not indicate how these characters are included in syllables, since they are not pronounceable.

* **TLD (Top-Level Domain) management:** The study focuses on the properties of the domain name (SLD), but does not detail the TLD selection process (e.g., `.com`, `.net`) required for the generated domain to be resolvable and functional within an evaluation framework.

---

#### Assumptions

* **Production rules and probability vector (ፀ):** To ensure that the branching rate is lower than 1, the production rules included in the example (Figure 3) are assumed, slightly modifying the first and the last rules to avoid generating empty strings and wasting generation cycles discarding them:

```text
S → B
B → aBc (p = 0.25)
B → ab  (p = 0.75)
```

* **Domain length control:** It is assumed that generated domains must contain between 2 and 4 syllables, as mentioned in Reference [48] (Kwyjibo). If the probabilistic tree generates a structure outside this range, or if the resulting final string exceeds reasonable length thresholds (taking the range of 3 to 10 characters), it is rejected and regenerated. These ranges are also configurable through parameters.

* **Syllable extraction (hyphenation) and training:** It is assumed that the model must be trained by partitioning the benign domains provided in the Tranco list (D1) into English-language syllables. To avoid introducing dependencies by using libraries such as `pyphen`, regular expressions are used to split syllables and add them to the mutually exclusive sets *A* and *C*. It is also assumed that the distribution between lists *A* and *C* is performed randomly with a 50/50 ratio.

* **Numeric insertion and TLDs:** To support the `pcfg_ipv4_num` variant mentioned by the authors, the injection of numbers (0–2000), each repeated 20 times, into list *C* is assumed, in order to balance their probability of occurrence with that of the syllables.

* **TLDs:** It is assumed that random TLDs are used. During the generation phase, a static list of 22 common TLDs (such as `.com`, `.org`, `.ru`, etc.) is assumed, and one is randomly selected and concatenated to the end of the generated text string.

## Yilmaz et al., 2020

### FGSM-based DGA

#### Information Gaps

* **Classifier hidden size:** Although the authors describe the use of a 256-dimensional Embedding layer and two LSTM layers, they do not specify the number of hidden units in the recurrent layers. This parameter is critical for the model's learning capacity and overall complexity.
* **Sequence length and padding handling:** Neural networks require fixed-size matrices (tensors). The paper does not describe the text normalization strategy, omitting both the maximum sequence length and the padding scheme used.
* **Constraints on cosine similarity projection:** After injecting the continuous perturbation ε into the latent space, the method uses cosine similarity to project the representation back into discrete characters. However, the paper does not clarify whether the resulting ASCII space is constrained to prevent symbols that are invalid under the DNS protocol (such as punctuation marks, uppercase letters, or spaces).
* **TLD (Top-Level Domain) preservation:** The authors do not specify whether the algorithm perturbs the entire domain (including extensions such as `.com`) or only the second-level domain. Modifying the TLD would reduce the realism of the attack by generating non-existent extensions that would be naturally rejected by real DNS servers.
* **Seed handling and origin:** As an adversarial gradient-based attack (FGSM), the model does not generate text from scratch through grammatical rules. Instead, it mutates existing domain names. The paper does not explain where these seed domains originate or how they are managed during large-scale generation.
* **Dataset scale vs. mass generation:** The authors do not clarify the operational procedure when the framework requests a number of adversarial domains (`num_domains`) greater than the number of malicious seed domains available in the local files.
* **Caching strategy for White-Box models:** The paper focuses on the theoretical attack but does not describe a persistence mechanism for storing the trained architecture, allowing the framework to avoid retraining the internal classifier every time the global script is executed.
* **Mathematical gradient objective (attack direction):** The paper generically mentions classifier evasion but does not provide an analytical formulation of the loss function, nor does it specify whether the attack is targeted (toward a specific class) or untargeted (away from the true class).

---

#### Assumptions

* **Network architecture (LSTM units):** A standardized configuration of 128 hidden units per LSTM layer is assumed. At the time of publication (2020), the architecture `Embedding(256) -> LSTM(128) -> LSTM(128)` represented the de facto standard for short-text binary classification tasks due to its balance between convergence performance and computational cost.
* **Sequence length and padding:** A standardized maximum length of 63 characters is applied exclusively to the SLD (Second-Level Domain), corresponding to the DNS protocol limit for a single label. Shorter domains receive post-padding (right-side zero padding) using the special `<PAD>` token (index 0) to produce homogeneous input matrices. This padding is removed when reconstructing the final domain in order to preserve the original seed length. Additionally, if the first or last character of the domain is a hyphen, it is replaced with the original first or last character of the domain, respectively.
* **Vocabulary projection (DNS filter):** The cosine similarity search space is strictly restricted to valid DNS alphanumeric characters and hyphens (`[a-z0-9\-]`). To prevent invalid outputs, a mathematical penalty mask of `-1e9` is applied to the padding index and any prohibited symbol, forcing the geometric projection toward the nearest valid DNS character.
* **TLD isolation:** The model separates the SLD from the TLD before computing gradients. Gradient computation through `GradientTape` and the subsequent mutation process operate exclusively on the SLD. Seeds are cached without their TLD, so every generated domain gets `.com` appended after the mutation stage. This guarantees formal validity of the generated domains.
* **Seed management and caching strategy (.h5):** Following the framework's logic, the model loads real malicious domains from DGArchive CSV files during the `fit()` method. A hybrid persistence mechanism is assumed, encapsulating the complete set of malicious seed domains and the classifier weights within two different `.h5` files using `h5py`. Stored byte sequences are explicitly decoded to UTF-8 to prevent Python type corruption issues. If no seeds are available, malicious domains are generated using another DGA model.
* **Dataset scaling (sampling without replacement):** `generate_domains(n)` walks a shuffled copy of the cached seed list and uses each seed once per pass, reshuffling only when `n` exceeds the number of seeds. The gradient-sign perturbation is deterministic for a given seed, so drawing seeds with replacement would repeat outputs. `generate_domain()` on its own still draws a single seed at random.
* **Training data:** For consistency, the internal classifier is trained using a balanced 50/50 dataset composed of benign samples from `TRANCO_D1_BENIGN` and real malicious domains extracted from the CSV files located in `dataset/dgarchive`.
* **Mathematical gradient objective:** The attack is explicitly implemented as an untargeted evasion attack. Gradients are computed using `tf.GradientTape` to maximize the Binary Cross-Entropy (BCE) loss function under the static assumption that the true label is malicious (`y = 1`). This mathematically pushes the character embeddings away from the malicious class, producing mutations that the classifier is more likely to confuse with benign distributions.

## Alaeiyan et al., 2020

### GADGA

#### Information Gaps

* **Exact composition of the Fitness Function:** The paper states that the fitness function combines a pronunciation score and a chi-square test to balance evasion capability and randomness, but it does not provide the exact mathematical formulation nor the scales used to combine both metrics.
* **Mutation rate and magnitude:** The authors indicate that mutation replaces characters with random letters or symbols, but neither the mutation probability nor the number of characters modified per mutation event is specified.
* **Parent selection strategy:** The pseudocode uses a `selectTwoRandomSample(population)` function without clarifying whether parent selection is purely uniform or weighted according to individual fitness values.
* **Crossover with variable-length domains:** The paper states that single-point crossover is applied at random locations. Since domain names have variable lengths, it does not explain how crossover indices are calculated when parents have different lengths, nor how out-of-range errors are avoided.
* **Contradiction between population dize and iteration cycles:** The pseudocode describes an internal loop iterating up to a `maxGen` value of 400. If two offspring are generated per iteration, this would produce up to 800 individuals, contradicting the fixed population size of 100 specified elsewhere in the paper.
* **DNS syntax handling:** The genetic operators freely modify characters, but the paper does not describe any mechanism to prevent invalid domain names (e.g., leading or trailing hyphens), nor does it explain how TLDs are managed.
* **Initial seed population and elite retention:** The authors do not specify how the initial set of 100 domains is sampled from the global malware dataset without introducing family bias. Furthermore, although the top 20% of the population is retained through elitism, the paper does not explain how duplicate individuals are handled when filling the remaining 80% of the population.

---

#### Assumptions

* **Fitness Function:** A weighted linear combination is assumed using the formula: 

    F(x) = α · S_pronunciation − β · χ²_normalized 

    The pronunciation score is computed using the log-probability of a bigram language model trained on benign Tranco domains. The chi-square value is normalized by dividing it by the maximum possible value for the given domain length, ensuring both components operate on comparable scales.
* **Mutation:** A conservative mutation rate of 5% (`0.05`) is assumed. Whenever a mutation occurs, exactly one randomly selected character within the domain is modified.
* **Parent selection:** The paper's description is interpreted literally by using pure uniform random sampling (`random.sample`) for parent selection. Selection pressure is therefore introduced exclusively through the elitism mechanism that retains the top 20% of the population.
* **Crossover:** A crossover point `k` is randomly selected within the interval: [1, min(len(p1), len(p2)) - 1] This guarantees a valid crossover operation and prevents out-of-range indexing errors when parents have different lengths.
* **Generation cycle and population size:** To resolve the inconsistency between the reported population size and the generation procedure, the literal interpretation of the 400-step multiplier is discarded. Instead, a `while` loop controlled by a safety threshold (`max_attempts = 2000`) continuously generates offspring until exactly the 80 missing population slots are filled. If the population shrinks prematurely, elite individuals are cloned to maintain population size.
* **DNS syntax and TLD handling:** The genetic algorithm isolates and mutates only the SLD (Second-Level Domain). Any leading or trailing hyphens generated during mutation are removed using `strip('-')` to preserve DNS validity. The TLD is assigned randomly from a list of 22 common extensions during the final domain generation stage.
* **Initial seed population (`initPop`):** The initial population is obtained through uniform random sampling without considering malware family labels. A fallback mechanism injects algorithmically generated domains whenever insufficient samples are available.
* **Duplicate handling and domain generation:** During evolution, a `set` data structure is used to ensure that all offspring remain unique. Additionally, for performance reasons, the genetic algorithm is not executed on every generation request. Instead, it is trained once, the final evolved population is cached, and each new domain request retrieves a random SLD from this cached population. A minor perturbation (50% probability of modifying one internal character) is then applied to provide unlimited variability and maintain evasion capability while avoiding collisions.
* **Zero-probability handling:** Laplace smoothing is assumed by adding a pseudo-count of `+1` to all possible first-character and bigram frequency counts.

## Gould et al., 2020

### WGAN-GP DeepDGA

#### Information Gaps

* **Layer dimensions:** The paper describes the use of Embedding, LSTM, parallel Conv1D layers (with filter sizes 2 and 3), and a Dense layer, but omits the embedding dimensionality, the number of LSTM memory units, and the number of convolutional filters.

* **Regularization and optimization hyperparameters:** The use of a Dropout layer, the Adam optimizer, and a WGAN-GP gradient penalty multiplier is mentioned, but the exact dropout rate, learning rate, and momentum parameters (β1 and β2) are not provided.

* **Training dynamics:** The paper does not specify the training batch size, the total number of epochs, or the discriminator-to-generator update ratio (*ncritic*).

* **Activations and initialization:** The activation functions used in the hidden layers and the weight initialization strategy are not documented.

* **The discrete output problem:** Traditional GANs generate continuous values, whereas domain names are discrete text sequences. The paper does not technically explain how this gap is bridged while maintaining differentiability during training.

* **Padding strategy and data representation:** The paper does not specify how domains shorter than the maximum length (63 characters) are padded, nor how real input domains are represented mathematically.

* **Handling invalid domains:** The inference-time behavior is not described when the model generates syntactically invalid DNS structures (e.g., domains that are too short or begin/end with hyphens).

---

#### Assumptions

* **Architecture and dimensions:** A batch size of 128 is assumed, with 64 units in the LSTM layer and 32 filters in each parallel Conv1D branch (kernel sizes 2 and 3 with `same` padding). The embedding dimension is fixed at 32 in order to represent a vocabulary of 38 symbols (36 alphanumeric characters, one hyphen, and one padding token).

* **Hyperparameters and optimizer:** To stabilize adversarial training, the Adam optimizer is assumed with a learning rate of `0.0001`, `β1 = 0.0`, and `β2 = 0.9`. The gradient penalty coefficient is set to `10.0`. A Dropout rate of `0.5` is assumed in the discriminator to reduce overfitting.

* **Training dynamics:** The model is assumed to be trained for a total of 100 epochs. The update ratio is set to `ncritic = 5`, meaning that the discriminator is trained five times for every generator update.

* **Activations and initialization:** LeakyReLU activations with a negative slope of `0.2` are assumed for all hidden layers. Weight initialization is assumed to follow the generalized `he_normal` strategy throughout the network.

* **Real data representation and discrete output handling:** The generator input noise is assumed to be a vector of 63 integers with values ranging from 0 to 37. To enable comparison between the generator's probabilistic output and real domain names, benign domains from the Tranco dataset are assumed to be transformed into one-hot encoded matrices of shape `(63, 38)`.

* **Padding strategy:** A post-padding approach is assumed, where a dedicated padding token (index `37`) is appended to the right side of each sequence until the maximum length of 63 characters is reached.

* **Domain generation (inference):** A rejection sampling mechanism is assumed during domain generation. The implementation performs up to 20 attempts to generate a valid domain. A domain is considered valid if, after removing any leading or trailing hyphens, its length is at least 5 characters. If all attempts fail, a random lower-case label is returned and a warning is logged.

* **TLD (framework convention):** The paper defines the generator on the second-level label only. Because the detectors are trained on full domains and the lexical metrics change with the TLD, the framework appends one to every generated SLD, drawn from the empirical public-suffix distribution of the D1 Tranco slice (top 50, `core/tld.py`). The same rule applies to every model in this situation so that the TLD is not a confounder between generators.

## Yun et al., 2020

### Khaos

#### Information Gaps

* **Training hyperparameters and WGAN-GP:** The paper details the use of a Wasserstein GAN architecture with gradient penalty (WGAN-GP), but omits critical details such as the specific optimizer, learning rate, batch size, gradient penalty coefficient, and the update ratio between the discriminator and the generator.

* **Initialization and normalization:** The article omits how the network weight matrices are initialized, a factor to which GANs are highly sensitive during early training stages. Furthermore, it does not specify the strict normalization rules (Batch vs. Layer Normalization) required by WGAN-GP mathematics to ensure that the gradient penalty is not compromised in the discriminator.

* **Dimensionality preservation and activations:** Although it is stated that the output shape maintains a constant spatial dimension (*ml*) throughout the convolutional layers, the paper does not specify which technique is used to prevent dimensional shrinkage when applying filters. It also does not specify the activation functions used in the first dense layer or in the layers outside the residual blocks.

* **Vocabulary and tensor handling:** The paper mentions a dictionary size of 5,000 n-grams, but does not explain how explicit space is reserved for the padding required to equalize sequences to the maximum length. It also does not address the mathematical incompatibility of mixing categorical real data (discrete) with probabilistic fake data (continuous) in the discriminator inputs.

* **Decoding, sanitization, and edge cases:** The paper does not define how the probabilistic numerical tensors returned by the generator are transformed into readable text strings, nor does it detail strict character limits to comply with RFC standards or how to handle domains that generate abnormally short lengths. Additionally, it lacks stopping criteria and dataset partitioning proportions.

---

#### Assumptions

* **Training hyperparameters:** To address the omissions in the original paper, the Adam optimizer is assumed with a standard learning rate of `lr=0.0001` and parameters `beta_1=0.5` and `beta_2=0.9`. A `batch_size=64` is used, with a gradient penalty coefficient (`gp_weight`) of `10.0`, and the discriminator is updated 5 times (`d_steps=5`) for every generator update in order to maintain the Lipschitz constraint. Training is performed for a fixed number of 100 epochs.

* **Initialization and normalization:** To prevent gradient explosion or vanishing gradients, a RandomNormal initializer with mean `0.0` and standard deviation `0.02` is applied to all layers. The Discriminator exclusively uses LayerNormalization to compute statistics independently for each sample and preserve the gradient penalty, while the Generator retains BatchNormalization.

* **Architecture and dimensionality:** To maintain a constant spatial dimension (*ml*), Zero-Padding (`padding='same'`) is assumed in all Conv1D layers. In the Generator, the original seed is projected through a dense layer with ReLU activation into a dimension of `ml * 64`, and is subsequently reshaped into a 3D tensor. The generator's convolutional output uses a linear activation before the Softmax function.

* **Dictionary size and padding:** The dictionary size (*ds*) is defined as exactly `5,001`. The first 5,000 indices correspond to the most frequent n-grams (from 1 to 4 characters) extracted from the dataset, while index 5,000 is reserved as a special token exclusively for right-padding. A maximum number of tokens `ml=15` is assumed, sufficient to remain within domain character limits.

* **Continuous-discrete representation:** The Discriminator is designed to always process continuous matrices. Real domains are converted into matrices using One-Hot Encoding (values `0.0` and `1.0`), which are then mathematically combined in the WGAN-GP loss with the floating-point probabilities directly emitted by the Generator's Softmax layer.

* **Decoding and final formatting:** During inference, the use of a mathematical operation (Argmax on the last axis) is assumed to extract the most probable indices from the tensor while ignoring padding tokens during decoding. Generated domains are truncated to a maximum of 63 characters, and leading or trailing hyphens are removed to comply with RFC standards. If the decoded domain contains fewer than 3 characters, a safety padding mechanism is assumed by concatenating valid random n-grams extracted from the dictionary until the minimum length is reached. Finally, the addition of the `.com` TLD is assumed.

* **Training data:** The Tranco D1 Benign dataset is used, assuming rigorous cleaning through regular expressions (`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`) and discarding domains whose length exceeds the 95th percentile (capped at 63 characters). For direct training, 80% of the cleaned data is allocated to the training set.

## Cao et al., 2020

### DnGAN

#### Information Gaps

* **Latent space dimensions and network architecture:** The paper indicates that the generator uses a random seed based on a Gaussian distribution, but omits the size of this noise vector ($Z$). Likewise, it mentions the use of a two-layer LSTM network, but does not specify the number of units in these layers or in the subsequent dense layers.

* **Optimization and training hyperparameters:** There is no mention of the optimizer used, its learning rate, momentum values, or the activation functions of the intermediate layers. Nor is an exact batch size provided, referring only to "small batches".

* **Continuous-to-text transformation and decoding:** Although the paper proposes generating continuous matrices, it does not detail the exact technical formulation used to project the generator output into the discriminator embedding space without breaking gradient flow. Furthermore, it does not specify how variable domain lengths are dynamically handled (padding), nor the process for generating domains shorter than the maximum length (30).

* **GAN phase and data mixing:** The use of historical samples and multiple batches to enforce diversity is mentioned, but the exact proportions used to feed the discriminator are not detailed, nor is the capacity of the historical domain buffer.

* **TLD management and formatting:** The process for selecting Top-Level Domains for the final evaluation is not documented, nor is the specific handling of invalid characters.

---

#### Assumptions

* **Latent space and architecture:** A noise vector ($Z$) with a dimension of 100 is assumed, aligning with the historical standard in text GANs. The two-layer LSTM networks in both the Generator and the Discriminator are set to 128 units in order to maintain dimensional consistency. The intermediate Dense layer contains 64 units, and the Generator output layer is assumed to have 128 units to match the continuous representation dimension.

* **Optimization and training:** The Adam optimizer is used with a learning rate of 0.0001 and a $\beta_1$ value of 0.5. The batch size is assumed to be 32 for both the adversarial phase and the pre-training phase. LeakyReLU with a negative slope of 0.2 is used for intermediate activations. In addition, the Label Smoothing technique is assumed, setting real labels to 0.9 instead of 1.0 in order to prevent the discriminator from becoming overconfident and to mitigate mode collapse.

* **Continuous representation and decoding:** It is assumed that the generator produces a softmax distribution over the 128 ASCII characters (indices 0–127), rather than a sigmoid output. This distribution is projected into a continuous embedding space through a bias-free Dense layer that shares weights with the embedding layer used for real data. For decoding, instead of always generating 30 characters, a length is randomly selected between 1 and the maximum length (30). Invalid characters (anything other than `a-z`, `0-9`, or `-`) are randomly replaced, and leading or trailing hyphens are removed. Regarding padding during training, post-padding with a value of 0 is assumed to fill the remaining positions up to 30 characters.

* **GAN phase and historical buffer:** The implementation of a FIFO buffer with a capacity of 10,000 historical samples is assumed. Each discriminator batch is divided according to strict proportions: 50% real domains, 25% newly generated fake domains, and 25% historical fake domains sampled from the buffer. For pre-training, 5 preliminary epochs are assumed using a balanced dataset composed of 5,000 benign domains and 5,000 DGA domains. For training, 100 epochs are assumed.

* **TLDs and training data:** The use of a static list of 22 common TLDs is assumed, from which one is randomly selected and concatenated to the end of the generated domain.

## Zheng et al., 2021

### ShadowDGA

#### Information Gaps

* **Latent space dimensionality and convolutional hyperparameters:** The paper indicates that the generator samples noise from a normal distribution and that both models use convolutional layers with residual blocks. However, it omits the exact dimensionality of the input noise vector, as well as the number of filters, kernel size, and padding configuration of the Conv1D layers.

* **Vocabulary encoding and sequence length:** The manuscript proposes the direct generation of second-level domain names (SLDs). However, it lacks details on how real benign domain characters (Ground Truth) are encoded to feed the discriminator, the exact size of the allowed character vocabulary, and how variable-length domains are handled in batch-based convolutional architectures.

* **Differentiability mechanism for discrete text:** Since traditional GANs cannot backpropagate gradients through discrete character indices, the paper does not indicate how this issue is addressed. It does not clarify whether any specialized technique (such as Gumbel-Softmax or reinforcement learning) is employed, or whether a continuous relaxation is used during training.

* **Normalization and residual block structure:** A simple residual connection is described. However, it is not specified whether channel or batch normalization is applied. In WGAN-GP architectures, Batch Normalization in the discriminator is discouraged because it alters the gradient penalty.

* **Loss parameters and discriminator activation function:** The paper defines the use of gradient penalty to stabilize WGAN-GP training. Nevertheless, it omits the specification of the penalty coefficient ($\lambda$) and the final activation function of the discriminator's last dense layer, which mathematically should be linear in order to emit continuous critic values rather than binary probabilities.

* **Validation:** The paper does not specify whether a validation mechanism should be used to ensure that the characters structured by the model comply with RFC network formatting standards.

---

#### Assumptions

* **Latent space and convolutions (Architecture):** A 100-dimensional input noise vector is assumed. For the Conv1D layers, 128 filters with a kernel size of 3 and `'same'` padding are configured in order to preserve temporal dimensions throughout the residual blocks. In the generator, the initial dense layer projects the 100-dimensional noise vector into a tensor of size `max_len * channels` (`16 * 128`) to be reshaped into a format compatible with the first convolution.

* **Encoding, length, and vocabulary:** A one-hot encoding scheme is assumed under a static vocabulary of 38 characters (26 lowercase letters a-z, 10 digits 0-9, the hyphen `-`, and index 0 reserved for the PAD padding token, which is mapped to an empty string during decoding). The sequence length is fixed at 16 characters. During output processing, any character matching the padding token or appearing after it is ignored in order to dynamically truncate the domain.

* **Differentiability and gradient flow:** During training, a continuous relaxation is applied in which the generator directly sends its continuous matrix of smoothed probabilities (obtained through a final softmax activation with a kernel-1 Conv1D layer) to the discriminator. During inference (actual generation), an argmax operation is applied over the vocabulary axis to discretize and reconstruct the domain as plain text.

* **Normalization and residual blocks:** A constant width of 128 channels is established for all internal convolutional layers across the 5 residual blocks, enabling direct addition of the residual connection without the need for intermediate convolutional projections. Furthermore, any type of normalization (such as Batch Normalization) is completely omitted from the discriminator to avoid interfering with the WGAN-GP gradient penalty computation.

* **WGAN-GP loss and linear output:** The standard value from the literature ($\lambda = 10.0$) is used to weight the gradient penalty in the discriminator loss. For mathematical consistency with the WGAN-GP algorithm, the discriminator's final dense layer uses a strictly linear activation in order to act as a continuous distance critic.

* **TLDs, validation, and training:** It is assumed that training is performed by extracting and processing benign SLDs from the Tranco dataset (filtering out those whose length is not between 6 and 16 characters or that contain characters outside the allowed vocabulary). During generation, the validity of the resulting SLD is verified through a regular expression that ensures compliance with RFC standards. If the domain is valid, a TLD selected at random from a static list of 8 common extensions (`.com`, `.net`, `.org`, `.info`, `.biz`, `.ru`, `.in`, `.cc`) is appended. Any incorrectly generated domain is automatically discarded and replaced through a continuous generation loop.

* **Training parameters:** Training is fixed at 5000 iterations with a batch size of 64, using Adam optimizers with a learning rate of `0.0001`, $\beta_1 = 0.5$, and $\beta_2 = 0.9$. The discriminator is updated 10 times for every generator update (`n_critic = 10`).

## Wang and Guo, 2021

### NDG

#### Information Gaps

* **GCNN filter dimensions and dense layers:** The paper indicates the use of stacked GCNN layers (kernel size 3, stride 1, zero padding) and a dense layer after Global Average Pooling (GAP). However, it omits the number of filters (channels) and the size/activation of the intermediate dense layers before compressing into the latent space $N = 16$.

* **Residual connections:** In the GCNN architecture, it is not specified how to resolve the mathematical incompatibilities that arise when adding the residual connection if the input channel dimensions differ from the convolutional filters.

* **Vocabulary and padding management:** A vocabulary of 39 characters and a maximum length of 30 are defined. However, it is not specified whether the text processing is case-sensitive and whether the padding character is applied at the beginning or at the end of the sequence.

* **Reconstruction loss implementation:** The reconstruction loss equation is theoretically defined as a logarithmic expectation, but the equivalent programmatic loss function is not mentioned, nor is it explained how it is scaled relative to the KL divergence.

* **VAE hyperparameters and epoch limit:** The authors detail the use of the Adam optimizer (`lr = 0.001`), a batch size of 256, and early stopping based on accuracy, but do not clarify whether this applies to the DGA generator or the classifier. They also do not specify the overall maximum epoch limit for the training loop.

* **Inference and stochasticity:** For the final generation of synthetic domains, the paper mentions selecting the character with the highest probability, but does not detail whether temperature sampling is used or how the output TLD is managed.

---

#### Assumptions

* **GCNN filters and dense architecture:** The use of 128 filters in the GCNN layers and 128 neurons (with ReLU activation) in the intermediate hidden layers is assumed. For the outputs corresponding to the latent mean and logarithmic variance ($\mu$ and $\sigma$), a linear activation is assumed. In the decoder, an initial projection through a dense layer followed by a Reshape operation to $30 \times 128$ is assumed. If dimensions do not match in the residual connection, a 1x1 1D convolution is applied.

* **Vocabulary and padding:** A mandatory conversion to lowercase is assumed, extracting only the Second-Level Domain (SLD). To reach the fixed length of 30 characters, post-padding is assumed (padding at the end of the sequence using index 38).

* **Loss function and regularization:** The use of Categorical Cross-Entropy summed across the time axis (30 steps) is assumed so that it competes in a balanced manner with the KL Divergence. No additional Dropout is assumed, relying instead on the regularization effect of the implemented GAP layer.

* **Training, epochs, and early stopping:** Given the ambiguity of the hyperparameters described in the paper, a learning rate of 0.0005, a batch size of 512 for the VAE, and a maximum of 100 epochs are assumed. Additionally, instead of accuracy, the use of `val_loss` as the Early Stopping metric is assumed, reserving 10% of the batch for validation.

* **Sampling (inference) and TLDs:** It is assumed that all variability comes from stochastic sampling of the latent space $z \sim \mathcal{N}(0, I)$. Decoding is performed using a deterministic `argmax` over the output Softmax probabilities. Finally, the static concatenation of the suffix `.com` to all generated domains is assumed.

* **Training-set deduplication:** Following Sec. 5.4 of the paper, SLDs that become duplicates once the TLD is removed are eliminated before training (the paper reports 926,816 distinct names out of the Alexa 1M).

* **Output validation:** The paper's post-processing truncates the padding characters and appends a TLD to obtain "a final full domain name". We read that as requiring a valid DNS label: after padding truncation the SLD must match RFC 1034/1035 (1 to 63 characters, `a-z0-9-`, no leading or trailing hyphen, no `.`). Outputs that fail the check (for example the empty string) are rejected and a new `z` is sampled, with a bounded number of attempts and a logged fallback, so generation cannot hang or silently yield empty domains. Decoding itself stays deterministic `argmax`.

## Liu, Zhang, et al., 2021

### CLETer

#### Information Gaps

* **Global Training Parameters:** The paper provides the data split and epoch limit, but omits the fundamental optimization configuration. It does not mention the loss function, optimizer, learning rate, or batch size used to train the target model.

* **CNN Model Architecture:** Although a one-dimensional CNN network is described, the dimension of the initial Embedding layer, the activation functions applied after the convolution operations, and the exact architecture of the post-aggregation dense layers are omitted.

* **Evasion Algorithm (Ties and Weights):** The CIS function uses a self-defined parameter *lambda* to adjust the weight between HIS and TIS, but it does not detail the mathematical tie-breaking mechanism in cases where multiple characters obtain the same importance score. Nor does it precisely define the exhaustive search within the set of valid characters.

* **Iteration Limits and TLD Management:** A strict maximum limit on how many substitutions are allowed per domain before aborting the evasion attempt is not specified, nor does the paper indicate whether the same TLD is reattached to the final modified domains.

---

#### Assumptions

* **For the target model (CNN):** The use of a default CNN model with the following layers and parameters is assumed: `Embedding(128) → Conv1D(128, k=5) → GlobalMaxPooling → Dense(64) → Dropout(0.5) → Dense(1, sigmoid)`. To compile and train this network, the standard configuration for binary classification is assumed: Adam optimizer with a learning rate of 0.001, Binary Cross-Entropy (BCE) loss, and a batch size of 64.

* **For the influence calculation function:** The CIS (*Combined Influence Score*) function is used by default. The value of the *lambda* variable is set and maintained strictly at 1.0 for all calculations.

* **For the tie-breaking criterion:** In cases where the influence scores of multiple characters are identical, the original positional order of the characters (from left to right in the string) is assumed to determine the evaluation order.

* **For the substitution search:** At each evaluated position, the algorithm assumes a complete search that iterates over all 37 valid supported characters (lowercase letters, digits, and hyphen) in order to find the one that yields the lowest malicious classification probability.

* **For the modification limit:** A maximum limit of 5 character substitutions per domain (`max_substitutions = 5`) is assumed in order to balance evasion effectiveness without completely destroying the original semantic structure of the domain.

* **For TLD handling:** It is assumed that, after completing the character modification process on the domain name (SLD), the original TLD or SLD is concatenated back to the resulting string to ensure that the domain remains valid within the framework context.

* **Oracle input:** The influence scores and the substitution checks query the target classifier with the SLD alone, as in the paper, whose data preprocessing removes the TLD and whose target models are trained on the SLD. The original TLD is appended only to the returned domain. Against the framework's full-domain detectors this makes the full-domain evaluation a transfer setting; the SLD-only analysis mode is the paper-comparable measurement (see `docs/analysis.md`).

* **Oracle batching:** The paper describes one classifier query per prefix, suffix and candidate substitution. Those queries are independent within each step of Algorithm 1, so the implementation sends them to the classifier in batches (one call for all the prefixes and suffixes of the CIS scores, one call for the 36 candidates of each substitution round) and keeps the first strict minimum, which reproduces the sequential search and its tie-breaking exactly. This only changes the cost (about 6 classifier calls per domain instead of about 300), not the evaluation order or the result.

* **Seeds:** The paper does not fix how many AGDs are modified (its experiments select 5,000 at random). The substitution search is deterministic for a given seed and classifier, so the implementation uses the whole D1 malicious split as seeds (SLDs longer than 3 characters), one seed per generated domain, cycled in a seeded shuffled order.

## Liu Yu, et al., 2021

### GeometricPerturbationDGA

#### Information Gaps

* **Exact network topology and architecture:** The paper states that the Adversarial Transformation Network (ATN) uses a SeqGAN to generate legitimate domain samples and that the target network ($D$) is based on a CNN architecture. However, it omits the network depth, embedding dimensions, the number of hidden units in the recurrent layers, convolutional filter sizes, and the configuration of the dense layers.

* **Training hyperparameters:** The paper presents a joint loss function coupled by a coefficient $\beta$, but does not provide its numerical value. It also does not specify the learning rate ($\alpha$), the optimization algorithm, the batch size, or the number of epochs required for pre-training and adversarial training.

* **Text differentiability and modulo operator computation:** Operating on discrete characters using the modulo operator $\vert{}2X' - M + Z\vert{} \pmod{\vert{}V\vert{}}$ within a computational graph breaks gradient differentiability. The paper does not explain the technical mechanism used to connect the generator's continuous outputs with the discrete inputs required by the target network while preserving the backpropagation flow.

* **Representation space, vocabulary, and TLD management:** The paper does not formalize the limits of the environment, such as the maximum domain length ($n$), the exact vocabulary composition ($\vert{}V\vert{}$), or how the hierarchical domain structure (distinction between SLD and TLD) is handled to avoid altering top-level domain extensions.

* **Pre-training strategy and initialization:** The paper mentions the use of Maximum Likelihood Estimation (MLE) to pre-train the generator, but does not specify whether the target network $D$ requires independent pre-training beforehand, nor the parameter freezing policy applied during generator optimization.

---

#### Assumptions

* **Network architecture ($G$ and $D$):**
  * **Generator $G$ (ATN):** Implemented with a 64-dimensional Embedding layer (for a vocabulary of 38 symbols), followed by an LSTM layer with 128 recurrent units returning full sequences, and a final Dense layer with softmax activation over the vocabulary size.
  * **Target Network $D$ (CNN Discriminator):** Defined with a 64-dimensional Embedding layer, a Conv1D layer with 128 filters and a kernel size of 5 using ReLU activation, followed by Global Max Pooling, an intermediate Dense layer with 64 units and ReLU activation, and a final Dense output layer with 1 unit and sigmoid activation for binary classification.

* **Differentiability and the modulo operator (Straight-Through Estimator):** To preserve differentiability during the computation of the perturbation $M' = \vert{}X' + X - M\vert{} \pmod{\vert{}V\vert{}}$, the generator produces a continuous projected representation $X'$ by computing the expected value of the softmax probabilities. The floating-point modulo operation is then applied to this continuous vector. Network $D$ takes one-hot rows through a bias-free Dense layer, which is the same computation as an Embedding lookup but keeps its input differentiable. A Straight-Through Estimator (STE) converts $M'$ for $D$: the forward pass uses the one-hot of the rounded index, clipped to $[0, \vert{}V\vert{}-1]$, while the backward pass flows through a softmax over the squared distance between the continuous index and every symbol.

* **Padding and length of $X'$:** Eq. 18 defines the encoded legal domain with $x_j \in [1, \vert{}V\vert{}]$ for $j \le \vert{}X\vert{}$ and $x_j = 0$ beyond it. The adversarial sample $X'$ is taken to follow the same structure: the generator's output at the padded positions of $X$ is forced to the padding symbol, both in training and in generation. Without that constraint $G$ can fill the tail of $X'$ with a symbol that $D$ scores as benign (for example a run of hyphens up to the 63-character limit), which through Eq. 15 yields meaningless labels. Since the modulo of Eq. 15 can still put a hyphen at either end of a label, which RFC 1034/1035 forbid, a pair is redrawn a bounded number of times until the label is valid.

* **Loss functions:** $L_G$ follows Eq. 12, the target network's evaluation of the adversarial legal sample $X'$ (mean of $D(X')$ over the batch, with the hard one-hot of $X'$ in the forward pass and the softmax in the backward pass). Minimising it pushes $X'$ to the malicious side of the boundary, as Eq. 10 requires, so that $Z = X - X'$ points across it. $L_D$ follows Eq. 13 as the binary cross-entropy between $D(M')$ and the benign label.

* **Training hyperparameters and procedure:**
  * $\beta = 0.5$ is used in the joint loss function $L = \beta L_G + L_D$, giving equal importance to deceiving the target network and enforcing the geometric constraint.
  * The Adam optimizer is used with a learning rate of $\alpha = 0.001$ and a batch size of 64.
  * 20 epochs of MLE pre-training are performed for the generator $G$ (predicting the next character), followed by 10 epochs of binary pre-training for the target network $D$.
  * Joint adversarial training of the generator is performed for 50 epochs while freezing the target network weights to prevent it from unlearning during the optimization of $G$.

* **Representation, vocabulary, and TLD management:** A maximum label length of $n = 63$ (the standard DNS protocol limit) and a vocabulary of $\vert{}V\vert{} = 38$ symbols are assumed (26 lowercase English letters, 10 digits, the hyphen `-`, and the token `0` reserved for padding). All encodings and geometric perturbations are applied strictly to the SLD (Second-Level Domain). During the generation phase, the original TLD from the DGA sample is preserved and concatenated back to the end of the perturbed string.

* **Training datasets:** The D1 slice of the Tranco dataset is used for benign domains, while the DGArchive `*.csv` files are used for DGA samples.

## Zhai et al., 2022

### CDGA

#### Information Gaps

* **NLP parameters:** The paper states the use of SentencePiece and Word2Vec but omits the vocabulary size, the underlying tokenisation model (BPE, Unigram, etc.), and the exact embedding vector dimension ($d_m$).

* **Autoencoder (AE) architecture:** The authors indicate that both the encoder and decoder are based on LSTM layers, denoting the number of layers as $L$, but they do not specify either the value of $L$ or the hidden state size of the recurrent networks.

* **WGAN-GP network details:** Although the use of a ResNet architecture for the discriminator and Gaussian noise for the generator is described, the paper does not specify the dimension of the initial noise vector, the number of residual blocks, the activation function, the gradient penalty coefficient ($\lambda$), or the number of critic iterations.

* **Training hyperparameters:** The article does not specify critical elements such as the batch size, the specific optimisers used for the GAN networks, the learning rates, or the total number of training epochs required for convergence.

* **Sequence and TLD handling:** The maximum text sequence length, the padding strategy used to equalise domain lengths, and whether the final model generates only a Second-Level Domain (SLD) or includes the Top-Level Domain (TLD) are not specified.

---

#### Assumptions

* **NLP preprocessing:** A vocabulary size of 5,000 is assumed using the Unigram model in SentencePiece. For Word2Vec, the continuous skip-gram algorithm is used with an embedding dimension of 128, a context window of 5, and a minimum frequency of 1.

* **Autoencoder architecture:** A design with 2 LSTM layers and 256 hidden units is assumed. The Adam optimiser is used with a learning rate of 0.001. The context vector is computed as the global average of the encoder hidden states using GlobalAveragePooling1D and is used to initialise the states of both LSTM layers in the decoder.

* **WGAN-GP network:** A latent noise vector ($Z$) dimension of 100 is assumed. The generator consists of a single Dense layer with 256 units and a linear activation function. The critic (discriminator) uses 3 one-dimensional residual blocks with a kernel size of 3 and a ReLU activation function. A gradient penalty coefficient of $\lambda = 10.0$ is assumed, and the critic is updated 5 times for every generator update. Both components use the Adam optimiser with a learning rate of 0.0001, $\beta_1 = 0.0$, and $\beta_2 = 0.9$.

* **Training hyperparameters (code-based):** A batch size of 64 is assumed. The Autoencoder is assumed to be trained for 50 epochs, while the WGAN is trained for 200 epochs.

* **Domain generation (padding, TLDs, and generation):** It is assumed that the model is trained to generate SLD labels exclusively, with TLDs removed from the Tranco dataset during extraction. Domain lengths are constrained to a minimum of 4 and a maximum of 63 characters or tokens. Shorter sequences are right-padded using a padding token corresponding to ID 0. During generation (Algorithm 1), the probabilities of invalid tokens (PAD, UNK, BOS) are forcibly masked to 0 to ensure compliance with domain registration constraints. If the algorithm fails to generate a valid domain (i.e., not starting with a hyphen and containing only alphanumeric characters) or produces duplicate domains after 100 attempts using different noise vectors, the model assumes a controlled failure, logs a warning and returns a random lower-case label.

* **Reproducibility:** Since the objective of the framework is to guarantee the reproducibility of the experiments, a fixed seed is used in the `main` function instead of a time-based seed, as the latter is an operational feature intended for synchronisation in real-world environments rather than academic evaluation.

* **TLD (framework convention):** The paper defines the generator on the second-level label only. Because the detectors are trained on full domains and the lexical metrics change with the TLD, the framework appends one to every generated SLD, drawn from the empirical public-suffix distribution of the D1 Tranco slice (top 50, `core/tld.py`). The same rule applies to every model in this situation so that the TLD is not a confounder between generators.

## Shu et al., 2022

### GWDGA

#### Information Gaps

* **Training Parameters:** The paper omits fundamental hyperparameters such as the optimizer used, the learning rate, the batch size, the exact number of training epochs for the 905,436 benign domains, and whether a validation split was employed.

* **Loss Function and Posterior Collapse:** Although the maximisation of the ELBO (composed of the KL divergence and reconstruction error) is mentioned, the exact reconstruction metric is not specified. The paper also does not address how to mitigate posterior collapse ("KL Vanishing"), a critical and common issue when applying VAEs to text sequence processing.

* **Architectural Details (GCNN and Embedding):** Critical technical specifications are missing, such as the padding strategy used in the 1D convolutions and the exact number of filters required. Furthermore, the paper does not explain how the weights of the Embedding layer are initialised for such a specialised statistical dictionary that combines both n-grams and whole words.

* **Generation and Latent Space:** The paper presents the mathematical reparameterisation trick but does not clarify whether the network predicts the variance ($\sigma^2$) or the logarithm of the variance ($\log \sigma^2$). During decoding, it is also unclear whether stochastic sampling or a purely deterministic approach is applied over the Softmax layer.

* **Segmentation and External Data:** No tie-breaking criterion is defined for the BiMM algorithm when the forward and backward passes produce identical results in terms of length and unigram count. Likewise, technical information is missing regarding how the external LDS frequent-word list is loaded and integrated in practice.

---

#### Assumptions

* **Training (hyperparameters):** The implementation assumes the use of the Adam optimiser with an initial learning rate of 0.001. A batch size of 128 is configured, and the model is trained for 50 epochs. To prevent overfitting and improve learning, 20% of the data is reserved for validation, and the Early Stopping (patience of 5) and ReduceLROnPlateau callbacks are employed.

* **Loss Function and VAE:** Since the tokens are not one-hot encoded, the reconstruction loss is computed using Sparse Categorical Cross-Entropy. To prevent the theoretically expected posterior collapse, the implementation assumes the use of KL Annealing, linearly increasing the weight of the $\beta$ parameter from 0.0 to 1.0 during the first 10 epochs.

* **Architecture (GCNN and Latent Space):** The 1D convolutional layers assume `padding="same"` and 128 filters to match the embedding dimension, with the sequence subsequently collapsed using a GlobalAveragePooling1D layer. In the encoder, for numerical stability, the variance layer is assumed to predict $\log \sigma^2$. In addition, Glorot Normal initialisation is used for dense and convolutional layers, while Random Uniform initialisation is used for the Embedding layer.

* **Dictionary and Tokeniser (BiMM):** Index 0 of the dictionary is reserved exclusively for the padding token, which justifies the final vocabulary size of 7,487 entries. For the BiMM segmentation algorithm, it is assumed that, in the event of a tie between both directions, the Forward Maximum Matching (FMM) result is prioritised by convention.

* **Domain Generation:** During the inference stage, a purely deterministic Greedy Search scheme based on `argmax` is assumed, discarding temperature-based sampling. During decoding, padding tokens are omitted. Finally, strict structural filters are applied, assuming that only generated domains with lengths between 5 and 63 characters that do not contain consecutive hyphens (`--`) or hyphens at either end are considered valid.

* **External Data (LDS):** It is assumed that the most frequent words are read from the local file `dataset/fixes.md` (obtained from https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103), extracting only the lines prefixed with `*` and removing any hyphens from the tokens. If the file does not exist, the implementation assumes a fallback mechanism that extracts the most common words directly from the training domains.

* **Output:** The implementation returns SLD labels rather than complete domain names.

* **TLD (framework convention):** The paper defines the generator on the second-level label only. Because the detectors are trained on full domains and the lexical metrics change with the TLD, the framework appends one to every generated SLD, drawn from the empirical public-suffix distribution of the D1 Tranco slice (top 50, `core/tld.py`). The same rule applies to every model in this situation so that the TLD is not a confounder between generators.

## Zhang et al., 2022

### WGAN-based DGA

#### Information Gaps

* **Learning rate:** The paper explicitly states that the optimizer used for both the generator and discriminator networks is RMSProp, but it omits the specific learning rate required for convergence.

* **Lipschitz constraint (Weight Clipping):** The core WGAN algorithm requires clipping the discriminator weights in order to compute the Wasserstein distance. However, the paper does not specify the weight clipping range used.

* **Discriminator update frequency:** In the canonical WGAN architecture, the discriminator is trained multiple times for each generator update. The paper does not specify this training ratio.

* **Batch size:** The paper mentions a configuration of 10,000 training iterations and batches of 100 generated domains for performance evaluation, but it does not explicitly state the batch size used to compute the gradients during each training step.

* **Input noise distribution and weight initialization:** The paper specifies a 100-dimensional input vector of "random noise" without defining its probability distribution (e.g., uniform or normal). It also does not document the method used to initialize the network weights before training.

---

#### Assumptions

* **Learning rate:** A learning rate of 0.00005 has been assumed for the RMSProp optimizer.

* **Weight clipping:** A clipping range of [-0.01, 0.01] is assumed and applied exclusively to the discriminator layers.

* **Update frequency:** It is assumed that the discriminator is updated five times for every generator update.

* **Batch size:** A batch size of 100 samples is assumed during network training.

* **Input noise distribution:** A standard normal (Gaussian) distribution is used to initialize the latent noise vector.

* **Weight initialization:** **He Normal** initialization is applied to hidden layers using ReLU activation functions, while **Glorot Uniform** initialization is used for the output layers.

* **Domain generation and TLD (framework convention):** The paper's generator only models the hostname label because the TLDs of each DGA family are fixed. The framework appends a TLD drawn from the empirical public-suffix distribution of the D1 Tranco slice (top 50, `core/tld.py`), the same rule used for every model whose paper generates only the SLD, so that the TLD is not a confounder between generators.

* **Case folding and output validation:** Table 1 of the paper encodes 63 characters, including `A-Z` (codes 37-62). DNS names are case-insensitive and the training hostnames are lowercase, so those codes never occur in the data and only appear when the generator is poorly trained. The decoded text is folded to lowercase (a lossless post-processing step) and every output must be a valid RFC 1034/1035 label (1 to 63 characters, `a-z0-9-`, no leading or trailing hyphen); invalid outputs are rejected and resampled, with a bounded number of attempts and a logged fallback.

* **Training data:** The generator is trained on the real AGD host names of the D1 malicious split, drawn through the shared `core.data_splits.load_malicious_seeds`. The paper does not state the size of its training set beyond the 20,000 real DGA samples used in the evaluation.

## Hu et al., 2023

### ReplaceDGA

#### Information Gaps

* **Training and optimization hyperparameters:** The paper specifies that the Adam optimizer is used, but omits crucial details such as the learning rate, weight decay, and the maximum number of training epochs.

* **Weight initialization:** Although the paper states that all deep learning models are implemented using PyTorch, it does not mention the weight initialization method used for the BiLSTM network and the Embedding layer.

* **Handling of unsupported characters:** The paper indicates that, after extracting the SLDs from the benign domains and counting the characters, there are thirty-eight valid characters in the benign domain list. However, it does not specify whether domains containing characters outside this set (e.g., Unicode or other special characters) were simply removed from the original dataset.

* **Specific TLD list:** The system selects a Top-Level Domain (TLD) from a fixed list to combine with the generated domain. However, the paper does not specify which exact TLDs are included in that list, stating only that it contains dozens of common TLDs.

---

#### Assumptions

* **Training hyperparameters:** Since the paper does not specify the training hyperparameters, standard default values for Adam in sequence processing are assumed to reduce the risk of overfitting. A learning rate of 0.001 and a maximum of 50 training epochs are used. In addition, an Early Stopping criterion with a patience of 5 epochs based on the validation loss is assumed.

* **Weight initialization:** Since no custom initialization method (such as Xavier or He initialization) is explicitly mentioned, the default PyTorch weight initialization is assumed.

* **Unsupported character filtering:** It is assumed that input domains containing any character outside the set of the 38 valid characters are discarded from the training dataset.

* **TLDs:** A static list of 22 common TLDs (such as `.com`, `.net`, `.org`, `.uk`, etc.) is assumed in the implementation. During domain generation, one TLD is selected randomly and appended to the generated SLD.

* **Data partitioning:** It is assumed that the dataset is split by using 100,000 clean benign domains for training and another 100,000 different benign domains for constructing the SLD repository used during generation. If fewer domains remain after filtering, the implementation dynamically assumes an 80%/20% split between the training set and the SLD generation set.

* **Domain length:** It is assumed that the operational domain length is restricted to between 4 and 60 characters, inclusive. If a domain in the dataset exceeds the maximum length of 60 characters, the sequence is truncated.

* **Randomness control:** The random seed is assumed to be the one globally defined in the `main` file.

* **Generation failures:** A maximum of 10 generation attempts is assumed. If the generated domain does not satisfy the required validity constraints (for example, if it starts or ends with a hyphen), the implementation logs a warning and returns a random lower-case label with a randomly selected TLD.

## Nie et al., 2023

### PKDGA

#### Information Gaps

* **Training parameters and scale:** The paper describes the reinforcement learning training process (Policy Gradient with Monte Carlo searches) and the use of LSTM networks, but omits the exact duration of the maximum likelihood estimation (MLE) pretraining phase.

---

#### Assumptions

* **Architecture and training cycle:** A policy network based on a single-layer LSTM with an embedding dimension of 32, a hidden dimension of 64, and a vocabulary of 39 characters is assumed. The process is divided into 1 epoch of MLE pretraining using 10,000 benign input domains, followed by 60 batches of adversarial RL training using 20 Monte Carlo searches per sample. These values have been taken from Table III of the paper and the source code provided at https://github.com/abcdefdf/PKDGA.

* **Target detector (Reward Oracle):** If no target detector is explicitly provided when constructing the model, the default assumption is the instantiation and initialization of a CNN-based detector, which returns a reward of 1 if the generated SLD is classified as benign (evasion) and 0 otherwise.

* **Sequence length and vocabulary:** A fixed maximum sequence length of 65 characters and a predefined alphabet of 39 tokens (alphanumeric characters, hyphen, underscore, and space) are assumed. The space token `' '` (index 0) is used as the right-padding element for shorter sequences.

* **Domain scope and TLDs:** The model generates the SLD; following the paper ("a valid domain name can be constructed by ... appending a top-level domain at the end"), a TLD is appended. The paper does not say which one, so the framework convention is used: a TLD drawn from the empirical public-suffix distribution of the D1 Tranco slice (top 50, `core/tld.py`). Optional third-level labels are not generated.

* **Output validation:** The 39-token alphabet of the authors' code includes `_`, which is not a valid hostname character, and the policy can emit sequences that are empty after removing padding. Every generated SLD is checked as an RFC 1034/1035 label (1 to 63 characters, `a-z0-9-`, no leading or trailing hyphen); invalid outputs are rejected and resampled, with a bounded number of attempts and a logged fallback. The alphabet itself is kept unchanged so that `vocab_size` matches the original implementation.

* **String processing:** After sampling from the generator, padding tokens (0) are discarded, leading and trailing hyphens are removed from the resulting string, and an empty or otherwise invalid string is rejected and resampled. During the Monte Carlo rollouts an empty sample is replaced by a random lower-case label so that the oracle always receives a label.

* **Training data:** The benign Tranco list (D1) is used, filtering only those SLDs whose characters belong entirely to the alphabet supported by the model.

* **Oracle input:** The reinforcement rewards (`compute_rewards` and the Monte Carlo rollout) query the target detector with the generated SLD alone, as in the paper, where the generator produces the second-level label and the TLD is appended afterwards. Against the framework's full-domain detectors this makes the full-domain evaluation a transfer setting; the SLD-only analysis mode is the paper-comparable measurement (see `docs/analysis.md`).

## Nazzal et al., 2024

### MintA

We decided not to include MintA in the framework because its detector evasion mechanism differs from that of the other models and from the purpose of the standardized evaluation developed.

The architecture of this framework is designed for the homogeneous evaluation of Domain Generation Algorithms (DGAs), focusing on measuring the syntactic and algorithmic capabilities of different generators to produce adversarial domains and comparing their evasion rates against classifiers.

MintA is not a DGA, but rather an attack technique focused on Graph Neural Networks (GNNs), which are used in Malicious Domain Detection (MDD) systems. It focuses on optimizing the behavior of malicious domains within the network by perturbating their topological relationships and lexical attributes so that the detector observes a distorted graph, causing it to misclassify the domains.

Integrating MintA would undermine the functional purpose of the framework. MintA is not a DGA, but rather an adversarial attack framework operating at inference time on deep learning graph topologies. Its inclusion would require implementing an evaluation specifically designed for this model, which is contrary to the objective of performing homogeneous and comparative evaluations of DGA algorithms.

## Selvaraj and Panjanathan, 2024

### WordDGA

#### Information Gaps

* **RCNN-BiLSTM architecture:** The paper mentions the use of convolutional filters of size ($h \times d$) in the RCNN layer, but does not specify the exact window size (h) or the number of filters. It also does not detail the number of units or the hidden state size for the BiLSTM layer.

* **Sequence length and embeddings:** There is a contradiction regarding the input sequence length; the text indicates 64 characters based on the RFC standard, while the pretraining section fixes the length at 70. Furthermore, it is not explained how the final concatenation between ELMo (1024 dimensions) and FastText CBOW (100 dimensions) is performed.

* **cWGAN architecture and parameters:** The hidden layer dimensions for the Generator (3 layers) and the Discriminator/Critic (5 layers) are not specified. In addition, the text states that ReLU is used in the Discriminator, whereas Table 6 of the same paper requires LeakyReLU. The value of the coefficient ($\lambda$) for the WGAN penalty term is also omitted.

* **Statistical features and fusion:** The paper mentions reducing 11 statistical features to 8 using a fully connected layer, but does not specify the activation function. Likewise, the technical details and the exact connection point where all branches converge into the final fusion layer before the classifier are missing.

* **Gumbel Softmax and Adam hyperparameters:** The numerical values for the initial temperature and its possible annealing schedule in the Gumbel Softmax are completely omitted. Similarly, the beta hyperparameters of the Adam optimizer, the discriminator-to-generator update ratio, and the exact weight assigned to the classification loss are not specified.

---

#### Assumptions

* **Domain length:** Prioritizing the strict RFC standard, the domain sequence length is assumed and limited to 64 characters/tokens by default.

* **Feature extraction (RCNN-BiLSTM):** One RCNN-BiLSTM branch, as in Section 4.2.1: a Conv1D layer with 128 filters and a window of h = 3 characters (the $h \times d$ filter of Eqs. 5 to 7), followed by a BiLSTM with 64 units per direction (128 in total, the 128-dimensional sequence feature of the paper). FastText embeddings projected to 100 dimensions are used as input to this branch. The 3 to 7 grams of the paper are the n-gram vocabulary of the input embedding and of the reputation feature, not separate networks.

* **cWGAN architecture:** Based on the standard Wasserstein GAN literature:
  * **Generator:** Three progressive hidden dense layers of 256, 512, and 1024 neurons (the "three hidden layers" of Section 4.1), using LeakyReLU with an $\alpha$ of 0.2, projected to the sequence of character logits.
  * **Critic:** The reverse process is applied after concatenation, with hidden layers of 1024, 512, 256, 128, and 64 neurons using LeakyReLU ($\alpha$ = 0.2).

  The paper does not state the critic-to-generator update ratio. One critic update per generator update (1:1) is used. The WGAN-GP default of 5:1 multiplies the cost of every step by the ratio, and Section 4.1 reports that the authors trained the whole cWGAN in about three hours on an 8-core Core i7 with 32 GB, which rules out the heavier readings.

* **Penalties and optimizers:** An Adam optimizer is assumed for both networks with a learning rate of 0.0002, ($\beta_1$ = 0.5), and ($\beta_2$ = 0.9) to avoid training collapse. The WGAN gradient penalty coefficient is fixed at ($\lambda$ = 10.0). A weight of 1.0 is assumed for the conditional loss.

* **Gumbel Softmax:** An initial temperature parameter of 1.0 is assumed, with exponential annealing applied throughout training until reaching a minimum value of 0.1.

* **Statistical features and TLDs:** A ReLU activation is assumed to reduce the 11 statistical variables to 8. Both training and generation assume a pure Second-Level Domain (SLD) scenario; therefore, TLDs and subdomains are removed from both the Tranco dataset and the malware families during preprocessing.

* **Soft approximation for ELMo and lexical heuristics:**
  * *Simplification:* ELMo requires the evaluation of string-based tensors. Instead of using native libraries that would break the computational graph during differentiable generation with Gumbel Softmax, the implementation assumes a simulated projection by passing the sequential tensor through a 1024-dimensional Dense layer, applying GlobalAveragePooling1D, and projecting it to 128 dimensions. Likewise, the 11-dimensional statistical variables computed over the generated samples during the `_train_step` do not rely on the Python `wordninja` library, but are instead approximated mathematically by summing probabilities (e.g., direct statistical counting of the probability of the "vowel" and "hyphen" tokens).
  * *Justification:* This approximation is used to make the model fully differentiable and end-to-end trainable. By approximating these metrics directly from the raw generated probabilities, the gradient can flow uninterrupted for the computation of the WGAN-GP penalty.

* **Epochs:** Although the paper specifies 5000 epochs, only 100 are used for simplicity. The Gumbel-Softmax temperature is annealed over the epochs.

* **Base domain per output:** Algorithm 1 iterates over the DGA domains x(i) and produces one z(i) from each. The implementation therefore cycles over the known DGA SLDs in a seeded shuffled order, one base per generated domain, and applies the word substitution R(.) to it.

* **Oracle batching:** The evasion loop of Algorithm 1 tries up to 50 candidate words per domain. The implementation draws the 50 candidates from the generator in one call and scores the valid ones with one classifier call, in attempt order, returning the first that evades. This only changes the cost (one classifier call per domain instead of up to 50), not the outcome.

* **Domain generation:** When generating domains using Algorithm 1, the maximum number of attempts is limited to 50.

* **Label length of the composed domain:** Algorithm 1 does not bound the length of z(i) = R(W, s(x(i))), and R(.) is only named as a "word rule function". The paper fixes the sequence length to 64 precisely because the RFC caps a DNS label at 63 characters (Section 3.1.1), so the same limit is applied to the composed SLD: a candidate longer than 63 characters is discarded and the next of the 50 attempts is tried, as with a collision. Without this check, W can be up to 63 characters and the remaining segments of the base are kept, so the result can exceed the limit.

* **Detector:** To evaluate the generated domains as described in Algorithm 1, a CNN-based detector is used by default.

* **TLD (framework convention):** The paper defines the generator on the second-level label only. Because the detectors are trained on full domains and the lexical metrics change with the TLD, the framework appends one to every generated SLD, drawn from the empirical public-suffix distribution of the D1 Tranco slice (top 50, `core/tld.py`). The same rule applies to every model in this situation so that the TLD is not a confounder between generators.

* **Oracle input:** The evasion check of Algorithm 1 queries the target classifier with the SLD alone, as in the paper ("the top-level domain is removed from the domain name during data preprocessing"). The TLD is appended only to the returned domain. Against the framework's full-domain detectors this makes the full-domain evaluation a transfer setting; the SLD-only analysis mode is the paper-comparable measurement (see `docs/analysis.md`).

## Nangong and Wu, 2025

### TLVDGA

#### Information Gaps

* **Vocabulary and sequences:** The authors construct a dictionary with the 5000 most frequent n-grams using NLTK. However, they do not detail how they handle variable domain lengths or whether they use special control tokens such as padding or start and end of sequence tokens.
* **Embedding dimensionality:** The use of the Word2Vec Skip-gram model to map the n-grams into a low-dimensional continuous vector space is indicated, but the exact size of this dimension is not specified.
* **Transformer Encoder architecture:** The document mentions the use of a Transformer with multi-head self-attention and global pooling. However, it omits the number of Transformer blocks, the number of attention heads, the internal feed-forward dimension, and the technical formulation of positional encoding.
* **Latent space:** Mean and variance vectors are extracted to apply the reparameterization trick, but the dimensionality of the resulting latent vector is unknown.
* **LSTM Decoder architecture:** A layer architecture is listed that includes an LSTM network connected to a Softmax activation, but the number of recurrent layers and the hidden-state dimension are missing.
* **Training hyperparameters and loss function:** For the KL annealing strategy, the authors define the decay formula with a control constant of 0.05 (k), but they do not provide the value of the temporal midpoint. They also do not detail the exact reconstruction loss function, optimizer, learning rate, batch size, or total number of epochs.
* **Tie-breaking in BiMM:** The study indicates that the data are split using forward and backward matching. The paper does not clarify how ties are resolved in the event that both directions return the same number of matching tokens for the same domain.
* **Internal validation:** The authors detail that the training dataset consists of 500,000 legitimate domains. They do not detail the partitioning of an internal validation set to monitor the loss and prevent overfitting.

---

#### Assumptions

* **For sequences and tokens:** A maximum sequence length of 20 tokens is assumed. In addition, the vocabulary has been expanded to 5004 tokens to explicitly include special tokens for padding, unknown tokens, and start and end of sequence tokens.
* **For dimensionality:** The Word2Vec embedding dimension is assumed to be fixed at 128, and the latent vector is mapped to a dimension of 64.
* **For the Transformer Encoder architecture:** An implementation of 2 Transformer layers is assumed, configured with 4 attention heads and a feed-forward hidden dimension of 512. Since the Transformer lacks a notion of order, the inclusion of a positional encoding layer added to the inputs is assumed. Finally, the use of Average Pooling at the end of the encoder is assumed.
* **For the LSTM Decoder architecture:** The use of 1 unidirectional LSTM layer with 256 hidden units is assumed.
* **For the training hyperparameters:** The use of the Adam optimizer with a learning rate of 0.001 and a batch size of 128 is assumed. A total training period of 100 epochs is assumed, setting the midpoint of the KL annealing at epoch 50 (halfway through).
* **For the loss function:** When predicting over dictionary indices, the use of Sparse Categorical Cross-Entropy is assumed for calculating the reconstruction loss.
* **For tie-breaking in BiMM:** It is assumed that, if the matchings differ, the segmentation that produces the smallest total number of tokens is selected. If the number of tokens is identical between both options, the forward matching (FMM) is selected by default.
* **For training and validation data:** For training, all of the benign domains extracted from the Tranco list (D1) are used. Likewise, an internal 80/20 split over the encoded data is assumed to separate the training and validation subsets.

* **TLD (framework convention):** The paper defines the generator on the second-level label only. Because the detectors are trained on full domains and the lexical metrics change with the TLD, the framework appends one to every generated SLD, drawn from the empirical public-suffix distribution of the D1 Tranco slice (top 50, `core/tld.py`). The same rule applies to every model in this situation so that the TLD is not a confounder between generators.

## Pregardier et al., 2025

### TITAN DGA

#### Information Gaps

* **Positional Encoding and sequence length:** The text defines the self-attention layers, but does not mention the use of Positional Encoding (mandatory in Transformers) or the maximum sequence length of the tokenized domains.
* **GAN dimensional mapping:** The authors detail that the Generator and Discriminator are single-layer MLPs with 128 neurons, which is mathematically incompatible with the latent sequence required by the decoder (a sequence of 512-dimensional vectors). The negative slope (alpha) for the LeakyReLU function is also not specified.
* **Training and regularization hyperparameters:** Key parameters are omitted, such as the total number of epochs for the initial phase and retraining, learning rates for the different modules, the *batch size*, discriminator loss scaling, Gaussian noise variance, and the numerical value of the weighting coefficient for the KL divergence.
* **Tokenizer Configuration and Loss Function:** The vocabulary size required by SentencePiece and the exact metric used to measure token reconstruction error (reconstruction loss) are not reported.
* **Decoding and optimization strategy:** The token feeding dynamics (e.g., Teacher Forcing) and network weight initialization are omitted.

---

#### Assumptions

* **For encoding and length (Autoencoder):** Standard sinusoidal positional encoding has been implemented and added to the embeddings. In addition, a maximum sequence length of **24 tokens** has been fixed. No minimum length is assumed for the generated domains.
* **For the GAN architecture:** It is assumed that the 128-neuron layer acts as a hidden layer. A linear output layer has been added that projects these neurons to the exact dimensionality of the latent space ($24 \times 512$), reshaping it into matrix form. A value of $\alpha = 0.2$ is used for LeakyReLU.
* **For hyperparameters and epochs:** An initial training cycle of 100 epochs and retraining cycles of 20 epochs are assumed, with a *batch size* of 256. Gaussian noise with mean 0 and standard deviation 0.1 is applied ($\epsilon \sim \mathcal{N}(0, 0.1)$). The KL coefficient starts at 0.1 and is annealed up to 1.0 to stabilize training. In addition, the discriminator loss is reduced by half (`disc_loss_scale = 0.5`) to balance the GAN.
* **For Tokenizer and Loss configuration:** A vocabulary size of **2000 tokens** using SentencePiece Unigram is assumed. Indices 0 to 3 are reserved for special tokens (UNK, BOS, EOS, PAD). For the reconstruction error metric, standard *Sparse Categorical Cross-Entropy* is used.
* **For the decoding and inference strategy:** Implicit *Teacher Forcing* is applied by passing complete sequences during training and *Greedy Decoding* (selecting the highest-probability token, filtering special tokens) during inference. The generated domains are verified to comply with RFC 1034/1035 syntactic rules before concatenating the TLD. If the model fails after 5 consecutive attempts, a *fallback* strategy based on concatenating random sub-tokens from the vocabulary is used.
* **For optimizers and initialization:** SGD (learning rate of 0.06) is used exclusively for the Autoencoder, while the GAN uses separate Adam optimizers (Generator: 0.0004, Discriminator: 0.0001). The default values $\beta_1 = 0.9$ and $\beta_2 = 0.999$ are used for Adam. All linear and projection layers use Xavier/Glorot uniform initialization.
* **For retraining and framework (Targeted Self-Augmentation):** For consistency with the framework, the framework's CNN detector (`detectors/cnn/`) is used instead of the classifiers mentioned in the paper. 1000 candidate domains are generated, and those that manage to evade this classifier are reintegrated into the legitimate dataset for retraining.
* **For TLDs and the dataset:** During the preparation phase, only the SLDs (Second-Level Domains) are extracted from a portion of 10,000 benign domains from Tranco, removing the TLDs. During generation, a static list of the most common TLDs (50 traditional TLDs and 50 new gTLDs, obtained from [https://domainnamestat.com/](https://domainnamestat.com/)) is used, where duplicates are assumed to be removed, resulting in a length of 89 instead of 100. A TLD is randomly selected from this list to complete the final domain.

## Luo et al., 2026

### SADGA

#### Information Gaps

* **Input vector dimensions:** The paper states that the Generator receives "noise" through a Dense layer, but does not specify the dimensionality of this latent space ($z$) or its statistical distribution.

* **Conv1D and Res-Blocks architecture hyperparameters:** The use of one-dimensional convolutions is mentioned, but the number of filters (channels), kernel size, and padding strategies to preserve the dimensions of the temporal sequence are omitted.

* **Projection and Reduction Formats:** It is not specified how the flat vector from the Generator's initial Dense layer is reshaped to feed the 3D residual block. Likewise, it is not detailed how the Discriminator collapses the final two-dimensional matrix before producing the scalar score.

* **WGAN-GP training hyperparameters:** Although the use of W-Loss with gradient penalty is mentioned, the paper omits the optimizer, learning rate, batch size, number of epochs, and update ratio between the Discriminator and the Generator.

* **Sequence Length and Padding:** Since BiMM segmentation produces variable lengths, a maximum temporal length must be defined to fix the network architecture, as well as how the padding token is handled.

* **Differentiability of the training cycle:** The paper states that the Generator output uses Softmax followed by an argmax to obtain words from the vocabulary. However, applying argmax within the training loop blocks the gradient flow because it is non-differentiable.

* **Ambiguity in domain filtering:** The text presents a contradiction by stating that long domains (longer than 6 characters) should be discarded to avoid collisions with short domains.

* **Internal Self-Attention Configuration:** It is not detailed whether the internal functions $f(x)$, $g(x)$, and $h(x)$ apply dimensionality reductions, which are important for the computational feasibility of the mechanism.

---

#### Assumptions

* **For latent noise and the initial architecture:** A noise vector of dimension 128 based on a standard normal distribution $z \sim \mathcal{N}(0, I)$ is assumed. In the Generator, the initial Dense layer projects this vector to a dimension of $128 \times 20$ and an explicit Reshape layer is applied to transform it into a three-dimensional tensor $(B, 20, 128)$ that the Conv1D layers can process.

* **For convolutional parameters and channels:** A base channel of $C = 128$ is assumed for the entire network. All Conv1D layers use a kernel_size = 3 with padding='same' to properly handle the sequence.

* **For the Self-Attention mechanism:** A standard state-of-the-art implementation is assumed, introducing channel-reduction projections: $f(x)$ and $g(x)$ reduce to $C/8$ (16 channels), while $h(x)$ reduces to $C/2$ (64 channels). The scalar multiplicative parameter $\gamma$ is implemented as a learnable weight strictly initialized to 0.

* **For differentiability and Softmax:** To avoid blocking gradients, the Generator is designed to output a continuous tensor of post-Softmax probabilities during training. The Discriminator directly receives these continuous values (relaxed One-Hot). The non-differentiable discrete function argmax is used exclusively during the domain generation/inference phase.

* **For length and vocabulary (Padding):** A parametric maximum length of 20 tokens is established. Dictionary index 0 is reserved for the padding token, increasing the vocabulary size from 6,673 to 6,674 to guarantee a constant tensor length.

* **For training (WGAN-GP):** The use of the Adam optimizer with a learning rate of 0.0001, $\beta_1 = 0.0$, and $\beta_2 = 0.9$ is assumed. A batch size of 64, training for 100 epochs, and 5 Discriminator updates for each Generator update are used.

* **For the Discriminator output:** The addition of a GlobalAveragePooling1D layer immediately after the last residual block is assumed. This allows the temporal dimension to be collapsed before the final Dense(1) layer, generating the scalar required by the W-Loss function.

* **For filtering and post-processing:** The removal of all domains with a length less than or equal to 6 is assumed. Only those that meet the minimum length of 7 and adhere to RFC specifications (letters, numbers, no hyphens at the ends) are retained.

* **For the BiMM Tokenizer:** In the event of a tie in the number of tokens generated between FMM (left to right) and BMM (right to left), the FMM result is selected. Additionally, it is assumed that full domains, including their TLDs, are used for tokenization.

* **For the Data:** The first 10,000 complete domains from the Tranco list are used as input. Complete domains are generated, not just the SLD.

* **For Lean Domain Search:** It is assumed that the most frequent words are read from the local file dataset/fixes.md (obtained from https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103), extracting only the lines prefixed with * and removing any possible hyphens from the tokens. If the file does not exist, we assume a fallback mechanism that extracts the most common words, with a length between 5 and 7, directly from the training domains.

* **TLD (framework convention):** The paper defines the generator on the second-level label only. Because the detectors are trained on full domains and the lexical metrics change with the TLD, the framework appends one to every generated SLD, drawn from the empirical public-suffix distribution of the D1 Tranco slice (top 50, `core/tld.py`). The same rule applies to every model in this situation so that the TLD is not a confounder between generators.
## Pelayo-Benedet et al., 2025 (RAMPAGE detectors)

### WoodbridgeLSTM, YuLSTM, Tweet2Vec, Tweet2Vec2, DBD, Zhang, YangCNN

The seven detectors reproduce the seven best models of the RAMPAGE evaluation ("RAMPAGE: A Software Framework To Ensure Reproducibility in Algorithmically Generated Domains Detection", Expert Systems With Applications; code at https://github.com/reverseame/RAMPAGE). Where the paper and its code disagree, the code is followed, because it is what produced the published numbers.

* **Seventh model:** The paper labels the seventh model "Parallel CNN (Yu et al., 2018)". Its appendix table and its Table 1 row correspond to RAMPAGE's `Expose` class, but the meta-model experiments (`APIrest/apirest.py`, `best.py`, `ablation.py`) ran `YangCNN` under that label. `YangCNN` is the one implemented.

* **CNN (Berman, 2019):** The appendix table under this label describes RAMPAGE's `CNNMaxPooling` class, which Table 1 reports as "Max Pooling (Berman, 2019)". The published results under "CNN (Berman, 2019)" come from the `Zhang` class, which is the one implemented.

* **Meta-model:** RAMPAGE combines the seven probabilities with a logistic regression (L2, C = 0.1). It is not implemented; each network is evaluated as an independent detector.

* **Training data and split:** RAMPAGE trained on its own datasets (UTL_DGA22 and Tranco, 250,000 + 250,000) with a 70/15/15 split. Here the seven detectors train on this framework's D2 (100,000 + 100,000, `core/data_splits.py`), the same set as the original detectors, holding out the last 20 percent of a seeded permutation (`SPLIT_SEED = 42`, shared by the seven) for validation. No test slice is taken from D2: D3 and the generated samples are the test data.

* **Epochs and early stopping:** The paper states 500 epochs with early stopping on validation accuracy and checkpointing of the best model, without giving the patience; the released code has `epochs = 1` and only a `ModelCheckpoint`. Up to 500 epochs, patience 10 on `val_accuracy` and restoring the best weights are used. Batch size 50, as in the code.

* **Input length:** RAMPAGE pads to 70 characters (`commonData.maxlen`; the appendix says 128). 75 is used, the length of the original detectors, so that no domain of the splits or of the samples is truncated (D3_benign holds a 75-character domain).

* **Encoding:** Characters map to `ord(c) - 33` as in RAMPAGE, on the lowercased domain (the splits and the samples are already lowercase). A character outside the embedding alphabet (256 or 128 symbols depending on the model) maps to the padding index instead of raising; the data contain none.

* **Optimizers:** Appendix B of the paper states a unified Adam (lr 0.001, weight decay 0.001) for all models, but the per-model tables and the code apply it only to the two LSTMs; the other four use the Keras default Adam and `YangCNN` RMSprop. The code is followed.

* **Decision threshold:** As in RAMPAGE, a domain is DGA when P(DGA) > 0.5; `detect()` returns benign for P(DGA) <= 0.5.
