import os
import random
import logging
import numpy as np
import h5py
from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D1_BENIGN, load_tranco, load_malicious_seeds

logger = logging.getLogger(__name__)

class GADGAModel(AdversarialModel):
    """
    Implementation of GADGA model proposed by Alaeiyan et al. (2020).

    REPRODUCIBILITY ASSUMPTIONS:
    1. Fitness function: F(x) = alpha * S_pronunciation - beta * chi_sq_normalized
       where S_pronunciation is the log-probability of the domain under a
       character-level bigram model (trained on benign domains from Tranco D1),
       and chi_sq_normalized is chi-squared divided by the maximum possible
       value (domain length). This balances both terms, as the paper specifies
       a composition of pronunciation score and chi-square without giving an
       explicit formula.
    2. Mutation: standard 5% rate, altering exactly one character per mutation.
    3. Parent selection: purely uniform random sampling (selectTwoRandomSample),
       as the paper's pseudocode indicates; selection pressure is applied only
       via elitism (retaining the top 20% of the population).
    4. Crossover: single-point crossover with variable lengths; the cut point k
       is chosen uniformly in [1, min(len(p1), len(p2))-1] to avoid out-of-range
       errors.
    5. Handling of maxGen: the paper's pseudocode (Algorithm 2) uses a for-loop
       up to maxGen=400, but the population size is fixed to 100. To resolve this
       contradiction, we interpret the loop as a mechanism to fill the 80 vacant
       slots (after elite retention) with unique offspring; we use a while loop
       until we have generated exactly candidate_needed individuals.
    6. TLD and DNS syntax: the genetic operations (crossover and mutation) are
       applied only on the Second-Level Domain (SLD). The TLD is preserved and
       randomly chosen from a list of 22 common TLDs. Leading/trailing hyphens are 
       stripped to maintain validity.
    7. Seed population (initPop): we perform uniform random sampling from all
       available malicious domain CSVs without considering family labels.
    8. Collision handling: duplicates are prevented by using a set for candidate
       offspring, ensuring population diversity.
    9. Domain length: we enforce a minimum length of 3 and a maximum of 63
       characters for the SLD.
    10. GA execution: instead of running the full genetic algorithm on every call
        to generate_domain(), we run it once in fit() (or load a cached final
        population) and then sample from the final population.
    """

    def __init__(self, name: str = "GADGA", **kwargs):
        super().__init__(name=name, **kwargs)

        # GADGA hyperparameters
        self.pop_size = 100
        self.max_iterations = 4
        self.mutation_rate = 0.05
        self.elite_retention = int(self.pop_size * 0.20)  # 20%
        self.candidate_needed = self.pop_size - self.elite_retention

        # Allowed alphabet for domains (SLD)
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-"

        # Fitness weights
        self.alpha = 1.0
        self.beta = 0.1  # chi-squared is scaled down to match pronunciation range

        # Language model for pronunciation score
        self.first_letter_probs = {}   # P(c) for first character of a word
        self.bigram_probs = {}         # P(c2 | c1) smoothed
        self.vocab = set(self.chars)

        self.seed_population = []
        self.final_population = []     # Population after GA evolution

        # Caching paths
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.weights_path = os.path.join(self.weights_dir, "gadga_state.h5")

        # Common TLDs
        self.tlds = [
            'com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'cz', 'de', 'eu',
            'fr', 'info', 'it', 'ru', 'lv', 'me', 'name', 'net', 'nz', 'org', 'us'
        ]

        self.trained_from_scratch = False

    def fit(self) -> None:
        """
        Prepares the model:
        1. Loads or builds the language model (bigram probabilities) from Tranco D1.
        2. Loads or generates the initial seed population (initPop).
        3. Runs the genetic algorithm once to obtain the final population.
        4. Caches everything to disk for future runs.
        """
        if self.is_fitted:
            return

        # Try to load everything from cache
        if os.path.exists(self.weights_path):
            try:
                self._load_cache()
                self.is_fitted = True
                self.trained_from_scratch = False
                return
            except Exception as e:
                logger.warning(f"[*] Failed to load GADGA cache: {e}. Rebuilding from scratch.")

        # Build language model from benign domains (Tranco D1)
        self._build_language_model()

        # Build seed population (initPop)
        self._build_seed_population()

        # Run the genetic algorithm to obtain the final population
        self.final_population = self._run_ga(self.seed_population)

        # Cache everything
        self._save_cache()

        self.is_fitted = True
        self.trained_from_scratch = True

    def _build_language_model(self):
        """
        Trains a bigram language model on benign domains (Tranco D1).
        """
        benign_domains = load_tranco(TRANCO_D1_BENIGN)
        slds = [d.split('.')[0].lower() for d in benign_domains if d and len(d.split('.')[0]) > 0]

        first_counts = {}
        bigram_counts = {}

        for sld in slds:
            if len(sld) < 2:
                continue
            # First character
            first_counts[sld[0]] = first_counts.get(sld[0], 0) + 1
            # Bigrams
            for i in range(len(sld) - 1):
                bigram = sld[i:i+2]
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        # Apply Laplace smoothing
        vocab_size = len(self.vocab)
        total_first = sum(first_counts.values())
        self.first_letter_probs = {
            c: (first_counts.get(c, 0) + 1) / (total_first + vocab_size)
            for c in self.vocab
        }

        # For bigrams, we compute conditional probabilities P(c2 | c1) with smoothing
        # First, count occurrences per first character
        first_char_counts = {}
        for bigram, count in bigram_counts.items():
            first_char = bigram[0]
            first_char_counts[first_char] = first_char_counts.get(first_char, 0) + count

        self.bigram_probs = {}
        for bigram, count in bigram_counts.items():
            first_char = bigram[0]
            denom = first_char_counts.get(first_char, 0) + vocab_size
            # Laplace smoothing: (count + 1) / (total_bigrams_for_first + vocab_size)
            self.bigram_probs[bigram] = (count + 1) / denom

        # For unseen bigrams, we assign a very low probability (smoothing)
        # We will handle missing bigrams with a fallback

    def _build_seed_population(self):
        """
        Generates the initial seed population from real AGDs (core.data_splits.load_malicious_seeds).
        """
        agd_domains = []
        try:
            # Real AGDs come from the shared, row-bounded seed draw in core/data_splits.py.
            # Oversample so the length filter below still leaves enough candidates.
            seeds = load_malicious_seeds(total=max(self.pop_size * 10, 1000))
            agd_domains = [d.split('.')[0] for d in seeds if len(d.split('.')[0]) > 3]  # TLD isolation
        except Exception as e:
            logger.warning(f"[*] Warning reading AGDs for GADGA: {e}")

        # Fallback: generate random domains if not enough
        if len(agd_domains) < self.pop_size:
            fallbacks = [
                "".join(random.choices(self.chars[:26], k=random.randint(8, 15)))
                for _ in range(self.pop_size - len(agd_domains))
            ]
            agd_domains.extend(fallbacks)

        # Uniform sampling (ignoring family labels)
        self.seed_population = random.sample(agd_domains, self.pop_size)

    def _run_ga(self, population):
        """
        Executes the genetic algorithm for max_iterations and returns the final population.
        """
        # Copy to avoid modifying seed
        pop = list(population)

        for iteration in range(self.max_iterations):
            candidate_set = set()
            attempts = 0
            max_attempts = 2000  # Safety guard to prevent infinite loops

            # Fill the candidate pool (candidate_needed individuals)
            while len(candidate_set) < self.candidate_needed and attempts < max_attempts:
                attempts += 1
                p1, p2 = self._select_two_random_sample(pop)
                s1, s2 = self._crossover(p1, p2)
                s1 = self._mutation(s1)
                s2 = self._mutation(s2)

                # Clean and validate
                s1 = s1.strip('-')
                s2 = s2.strip('-')
                
                if s1 and (3 <= len(s1) <= 63) and (s1 not in pop):
                    candidate_set.add(s1)
                if s2 and (3 <= len(s2) <= 63) and (s2 not in pop) and (len(candidate_set) < self.candidate_needed):
                    candidate_set.add(s2)

            # Evaluate and select elite from current population
            pop.sort(key=self._fitness_function, reverse=True)
            elite = pop[:self.elite_retention]

            # Select best candidates from the candidate pool
            candidates = list(candidate_set)
            candidates.sort(key=self._fitness_function, reverse=True)
            best_candidates = candidates[:self.candidate_needed]

            # Rebuild population
            pop = elite + best_candidates
            
            # If population shrank below pop_size due to an early breakout, backfill with elite copies
            while len(pop) < self.pop_size and len(elite) > 0:
                pop.append(random.choice(elite))

        return pop

    def _select_two_random_sample(self, population):
        """
        Uniform random parent selection.
        """
        return random.sample(population, 2)

    def _crossover(self, p1, p2):
        """
        Single-point crossover with variable length.
        """
        min_len = min(len(p1), len(p2))
        if min_len <= 1:
            return p1, p2
            
        # k is between 1 and the length of the shortest domain
        k = random.randint(1, min_len - 1)
        s1 = p1[:k] + p2[k:]
        s2 = p2[:k] + p1[k:]
        return s1, s2

    def _mutation(self, domain):
        """
        Point mutation: 5% chance, changes one character.
        """
        if random.random() < self.mutation_rate and len(domain) > 0:
            idx = random.randint(0, len(domain) - 1)
            random_char = random.choice(self.chars)
            domain = domain[:idx] + random_char + domain[idx+1:]
        return domain

    def _pronunciation_score(self, domain):
        """
        Computes the log-probability of the domain under the bigram language model.
        Returns a score in [0,1] after sigmoid scaling for better numerical stability.
        """
        if not domain:
            return 0.0

        # Compute log-probability: P(first) * ∏ P(c_i | c_{i-1})
        log_prob = 0.0
        first_char = domain[0]
        # First character probability
        prob_first = self.first_letter_probs.get(first_char, 1e-6)
        log_prob += np.log(prob_first)

        # Bigram probabilities
        for i in range(len(domain) - 1):
            bigram = domain[i:i+2]
            prob = self.bigram_probs.get(bigram, 1e-6)  # fallback for unseen
            log_prob += np.log(prob)

        # Average log-probability per character, normalizing by length to avoid bias 
        # towards shorter domains
        avg_log = log_prob / len(domain)
        raw_prob = np.exp(avg_log)  # geometric mean of probabilities
        return min(raw_prob, 1.0)

    def _chi_squared_normalized(self, domain):
        """
        Computes chi-squared statistic and normalizes it by the maximum possible value
        (which is approximately domain length). Returns a value in [0,1].
        """
        if len(domain) == 0:
            return 1.0
        k = 4
        expected = len(domain) / k
        if expected == 0:
            return 0.0
        bins = [0] * k
        bin_size = len(self.chars) / k
        for char in domain:
            idx = self.chars.find(char)
            if idx != -1:
                bin_idx = min(int(idx / bin_size), k - 1)
                bins[bin_idx] += 1
        chi_sq = sum(((obs - expected) ** 2) / expected for obs in bins)
        max_chi = len(domain) * (k - 1)
        if max_chi == 0:
            return 0.0
        return chi_sq / max_chi

    def _fitness_function(self, domain):
        """
        F(x) = alpha * S_pronunciation - beta * chi_sq_normalized
        """
        s_pron = self._pronunciation_score(domain)
        chi_norm = self._chi_squared_normalized(domain)
        return self.alpha * s_pron - self.beta * chi_norm

    def generate_domain(self) -> str:
        """
        Returns a random domain from the final evolved population.
        """
        if not self.is_fitted:
            self.fit()

        if not self.final_population:
            logger.error("[*] GADGA final population is empty. Generating an algorithmic fallback.")
            fallback_sld = "".join(random.choices(self.chars[:26], k=random.randint(8, 15)))
            return f"{fallback_sld}.{random.choice(self.tlds)}"

        # Choose a random individual
        base_sld = random.choice(self.final_population)
        
        # Light mutation to make it unique, with a 50% probability.
        if random.random() < 0.5 and len(base_sld) > 3:
            idx = random.randint(0, len(base_sld) - 1)
            random_char = random.choice(self.chars)
            if idx == 0 or idx == len(base_sld) - 1:
                random_char = random.choice(self.chars[:26]) # only letters
            base_sld = base_sld[:idx] + random_char + base_sld[idx+1:]

        tld = random.choice(self.tlds)
        return f"{base_sld}.{tld}"

    # ---------- Caching helpers ----------
    def _save_cache(self):
        """
        Saves seed population, language model, and final population to HDF5.
        """
        os.makedirs(self.weights_dir, exist_ok=True)
        try:
            with h5py.File(self.weights_path, 'w') as f:
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset("seed_population", data=self.seed_population, dtype=dt)
                f.create_dataset("final_population", data=self.final_population, dtype=dt)
                # Save language model as JSON strings
                import json
                f.create_dataset("first_letter_probs", data=json.dumps(self.first_letter_probs), dtype=dt)
                f.create_dataset("bigram_probs", data=json.dumps(self.bigram_probs), dtype=dt)
                # Save hyperparameters
                f.create_dataset("alpha", data=self.alpha)
                f.create_dataset("beta", data=self.beta)
        except Exception as e:
            logger.error(f"[*] Error saving GADGA cache: {e}")

    def _load_cache(self):
        """
        Loads all cached data from HDF5.
        """
        import json
        with h5py.File(self.weights_path, 'r') as f:
            self.seed_population = [
                x.decode('utf-8') if isinstance(x, bytes) else x
                for x in f["seed_population"][:]
            ]
            self.final_population = [
                x.decode('utf-8') if isinstance(x, bytes) else x
                for x in f["final_population"][:]
            ]
            self.first_letter_probs = json.loads(
                f["first_letter_probs"][()].decode('utf-8') if isinstance(f["first_letter_probs"][()], bytes) else f["first_letter_probs"][()]
            )
            self.bigram_probs = json.loads(
                f["bigram_probs"][()].decode('utf-8') if isinstance(f["bigram_probs"][()], bytes) else f["bigram_probs"][()]
            )
            if "alpha" in f:
                self.alpha = float(f["alpha"][()])
            if "beta" in f:
                self.beta = float(f["beta"][()])

# Quick test
if __name__ == "__main__":
    gadga = GADGAModel()
    gadga.fit()
    print("GADGA generated domain:", gadga.generate_domain())