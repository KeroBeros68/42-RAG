# Cahier des Charges — RAG against the machine

**Version :** 3.0
**Date :** 19/05/2026
**Auteur(s) :** kebertra
**Statut :** Brouillon

---

## Sommaire

1. [Présentation du projet](#1-présentation-du-projet)
2. [Périmètre fonctionnel](#2-périmètre-fonctionnel)
3. [Exigences techniques](#3-exigences-techniques)
4. [Architecture et intégrations](#4-architecture-et-intégrations)
5. [Livrables et planning](#5-livrables-et-planning)
6. [Contraintes et hypothèses](#6-contraintes-et-hypothèses)
7. [Critères de recette](#7-critères-de-recette)
8. [Annexes](#8-annexes)

---

## 1. Présentation du projet

### 1.1 Contexte

Ce projet s'inscrit dans le cadre du cursus pédagogique de l'école 42. Il fait suite au projet `call_me_maybe` (introduction au function calling) et constitue une nouvelle étape dans l'exploration des systèmes d'Intelligence Artificielle.

Le problème adressé est fondamental dans l'usage pratique des LLM (Large Language Models) : un modèle entraîné possède une connaissance figée à sa date de coupure et ne peut pas, nativement, répondre à des questions portant sur une base de code ou un corpus documentaire spécifique sans y avoir été exposé lors de l'entraînement. Lui demander de raisonner sur le dépôt vLLM sans contexte externe reviendrait à demander à quelqu'un de commenter un livre qu'il n'a jamais lu.

La technique **RAG (Retrieval-Augmented Generation)** résout ce problème en donnant au modèle un accès dynamique à une source de connaissance externe au moment de l'inférence, sans nécessiter de ré-entraînement. Le corpus de référence choisi est le dépôt open-source **vLLM v0.10.1**, un moteur d'inférence LLM haute performance écrit principalement en Python.

### 1.2 Objectifs

- **Objectif 1 — Ingestion :** Construire un pipeline capable de lire, segmenter et indexer l'intégralité du dépôt vLLM (fichiers `.py` et `.md`) en moins de 5 minutes, avec des stratégies de chunking distinctes via `RecursiveCharacterTextSplitter` LangChain : `Language.PYTHON` pour les `.py`, `Language.MARKDOWN` pour les `.md`. La taille de chunk est mesurée en tokens via le tokenizer `Qwen/Qwen3-0.6B` (`transformers`) en plus de la limite en caractères.
- **Objectif 2 — Retrieval hybride :** Maintenir un **index BM25 unique** (`bm25s`) couvrant l'ensemble du corpus (`.py` + `.md`) pour la recherche lexicale. Combiner cet index avec ChromaDB (embeddings sur les `.md` uniquement) via `EnsembleRetriever` LangChain (Reciprocal Rank Fusion), suivi d'un reranking cross-encoder. Recall@5 cible : ≥ 80% docs, ≥ 50% code.
- **Objectif 3 — Génération :** Intégrer `Qwen/Qwen3-0.6B` via un serveur vLLM local (API OpenAI-compatible sur `localhost:8000`), piloté par LangChain (`ChatOpenAI`) et un pipeline LCEL. DSPy optimise le prompt via `BootstrapFewShot` sur les datasets publics. Le contexte est borné en tokens avant envoi au LLM.
- **Objectif 4 — Évaluation :** Fournir un système d'évaluation basé sur la métrique recall@k permettant de mesurer objectivement la qualité du retrieval par rapport à des annotations de vérité-terrain.
- **Objectif 5 — Ergonomie :** Exposer l'ensemble des fonctionnalités via une interface CLI complète et robuste (Python Fire), avec gestion des erreurs et barres de progression.

### 1.3 Vision

> "Permettre à un développeur de poser n'importe quelle question sur le dépôt vLLM en langage naturel et d'obtenir une réponse précise, sourcée et non-hallucinatoire en moins de 90 secondes."

### 1.4 Parties prenantes

| Rôle | Identifiant | Responsabilité |
|------|-------------|----------------|
| Commanditaire / Évaluateur | Staff 42 | Validation finale via peer-evaluation et moulinette automatique |
| Développeur & Chef de projet | kebertra | Conception, développement, tests, documentation |
| Collaborateurs de référence | @ldevelle, @pcamaren, @crfernan | Co-auteurs du sujet, ressources de peer-review |

---

## 2. Périmètre fonctionnel

### 2.1 Fonctionnalités — Méthode MoSCoW

**Must have (indispensable)**
- [ ] Ingestion et indexation du dépôt vLLM avec chunking adaptatif par type de fichier
- [ ] Chunking `.py` via `RecursiveCharacterTextSplitter` LangChain avec `Language.PYTHON` — découpage sur les frontières sémantiques Python (classes, fonctions, blocs) avec fallback sliding window
- [ ] Chunking `.md` via `RecursiveCharacterTextSplitter` LangChain avec `Language.MARKDOWN` — découpage par headers (H1/H2/H3) avec fallback sliding window
- [ ] Validation de la taille des chunks en tokens via le tokenizer `Qwen/Qwen3-0.6B` (`transformers`)
- [ ] **Un seul index BM25 unifié** (`bm25s`) couvrant l'ensemble du corpus (`.py` + `.md`), persisté sous `data/processed/bm25_index/`
- [ ] Retrieval top-k par BM25 retournant `file_path`, `first_character_index`, `last_character_index`
- [ ] Génération de réponses avec `Qwen/Qwen3-0.6B` à partir du contexte récupéré
- [ ] Gestion du token budget du contexte avant envoi au LLM (truncation par token count)
- [ ] Sortie JSON conforme aux modèles Pydantic (`StudentSearchResults`, `StudentSearchResultsAndAnswer`)
- [ ] Métadonnées LangChain (`Document.metadata`) enrichies avec `file_path`, `first_character_index`, `last_character_index`, `file_type` à l'indexation
- [ ] CLI complète avec les commandes : `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`
- [ ] Taille de chunk configurable via argument CLI (`--max_chunk_size`, défaut : 2000 caractères)
- [ ] Système d'évaluation recall@k avec seuil de recouvrement à 5%
- [ ] Barres de progression (`tqdm`) sur toutes les opérations longues
- [ ] Respect des standards de code : `flake8`, `mypy`, `pydantic`, type hints, docstrings PEP 257

**Should have (important)**
- [ ] Index BM25 et index ChromaDB alimentés depuis les mêmes `Document` LangChain (source unique de vérité pour les chunks)
- [ ] Gestion robuste des erreurs CLI avec messages explicites sur les entrées dégénérées
- [ ] Fichier `.gitignore` configuré pour exclure artefacts Python, modèles et données générées
- [ ] Utilisation d'environnements virtuels via `uv` (isolation des dépendances)
- [ ] README.md complet (architecture, chunking, retrieval, performances, décisions de design, défis, exemples)

**Could have (souhaitable — bonus)**
- [ ] **Hybrid Retrieval** : fusion BM25 unifié (`.py` + `.md`) + ChromaDB (`.md`) via `EnsembleRetriever` LangChain (Reciprocal Rank Fusion automatique) — ChromaDB indexe uniquement les `.md` pour optimiser le ratio coût/bénéfice
- [ ] **Re-ranking cross-encoder** : après fusion RRF, un `CrossEncoderReranker` LangChain Community re-score le pool top-N (N = 3×k à 5×k) et sélectionne le top-k final
- [ ] **Pipeline LCEL complet** : chaînage `EnsembleRetriever | format_docs | ChatPromptTemplate | ChatOpenAI(vLLM) | StrOutputParser` avec `RunnableParallel` pour capturer sources + réponse simultanément
- [ ] **Inférence via serveur vLLM local** : `Qwen/Qwen3-0.6B` servi via `vllm.entrypoints.openai.api_server` sur `localhost:8000`, consommé par LangChain via `ChatOpenAI(base_url="http://localhost:8000/v1")` et par DSPy via `dspy.LM(model="openai/Qwen/Qwen3-0.6B", api_base=...)`
- [ ] **Query Expansion via DSPy** : `dspy.Signature` + `dspy.Predict` pour reformuler/enrichir la requête avant retrieval ; compilation `BootstrapFewShot` sur les `AnsweredQuestions` publics ; module compilé sauvegardé et rechargé pour l'inférence
- [ ] **Result Caching** : mise en cache des résultats de recherche (clé : requête + k + max_chunk_size) via `diskcache` ou dictionnaire JSON persisté

**Won't have (hors périmètre)**
- Interface graphique (GUI/web) — le projet est un outil CLI/batch
- Support d'autres dépôts que vLLM dans la partie obligatoire
- Authentification ou gestion multi-utilisateurs
- Déploiement en production (serveur, Docker, cloud)
- Vectorisation des fichiers `.py` complets dans ChromaDB (coût trop élevé ; seuls les docstrings extraits pourraient être envisagés en bonus additionnel)

---

### 2.2 User Stories

| ID | Rôle | Action souhaitée | Bénéfice | Priorité |
|----|------|-----------------|----------|----------|
| US-01 | Développeur | Lancer `index` sur le dépôt vLLM | Créer un index BM25 unifié (`.py` + `.md`) + un index ChromaDB (`.md`) persistés sur disque | Must |
| US-02 | Développeur | Lancer `search "ma question"` | Obtenir les k passages les plus pertinents avec leur localisation précise | Must |
| US-03 | Développeur | Lancer `search_dataset --dataset_path ...` | Traiter un jeu de questions en batch et sauvegarder les résultats | Must |
| US-04 | Développeur | Lancer `answer "ma question"` | Obtenir une réponse en langage naturel sourcée depuis le corpus | Must |
| US-05 | Développeur | Lancer `answer_dataset` sur les résultats de search | Générer des réponses pour tout un dataset en une commande | Must |
| US-06 | Évaluateur (moulinette) | Lancer `evaluate` avec un chemin vers les résultats étudiants | Calculer automatiquement recall@1, @3, @5, @10 | Must |
| US-07 | Développeur | Passer `--max_chunk_size 500` à la commande `index` | Expérimenter différentes granularités de chunking | Must |
| US-08 | Développeur | Activer le retrieval hybride BM25 + ChromaDB via `EnsembleRetriever` | Améliorer le recall sur les questions sémantiquement complexes | Could |
| US-09 | Développeur | Activer l'expansion de requête via DSPy (`BootstrapFewShot`) | Améliorer le recall sur les requêtes courtes ou ambiguës | Could |
| US-10 | Développeur | Démarrer le serveur vLLM et interroger `answer` ou `answer_dataset` | Bénéficier du batching automatique vLLM pour les 100 questions | Could |
| US-11 | Développeur | Consulter les sources récupérées dans la réponse JSON | Vérifier que les chunks cités existent réellement dans le corpus | Must |

### 2.3 Cas d'usage principaux

**Cas d'usage 1 : Indexation du dépôt (première utilisation)**

1. L'utilisateur place le dépôt vLLM dézippé sous `data/raw/vllm-0.10.1/`.
2. Il exécute : `uv run python -m student index --max_chunk_size 2000`
3. Le système parcourt récursivement les fichiers jugés utiles (`.py`, `.md`), applique `RecursiveCharacterTextSplitter` LangChain avec le langage adapté (`Language.PYTHON` ou `Language.MARKDOWN`), construit l'index BM25 unifié et le persiste sous `data/processed/bm25_index/`.
4. Les chunks sont sérialisés sous `data/processed/chunks/` avec leurs métadonnées (`file_path`, indices de caractères, `file_type`) pour permettre la reconstruction du contexte lors de la génération.
5. Le terminal affiche des barres de progression et confirme : `Ingestion complete! Indices saved under data/processed/`

*Cas d'erreur :* si le répertoire source est absent ou vide, le programme affiche un message d'erreur explicite et se termine avec un code de sortie non-nul sans lever d'exception non gérée.

---

**Cas d'usage 2 : Réponse à une question unitaire**

1. L'utilisateur exécute : `uv run python -m student answer "How to configure OpenAI server?" --k 10`
2. Le système charge l'index BM25 depuis le disque (cold start ≤ 60 s).
3. Il encode la requête, effectue le retrieval des 10 meilleurs chunks.
4. Il construit le contexte en respectant la limite de tokens du modèle.
5. `Qwen/Qwen3-0.6B` génère une réponse auto-contenue, sourcée et non-hallucinatoire.
6. La réponse est affichée dans le terminal au format JSON conforme au modèle `MinimalAnswer`.

*Cas d'erreur :* si l'index n'a pas encore été généré, le programme l'indique explicitement et suggère de lancer `index` d'abord.

---

**Cas d'usage 3 : Évaluation automatique (moulinette)**

1. La moulinette exécute : `uv run python -m moulinette evaluate_student_search_results --student_answer_path ... --dataset_path ... --k 10 --max_context_length 2000`
2. Le système compare chaque source récupérée par l'étudiant avec les sources de vérité-terrain.
3. Un recouvrement d'au moins 5% en termes d'indices de caractères entre une source récupérée et une source correcte compte comme "trouvé".
4. Les métriques recall@1, @3, @5, @10 sont calculées et affichées.

*Critère de succès :* recall@5 ≥ 0.80 sur `dataset_docs_public.json` et ≥ 0.50 sur `dataset_code_public.json`.

---

## 3. Exigences techniques

### 3.1 Stack technologique

| Composant | Technologie | Justification |
|-----------|------------|---------------|
| Langage | Python 3.10 | Imposé par le sujet ; compatibilité garantie avec l'ensemble de l'écosystème ML/NLP utilisé |
| Gestionnaire de paquets | `uv` | Imposé par le sujet ; alternative moderne et rapide à `pip`/`venv`, résolution de dépendances déterministe via `uv.lock` |
| Validation des données | `pydantic` v2 | Imposé par le sujet ; garantit l'intégrité des modèles de données tout au long du pipeline |
| Chunking `.py` | `RecursiveCharacterTextSplitter` LangChain (`Language.PYTHON`) | Découpage sur les frontières sémantiques Python (classes, fonctions, blocs) sans implémentation manuelle d'un parseur AST ; cohérent avec le chunker `.md` (même classe, même interface) |
| Chunking `.md` | `RecursiveCharacterTextSplitter` LangChain (`Language.MARKDOWN`) | Découpage par headers (H1/H2/H3) sans implémentation manuelle ; fiable sur du Markdown bien structuré comme la doc vLLM |
| Mesure taille chunks (tokens) | `AutoTokenizer` (`transformers`) | Valide le budget tokens de chaque chunk avec le tokenizer exact du modèle cible ; fallback sliding window si dépassement |
| Objets de données unifiés | `Document` LangChain (`langchain_core`) | Source unique de vérité pour les chunks : `page_content` + `metadata` (`file_path`, `first_character_index`, `last_character_index`, `file_type`) alimentent BM25, ChromaDB et le pipeline de génération |
| Retrieval lexical | `bm25s` + `BM25Retriever` LangChain Community | **Index unique** couvrant l'ensemble du corpus (`.py` + `.md`) ; simplicité de gestion, un seul fichier persisté ; `BM25Retriever` s'intègre nativement dans `EnsembleRetriever` |
| Retrieval vectoriel (bonus) | `chromadb` + `Chroma` LangChain | Base vectorielle embarquée sur les `.md` uniquement ; intégration native LangChain via `as_retriever()` |
| Fusion hybride (bonus) | `EnsembleRetriever` LangChain | Reciprocal Rank Fusion automatique entre BM25 et ChromaDB ; poids configurables (`weights=[0.5, 0.5]`) |
| Re-ranking (bonus) | `CrossEncoderReranker` LangChain Community | Pool top-N → cross-encoder → top-k final ; modèle recommandé : `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Pipeline de génération (bonus) | LCEL LangChain (`RunnableParallel`, `RunnablePassthrough`) | Chaînage déclaratif retriever → prompt → LLM → parser ; `RunnableParallel` capture sources + réponse simultanément |
| LLM | `Qwen/Qwen3-0.6B` | Imposé par le sujet ; modèle léger adapté CPU |
| Inférence LLM (base) | `transformers` (`AutoModelForCausalLM`) | Chargement direct en mémoire ; utilisé si le serveur vLLM n'est pas disponible |
| Inférence LLM (bonus) | Serveur `vllm` (`vllm.entrypoints.openai.api_server`) | Servi sur `localhost:8000` ; consommé par LangChain via `ChatOpenAI(base_url=...)` et par DSPy via `dspy.LM(model="openai/Qwen/Qwen3-0.6B", api_base=...)` ; batching automatique pour `answer_dataset` |
| Optimisation prompts (bonus) | `dspy` (`dspy.Signature`, `dspy.Predict`, `BootstrapFewShot`) | Compilation offline sur les `AnsweredQuestions` publics ; module compilé sauvegardé et rechargé pour l'inférence |
| CLI | `fire` (Python Fire) | Imposé par le sujet |
| Barres de progression | `tqdm` | Imposé par le sujet |
| Linting | `flake8` | Imposé par le sujet |
| Typage statique | `mypy` | Imposé par le sujet |
| Tests | `pytest` ou `unittest` | Recommandé par le sujet ; non soumis |

### 3.2 Performances

| Critère | Valeur cible | Justification |
|---------|-------------|---------------|
| Temps d'indexation | ≤ 5 minutes | Seuil imposé par le sujet pour garantir la praticité lors des évaluations |
| Cold start latency | ≤ 60 secondes | Temps de chargement initial du modèle + premier retrieval ; imposé par le sujet |
| Warm retrieval throughput | ≤ 90 secondes pour 1000 questions | Performance batch après le cold start ; imposé par le sujet |
| Recall@5 — questions docs | ≥ 80% | Seuil minimal d'acceptation pour la partie obligatoire |
| Recall@5 — questions code | ≥ 50% | Seuil minimal d'acceptation ; la recherche sur du code est intrinsèquement plus difficile |
| Taille maximale d'un chunk | 2000 caractères (configurable) | Équilibre entre granularité sémantique et coût computationnel |

> **Note sur l'environnement matériel :** le projet s'exécute sur une machine locale sous Linux Ubuntu, sans GPU dédié. Les temps de performance ci-dessus sont définis dans ce contexte CPU-only. L'utilisation de `Qwen/Qwen3-0.6B` (modèle 0.6B paramètres, ~1.2 Go en float16) a été validée pour fonctionner sur CPU, au prix d'une latence de génération plus élevée qu'avec GPU.

### 3.3 Sécurité

Compte tenu de la nature du projet (outil CLI local, sans réseau, sans données personnelles), les exigences de sécurité sont limitées mais non nulles.

- [ ] **Gestion des ressources :** Utilisation systématique de context managers (`with`) pour les handles de fichiers et connexions afin d'éviter les fuites (imposé par le sujet)
- [ ] **Robustesse des entrées :** Validation de toutes les entrées CLI via Pydantic avant tout traitement — les arguments mal formés ou manquants doivent produire un message d'erreur clair, pas un crash avec traceback
- [ ] **Exceptions gracieuses :** Tous les blocs potentiellement échouants (I/O, chargement de modèle, parsing JSON) sont encadrés par des `try/except` avec logging explicite
- [ ] **Conformité RGPD :** Non applicable — le projet ne traite aucune donnée personnelle
- [ ] **Intégrité des sorties :** Validation Pydantic des objets JSON avant écriture sur disque, garantissant la conformité du format pour la moulinette

### 3.4 Compatibilité

| Type | Cible |
|------|-------|
| OS d'exécution | Linux Ubuntu (machine locale de l'étudiant et de l'évaluateur) |
| Version Python | 3.10 strictement (imposé par le sujet) |
| Gestionnaire de paquets | `uv` (imposé) |
| Interface | CLI uniquement — pas d'interface navigateur |
| Architecture matérielle | x86-64, CPU uniquement (pas de CUDA requis) |

---

## 4. Architecture et intégrations

### 4.1 Schéma d'architecture

Le système est organisé en cinq sous-systèmes communiquant via le système de fichiers et une interface LangChain unifiée.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLI (Python Fire)                          │
│   index | search | search_dataset | answer | answer_dataset |       │
│   evaluate                                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼──────────────────────┐
        │                    │                       │
        ▼                    ▼                       ▼
┌───────────────────┐ ┌──────────────────────┐ ┌────────────────────┐
│  [1] INGESTION    │ │  [2] RETRIEVAL       │ │  [4] ÉVALUATION    │
│                   │ │                      │ │                    │
│ RCTS Language.PY  │ │  BM25Retriever       │ │  recall@k          │
│  → chunks .py     │ │  (index unifié       │ │  overlap ≥ 5%      │
│                   │ │   .py + .md)        ─┤ │  vs vérité-terrain │
│ RCTS Language.MD  │ │                      │ └────────────────────┘
│  → chunks .md     │ │  Chroma (.md)       ─┤
│                   │ │         ↓             │
│ transformers      │ │  EnsembleRetriever   │
│  tokenizer        │ │  (Fusion RRF)        │
│  → validation     │ │         ↓             │
│    token budget   │ │  CrossEncoderReranker│
│                   │ │  top-k final         │
│ Document LangChain│ └──────────┬───────────┘
│  metadata enrichie│            │
│         ↓         │            ▼
│ bm25_index/       │ ┌──────────────────────┐
│ ChromaDB (.md)    │ │  [3] GÉNÉRATION      │
│ chunks/*.json     │ │                      │
└───────────────────┘ │  token budget mgmt   │
                      │  (transformers)      │
                      │         ↓            │
                      │  ChatPromptTemplate  │
                      │  (LangChain)         │
                      │         ↓            │
                      │  ChatOpenAI          │
                      │  → localhost:8000    │
                      │  (vLLM server)       │
                      │         ↓            │
                      │  DSPy BootstrapFewShot│
                      │  (prompt optimisé)   │
                      │         ↓            │
                      │  RunnableParallel    │
                      │  answer + sources    │
                      │         ↓            │
                      │  MinimalAnswer JSON  │
                      └──────────────────────┘

Persistance sur disque :
  data/raw/vllm-0.10.1/              ← dépôt source (non versionné)
  data/processed/bm25_index/         ← index BM25 unifié (.py + .md)
  data/processed/chroma/             ← index vectoriel ChromaDB (.md)
  data/processed/chunks/all.json     ← tous les chunks avec métadonnées
  data/processed/dspy_compiled.json  ← module DSPy compilé (bonus)
  data/datasets/                     ← datasets Q&A (non versionnés)
  data/output/                       ← résultats de search et answers
```

### 4.2 Intégrations externes

| Service / Bibliothèque | Usage | Type d'intégration |
|------------------------|-------|-------------------|
| `langchain-text-splitters` | Chunking des `.py` via `RecursiveCharacterTextSplitter(Language.PYTHON)` et des `.md` via `RecursiveCharacterTextSplitter(Language.MARKDOWN)` — interface unifiée pour les deux types | Import Python direct |
| `langchain-core` | `Document` (chunks unifiés), `ChatPromptTemplate`, LCEL (`RunnableParallel`, `RunnablePassthrough`), `StrOutputParser` | Import Python direct |
| `langchain-community` | `BM25Retriever`, `EnsembleRetriever`, `CrossEncoderReranker`, `VLLMOpenAI` | Import Python direct |
| `langchain-openai` | `ChatOpenAI` pointé sur le serveur vLLM local (`base_url="http://localhost:8000/v1"`) | Import Python direct |
| `langchain-chroma` | `Chroma` vectorstore sur les chunks `.md`, exposé via `as_retriever()` | Import Python direct — base embarquée, sans serveur |
| `bm25s` | Moteur BM25 sous-jacent au `BM25Retriever` LangChain ; **instance unique** couvrant `.py` + `.md` | Import Python direct |
| `transformers` (HuggingFace) | `AutoTokenizer` pour mesure token budget des chunks et du contexte ; `AutoModelForCausalLM` pour inférence directe fallback | Import Python direct |
| `vllm` (bonus) | Serveur d'inférence `Qwen/Qwen3-0.6B` sur `localhost:8000` ; API OpenAI-compatible consommée par LangChain et DSPy | Processus local lancé via `subprocess` ou manuellement |
| `dspy` (bonus) | `dspy.Signature` + `dspy.Predict` pour encapsuler la génération RAG ; `BootstrapFewShot` pour compilation offline du prompt sur les datasets publics ; `dspy.LM` configuré sur le serveur vLLM | Import Python direct |

> **Note :** aucune intégration réseau externe n'est requise après le téléchargement initial des modèles HuggingFace. Le système fonctionne entièrement en local.

### 4.3 Modèle de données

Les modèles Pydantic suivants constituent le contrat de données du système. Ils sont imposés par le sujet et ne peuvent pas être restreints, seulement étendus.

**Modèles source (localisation d'un passage dans le corpus) :**
- `MinimalSource` : `file_path: str`, `first_character_index: int`, `last_character_index: int`

**Modèles question :**
- `UnansweredQuestion` : `question_id: str` (UUID auto-généré), `question: str`
- `AnsweredQuestion` hérite de `UnansweredQuestion` + `sources: List[MinimalSource]`, `answer: str`
- `RagDataset` : `rag_questions: List[AnsweredQuestion | UnansweredQuestion]`

**Modèles de résultats :**
- `MinimalSearchResults` : `question_id: str`, `question: str`, `retrieved_sources: List[MinimalSource]`
- `MinimalAnswer` hérite de `MinimalSearchResults` + `answer: str`
- `StudentSearchResults` : `search_results: List[MinimalSearchResults]`, `k: int`
- `StudentSearchResultsAndAnswer` hérite de `StudentSearchResults` — `search_results: List[MinimalAnswer]`

**Relations :** Un `RagDataset` contient N questions. Chaque question est associée à K sources (`MinimalSource`), chacune pointant vers un intervalle de caractères d'un fichier du corpus.

**Extension possible :** les modèles peuvent être étendus (champs supplémentaires) sans briser la compatibilité avec la moulinette, qui valide uniquement la présence des champs obligatoires.

---

## 5. Livrables et planning

### 5.1 Livrables

| Livrable | Description | Format | Remarque |
|----------|-------------|--------|----------|
| `src/` | Code source complet du projet | Python (modules) | Doit être exécutable via `uv run python -m student` |
| `pyproject.toml` | Déclaration des dépendances et métadonnées du projet | TOML | Géré par `uv` |
| `uv.lock` | Fichier de verrouillage des dépendances | Lockfile | Garantit la reproductibilité de l'environnement chez l'évaluateur |
| `Makefile` | Automatisation des tâches courantes | Makefile | Cibles obligatoires : `install`, `run`, `debug`, `clean`, `lint` |
| `README.md` | Documentation complète du projet | Markdown (EN) | Sections obligatoires listées en §8.2 |
| `.gitignore` | Exclusion des artefacts | Gitignore | Doit exclure `__pycache__`, `.mypy_cache`, `data/`, modèles |
| Tests | Programmes de vérification (non soumis) | `pytest` | Pour usage local pendant le développement |

> **Important :** les fichiers de données (dépôt vLLM, datasets, modèles, outputs générés) **ne doivent pas** être versionnés dans le dépôt Git. L'évaluateur les génère lui-même lors de l'évaluation.

### 5.2 Planning macro

Aucune date de rendu imposée n'a été communiquée. Le planning suivant est proposé à titre indicatif pour structurer le développement en solo :

```
Phase 1 — Fondations et ingestion          [J+0  → J+3 ]
  · Mise en place de l'environnement uv, Makefile, structure src/
  · Implémentation des modèles Pydantic
  · Chunking Python (AST-based) et Markdown (header-based)
  · Indexation BM25 et persistance

Phase 2 — Retrieval et évaluation          [J+3  → J+6 ]
  · CLI complète (fire) avec toutes les commandes
  · Retrieval BM25 top-k avec format de sortie JSON conforme
  · Système d'évaluation recall@k
  · Mesure baseline sur datasets publics

Phase 3 — Génération de réponses           [J+6  → J+9 ]
  · Intégration Qwen/Qwen3-0.6B via transformers
  · Gestion du budget de tokens (context truncation)
  · Commandes answer / answer_dataset
  · Tests end-to-end du pipeline complet

Phase 4 — Qualité et bonus                 [J+9  → J+14]
  · Optimisation recall (tuning chunking/BM25)
  · Implémentation des bonus (hybrid ChromaDB, query expansion DSPy,
    vLLM inference, caching, re-ranking)
  · Tests robustesse CLI (edge cases, entrées dégénérées)
  · Rédaction README.md complet

Phase 5 — Revue finale et soumission       [J+14 → J+16]
  · flake8 + mypy clean
  · Vérification structure repo Git
  · Préparation à l'évaluation (recode potentiel)
```

---

## 6. Contraintes et hypothèses

### 6.1 Contraintes

| Type | Description |
|------|-------------|
| Technique — langage | Python 3.10 strictement (pas 3.11+) |
| Technique — gestionnaire | `uv` obligatoire ; `pip` seul non accepté |
| Technique — modèle | `Qwen/Qwen3-0.6B` obligatoire comme modèle par défaut ; d'autres modèles peuvent coexister |
| Technique — CLI | Python Fire obligatoire ; argparse ou click non acceptés |
| Technique — chunking | Taille maximale : 2000 caractères (valeur par défaut configurable) |
| Technique — index | Temps d'indexation ≤ 5 minutes sur la machine d'évaluation |
| Performance | Cold start ≤ 60 s ; warm throughput ≤ 90 s / 1000 questions |
| Performance | Recall@5 ≥ 0.80 (docs), ≥ 0.50 (code) pour valider la partie obligatoire |
| Code — qualité | `flake8` sans warning, `mypy` sans erreur (flags obligatoires définis dans le Makefile) |
| Code — style | Type hints complets, docstrings PEP 257 (Google ou NumPy style) sur toutes les fonctions |
| Robustesse | Le programme ne doit jamais crash sur une entrée dégénérée — les exceptions non gérées invalident le projet |
| Ressources | Équipe solo (kebertra) — pas de budget alloué |
| Versionnement | Données volumineuses exclues du repo Git (modèles, dépôt vLLM, outputs) |
| Évaluation | Un "recode" partiel peut être demandé en séance d'évaluation — le code doit être maîtrisé dans son intégralité |

### 6.2 Hypothèses

- On suppose que l'évaluateur dispose d'une machine Linux Ubuntu avec `uv` installé et un accès internet pour télécharger les dépendances et le modèle HuggingFace.
- On suppose que `Qwen/Qwen3-0.6B` peut fonctionner en inférence CPU dans des délais acceptables pour la génération de réponses unitaires (pas nécessairement rapide, mais fonctionnel).
- On suppose que les datasets `dataset_docs_public.json` et `dataset_code_public.json` sont fournis via les pièces jointes du sujet et placés dans `data/datasets/` avant l'exécution.
- On suppose que les types de fichiers à indexer dans le dépôt vLLM sont principalement `.py` et `.md`. La décision d'inclure d'autres types (`.rst`, `.yaml`, `.txt`) est laissée à la discrétion du développeur et documentée dans le README.
- On suppose que la bibliothèque `chromadb` peut s'exécuter en mode embarqué (sans serveur dédié) dans l'environnement d'évaluation — ce mode est documenté et stable.
- On suppose que les performances décrites dans le sujet ont été définies pour une machine de l'école 42 de spécifications comparables à une machine locale standard (sans GPU).

---

## 7. Critères de recette

### 7.1 Processus de validation

1. **Validation automatique (moulinette)** : exécution de la commande `evaluate` sur les datasets publics et privés — vérification du format JSON et calcul des métriques recall@k.
2. **Validation de robustesse CLI** : passage d'arguments invalides, de chemins inexistants, de datasets vides — vérification que le programme ne crash pas.
3. **Validation de conformité code** : exécution de `make lint` — vérification que `flake8` et `mypy` passent sans erreur.
4. **Validation des performances temporelles** : mesure du temps d'indexation, de cold start et de throughput sur 1000 questions.
5. **Recode (si demandé)** : modification mineure du comportement d'une fonction ou d'une structure de données — validation de la compréhension réelle du code.
6. **Validation finale** : la note est attribuée par les pairs lors de la peer-evaluation selon la grille définie par le staff 42.

### 7.2 Critères d'acceptation

| Fonctionnalité | Critère de succès |
|---------------|------------------|
| Indexation | Termine en ≤ 5 min ; crée `data/processed/bm25_index/`, `data/processed/chunks/all.json` avec métadonnées complètes |
| Chunking | Chaque `Document` LangChain expose `file_path`, `first_character_index`, `last_character_index`, `file_type` dans ses métadonnées ; les indices sont corrects et `first < last` |
| Retrieval unitaire (`search`) | Retourne un JSON valide `StudentSearchResults` avec k sources ; chaque source a `file_path`, `first_character_index`, `last_character_index` |
| Retrieval batch (`search_dataset`) | Traite 100 questions en < 90 s après cold start ; sauvegarde un JSON valide |
| Génération (`answer`) | Retourne un JSON `MinimalAnswer` avec `answer` non vide, auto-contenu, sourcé |
| Token budget | Le contexte envoyé au LLM ne dépasse jamais la limite configurée (vérifiable via `count_tokens(context) ≤ max_context_tokens`) |
| Évaluation (`evaluate`) | Affiche recall@1, @3, @5, @10 ; recall@5 ≥ 0.80 docs, ≥ 0.50 code |
| Robustesse CLI | Aucun crash sur entrée dégénérée (path inexistant, JSON malformé, k=0, k négatif, serveur vLLM absent) |
| Conformité code | `make lint` passe sans erreur (`flake8` + `mypy` avec les flags obligatoires) |
| Structure repo | Contient `src/`, `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, `.gitignore` |

### 7.3 Tests attendus

- [ ] **Tests unitaires chunking** : RCTS `.py` (`Language.PYTHON`) et RCTS `.md` (`Language.MARKDOWN`) — vérifier que les frontières de découpage respectent la syntaxe, que le fallback sliding window se déclenche correctement, et que `first_character_index` / `last_character_index` sont valides et recalculables depuis le source original
- [ ] **Tests unitaires token budget** : `count_tokens()` sur des textes connus ; `build_context_within_budget()` avec budget serré
- [ ] **Tests unitaires modèles Pydantic** : validation et rejet des modèles ; calcul recall@k (k=0, overlap exactement 5%, aucune source trouvée)
- [ ] **Tests d'intégration** : pipeline complet `index → search → answer` sur un sous-ensemble réduit du corpus ; vérification que les métadonnées `Document` LangChain survivent à l'aller-retour BM25/ChromaDB
- [ ] **Tests hybrides (bonus)** : `EnsembleRetriever` retourne bien des sources des deux types de fichiers ; `CrossEncoderReranker` retourne bien top-k ≤ pool N
- [ ] **Tests de robustesse CLI** : arguments manquants, types incorrects, fichiers absents, datasets vides, `--k 0`, `--k -1`, `--max_chunk_size 1`, serveur vLLM non démarré
- [ ] **Tests de performance** : chronométrage indexation ≤ 5 min ; warm throughput ≤ 90 s / 1000 questions
- [ ] **Tests de conformité de sortie** : validation Pydantic des JSON produits par `search_dataset` et `answer_dataset`

> Les tests ne sont pas soumis ni évalués directement, mais leur absence rend le projet fragile face aux edge cases testés par la moulinette.

---

## 8. Annexes

### 8.1 Glossaire

| Terme | Définition |
|-------|-----------|
| RAG | Retrieval-Augmented Generation — technique combinant recherche dans un corpus externe et génération LLM pour produire des réponses ancrées dans des sources vérifiables |
| BM25 | Best Match 25 — algorithme de ranking probabiliste, extension de TF-IDF intégrant une normalisation de la longueur des documents ; standard de facto en recherche d'information |
| TF-IDF | Term Frequency–Inverse Document Frequency — pondération statistique mesurant l'importance relative d'un terme dans un document par rapport au corpus |
| Chunking | Segmentation d'un document en unités de taille fixe ou sémantique pour l'indexation ; critique pour la qualité du retrieval |
| RCTS | `RecursiveCharacterTextSplitter` — splitter LangChain qui découpe en respectant une liste ordonnée de séparateurs spécifiques au langage cible (`Language.PYTHON` : classes, fonctions, blocs ; `Language.MARKDOWN` : headers, paragraphes) avec fallback sur des séparateurs plus fins |
| Document LangChain | Objet `langchain_core.documents.Document` composé de `page_content` (texte du chunk) et `metadata` (dict) ; unité de données commune à tous les composants LangChain (BM25Retriever, Chroma, LCEL) |
| Token budget | Nombre maximal de tokens alloués au contexte envoyé au LLM ; mesuré via `AutoTokenizer.encode()` pour correspondre exactement à la fenêtre de contexte du modèle cible |
| Recall@k | Métrique d'évaluation : parmi les k résultats retournés, quelle proportion des sources correctes ont été retrouvées |
| Overlap | Recouvrement d'intervalles de caractères entre une source récupérée et une source de référence ; ≥ 5% = trouvé |
| Cold start | Temps du premier retrieval après démarrage du système, incluant le chargement du modèle en mémoire |
| Warm throughput | Débit de traitement après le cold start, une fois le modèle chargé |
| Embeddings | Représentation vectorielle d'un texte dans un espace de haute dimension permettant la comparaison sémantique par similarité cosinus |
| Hybrid retrieval | Combinaison d'une recherche lexicale (BM25 unifié sur `.py` + `.md`) et sémantique (embeddings ChromaDB sur `.md`) fusionnées via `EnsembleRetriever` (RRF) |
| RRF | Reciprocal Rank Fusion — algorithme de fusion de listes de résultats qui combine les rangs de plusieurs retriever sans nécessiter de normalisation des scores |
| EnsembleRetriever | Composant LangChain implémentant la fusion RRF entre plusieurs retrievers hétérogènes (BM25 + Chroma) avec poids configurables |
| Query expansion | Technique consistant à enrichir la requête originale avec des termes synonymes ou reformulations avant le retrieval ; implémentée via DSPy |
| Re-ranking | Passage d'un cross-encoder (`CrossEncoderReranker`) sur le pool fusionné pour affiner le classement final ; N candidats → top-k |
| Cross-encoder | Modèle de scoring prenant la paire (requête, passage) en entrée ; plus précis qu'un bi-encoder mais non précomputé ; appliqué en dernier pour minimiser le nombre d'inférences |
| LCEL | LangChain Expression Language — syntaxe de chaînage déclaratif (`|`) permettant de composer des pipelines retriever → prompt → LLM → parser |
| RunnableParallel | Composant LCEL exécutant plusieurs branches en parallèle et retournant un dict ; utilisé ici pour capturer simultanément `answer` et `sources` |
| vLLM | Bibliothèque open-source d'inférence LLM haute performance exploitant le PagedAttention pour maximiser le débit ; utilisé ici comme serveur local OpenAI-compatible |
| PagedAttention | Mécanisme de gestion de la mémoire KV-cache de vLLM inspiré de la pagination OS ; permet le batching dynamique et réduit les latences |
| DSPy | Framework de programmation de pipelines LLM ; `dspy.Signature` déclare le contrat entrées/sorties, `BootstrapFewShot` optimise automatiquement le prompt sur des exemples annotés |
| BootstrapFewShot | Optimiseur DSPy qui explore des variations de prompts et sélectionne les exemples few-shot maximisant la métrique cible ; compilation offline, inférence via module sauvegardé |
| ChromaDB | Base de données vectorielle embarquée (sans serveur dédié), utilisée ici pour l'index sémantique des `.md` |
| Pydantic | Bibliothèque Python de validation de données par déclaration de types ; utilisée ici pour garantir la conformité des formats JSON |
| Moulinette | Système d'évaluation automatique de l'école 42 exécutant les tests sur le projet soumis |
| MVP | Minimum Viable Product — version minimale fonctionnelle satisfaisant les exigences obligatoires |
| CDC | Cahier des charges |
| MoSCoW | Méthode de priorisation : Must have / Should have / Could have / Won't have |
| LLM | Large Language Model — modèle de langage de grande taille entraîné sur des corpus massifs |

### 8.2 Documents de référence

**Sujet officiel :**
- `rag.pdf` — sujet du projet (version 1.6), co-rédigé par @ldevelle, @pcamaren, @crfernan

**Documentation technique :**
- [HuggingFace — Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [bm25s — documentation officielle](https://github.com/xhluca/bm25s)
- [LangChain — Text Splitters](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
- [LangChain — BM25Retriever](https://python.langchain.com/docs/integrations/retrievers/bm25/)
- [LangChain — EnsembleRetriever](https://python.langchain.com/docs/how_to/ensemble_retriever/)
- [LangChain — LCEL](https://python.langchain.com/docs/concepts/lcel/)
- [LangChain — ChatOpenAI](https://python.langchain.com/docs/integrations/chat/openai/)
- [ChromaDB — documentation officielle](https://docs.trychroma.com)
- [DSPy — documentation officielle](https://dspy-docs.vercel.app)
- [vLLM — documentation officielle](https://docs.vllm.ai)
- [vLLM — OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Python Fire — documentation officielle](https://github.com/google/python-fire)
- [Pydantic v2 — documentation officielle](https://docs.pydantic.dev/latest/)
- [uv — documentation officielle](https://docs.astral.sh/uv/)
- [LangChain — Text Splitters (Language enum)](https://python.langchain.com/docs/how_to/code_splitter/)

**Articles de référence sur le RAG :**
- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (article fondateur du RAG)
- Robertson & Zaragoza (2009) — *The Probabilistic Relevance Framework: BM25 and Beyond*
- Cormack et al. (2009) — *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*

**README.md — sections obligatoires pour ce projet :**
La première ligne doit être italique : *This project has been created as part of the 42 curriculum by kebertra.*
Le README doit contenir : Description, Instructions (installation/exécution), Resources (références + usage de l'IA), System architecture, Chunking strategy, Retrieval method, Performance analysis, Design decisions, Challenges faced, Example usage.

### 8.3 Description des fonctionnalités bonus

Les bonus suivants sont tous envisagés pour implémentation, classés par ordre de priorité et d'impact sur le recall. L'architecture a été conçue pour les accueillir nativement via LangChain.

**Bonus 1 — Hybrid Retrieval BM25 + ChromaDB via `EnsembleRetriever`** *(priorité haute)*
L'index BM25 unifié (`.py` + `.md`) et l'index ChromaDB (`.md`) sont exposés comme des `Retriever` LangChain et fusionnés automatiquement via `EnsembleRetriever` (RRF). ChromaDB est limité aux `.md` pour optimiser le ratio coût/bénéfice : les fichiers `.md` contiennent du langage naturel, ce pour quoi les modèles d'embeddings ont été entraînés. Le code `.py` reste couvert par BM25 seul, où la correspondance lexicale exacte est plus pertinente. Les poids RRF sont configurables (`weights=[0.5, 0.5]` par défaut).

**Bonus 2 — Re-ranking cross-encoder** *(priorité haute)*
Après la fusion RRF (`EnsembleRetriever`), un `CrossEncoderReranker` LangChain Community (modèle recommandé : `cross-encoder/ms-marco-MiniLM-L-6-v2`) re-score les paires (requête, chunk) du pool top-N (N = 3×k à 5×k) et retient le top-k final. Le reranker est appliqué **après** la fusion, en un seul appel, pour minimiser le coût computationnel (CPU).

**Bonus 3 — Pipeline LCEL complet** *(priorité haute)*
Le pipeline de génération est implémenté en LCEL : `RunnableParallel({"answer": retriever | format_docs | prompt | llm | parser, "sources": retriever})`. Cette approche capture simultanément la réponse et les sources récupérées dans un seul appel, sans duplication du retrieval. La conversion vers `MinimalAnswer` (Pydantic) s'effectue depuis le dict retourné par `RunnableParallel`.

**Bonus 4 — Inférence via serveur vLLM local** *(priorité moyenne)*
`Qwen/Qwen3-0.6B` est servi via `vllm.entrypoints.openai.api_server` sur `localhost:8000`. LangChain le consomme via `ChatOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")` — aucun changement dans le reste du pipeline. DSPy le consomme via `dspy.LM(model="openai/Qwen/Qwen3-0.6B", api_base="http://localhost:8000/v1")`. Le batching automatique de vLLM réduit significativement le temps de `answer_dataset` sur 100 questions en CPU.

**Bonus 5 — Query Expansion via DSPy** *(priorité moyenne)*
`dspy.Signature` déclare le contrat (requête → variantes enrichies), `dspy.Predict` encapsule la génération. `BootstrapFewShot` compile le module offline sur les `AnsweredQuestions` publics et sauvegarde le résultat sous `data/processed/dspy_compiled.json`. À l'inférence, le module compilé est rechargé — la compilation n'a lieu qu'une fois.

**Bonus 6 — Result Caching** *(priorité basse)*
Mise en cache des résultats de recherche via `diskcache` (clé : hash(requête + k + max_chunk_size)). Élimine les recalculs sur des requêtes identiques, particulièrement utile lors du développement et du tuning des paramètres.

### 8.4 Historique des révisions

| Version | Date | Auteur | Modifications |
|---------|------|--------|--------------|
| 1.0 | 19/05/2026 | kebertra | Création initiale du cahier des charges |
| 2.0 | 19/05/2026 | kebertra | Architecture technique finalisée : deux index BM25 séparés (`.py`/`.md`), LangChain comme framework unificateur, `ast` stdlib pour chunking Python, serveur vLLM, DSPy `BootstrapFewShot` |
| 3.0 | 19/05/2026 | kebertra | Simplification architecture : suppression `ast` stdlib remplacé par `RecursiveCharacterTextSplitter(Language.PYTHON)` LangChain (cohérence avec le chunker `.md`) ; fusion des deux index BM25 en un seul index unifié (`.py` + `.md`) — réduction de la complexité de gestion sans perte de qualité de retrieval |