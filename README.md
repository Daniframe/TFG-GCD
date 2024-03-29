# Code of 'Model robustness under data distribution shifts. Analysing and predicting the impact of text perturbations on NLP models' 
Code employed for my Final Year Project in Data Science 'Model robustness under data distribution shifts. Analysing and predicting the impact of text perturbations on NLP models' conducted in the Univeristat Politècnica de València (UPV)

------

Large language models are usually trained using curated datasets, which lack impurities such as typographic errors, contractions, etc. Therefore, there is a gap between the training data of these models and the data they encounter in deployment situations. This work evaluates the robustness of four models in five different Natural Language Processing tasks against perturbed inputs. For that purpose, three perturbation types are analysed: character level perturbations, word level perturbations, and other types of perturbations. Datasets are perturbed and their predictions are compared against those of the unaltered datasets. Results show that models are sensitive to perturbed inputs, with some models being more sensitive that others depending on the task and the perturbation type. Precisely, the XLNet model is in general the most robust, and the most sensitive task is grammatical coherence.

Keywords : Natural Language Processing, text perturbation, robustness, transformers, natural language inference, sentiment analysis, hate speech and offensive language, semantic similarity, linguistic acceptability

------

The repository is organised as follows:

- FINETUNING: code and results for the finetuning phase, including instance-level results
- EVALUATION: code and results for the evaluation phase

Each folder is divided by model and then by NLP task. 
