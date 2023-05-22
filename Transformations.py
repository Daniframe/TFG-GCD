from __future__ import annotations
from numpy.typing import ArrayLike

#PERTURBATIONS
from textflint.input.component.sample.ut_sample import UTSample
from textflint.generation.transformation.transformation import Transformation
from textflint.generation.transformation.UT import (
    Contraction,
    InsertAdv,
    Keyboard,
    Ocr,
    Prejudice,
    Punctuation,
    ReverseNeg,
    SpellingError,
    SwapNamedEnt,
    SwapNum,
    SwapSynWordNet,
    Tense,
    TwitterType,
    Typos,
    WordCase
)

#DATASETS & SAMPLES
from textflint.input.dataset.dataset import Dataset as TextflintDataset
from textflint.input.component.sample.ut_sample import UTSample

from datasets import Dataset

#HUGGINFACE FINE-TUNING DATASET PREPARATIONS
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification
from tensorflow import keras

#MISCELANEOUS
import numpy as np
import re

REMOVE_SPACES_PATTERN = r"([.,!?])\s*"

ALLOWED_CHARACTER_LEVEL_PERTURBATIONS = {
    "Keyboard" : Keyboard,
    "Ocr" : Ocr,
    "SpellingError" : SpellingError,
    "Typos" : Typos
}

ALLOWED_WORD_LEVEL_PERTURBATIONS = {
    "SNE" : SwapNamedEnt,
    "SSWN" : SwapSynWordNet,
}

ALLOWED_OTHER_PERTURBATIONS = {
    "Contraction" : Contraction,
    "InsertAdv": InsertAdv,
    "Prejudice" : Prejudice,
    "Punctuation" : Punctuation,
    "ReverseNeg" : ReverseNeg,
    "SwapNum" : SwapNum,
    "VerbTense" : Tense,
    "Twitter" : TwitterType,
    "WordCase" : WordCase
}

def text_to_utsample(
    text: str,
    x_field_name: str = "x") -> UTSample:

    """
    Transforms a sentence into a textflint UTSample

    Args:
    ------------------------------------------------------
    text: str
        Sentence to transform
    
    x_field_name: str, default = "x"
        Name of the textflint field containing the text.
        For the majority of the functionalities of
        textflint to work, this field must be named "x"
    ------------------------------------------------------

    Returns:
    ------------------------------------------------------
    UTSample
        Textflint Universal Transformation Sample
    ------------------------------------------------------
    """

    return UTSample(data = {x_field_name : text})

def load_classifier_model(
    checkpoint: str,
    num_labels: int) -> object:

    return TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels)

def configure_dynamic_lr(
    scheduler_fn: function,
    num_train_steps: int) -> object:

    lr_scheduler = scheduler_fn(
        initial_learning_rate = 5e-5,
        end_learning_rate = 0.0,
        decay_steps = num_train_steps)
    
    return keras.optimizers.Adam(learning_rate = lr_scheduler)

def save_model_result_csv(
    observed: ArrayLike,
    namefile: str,
    label = "label") -> None:

    cad = f"index;{label}\n"
    for i, value in enumerate(observed):
        cad += f"{i};{value}\n"
    
    with open(namefile, "w", encoding = "utf-8") as file:
        file.write(cad)

class CharacterPerturbation:
    def __init__(self,
                 perturbation_type: str,
                 proportion_characters: float,
                 **kwargs):

        """
        Perturbation that alters the characters of a sentence

        Args:
        ------------------------------------------------------
        proportion_characters: float
            Number between 0 and 1 representing the % of
            characters to be altered during the perturbation
        ------------------------------------------------------

        Returns:
        ------------------------------------------------------
        Transformation from the textflint module
        ------------------------------------------------------
        """
        assert perturbation_type in ALLOWED_CHARACTER_LEVEL_PERTURBATIONS, f"'perturbation_type' must be on of the following: {', '.join(list(ALLOWED_CHARACTER_LEVEL_PERTURBATIONS.keys()))}"
        assert 0 < proportion_characters <= 1, "'proportion_characters' parameter must be a number in (0, 1]"

        self.perturbation = ALLOWED_CHARACTER_LEVEL_PERTURBATIONS[perturbation_type]
        self.proportion_characters = proportion_characters
        self.other_args = kwargs

    def apply(self, 
              sample: UTSample,
              text_field: str = "x") -> str:

        """
        Apply a character-level perturbation to a UTSample

        Args:
        -------------------------------------------------------
        sample: UTSample
            Universal Transformation sample from the textflint
            library. Usually the output of `text_to_utsample`

        text_field: str, default = "x"
            Name of the text field in the UTSample 
        -------------------------------------------------------

        Returns:
        -------------------------------------------------------
        perturbed_sample: str
            String representing the perturbed sample. If the
            sample couldn't be perturbed, it returns the 
            original sample
        
        perturbed: bool
            Whether the sample was perturbed or not. Mainly
            used to track the number of perturbed samples
            in the dataset
        -------------------------------------------------------
        """
        sample_text = sample.dump()[text_field]
        n_chars_to_change = max(int(len(sample_text) * self.proportion_characters), 1)
        perturbation = self.perturbation(
            trans_min = n_chars_to_change,
            trans_max = n_chars_to_change,
            trans_p = 1,
            min_char = 5,
            **self.other_args
        )

        perturbed_sample_list = perturbation.transform(sample)
        if not perturbed_sample_list:
            # List is empty because perturbation failed: return original sample
            return sample_text
        else:
            perturbed_sample_text = perturbed_sample_list[0].dump()[text_field]
            return perturbed_sample_text
            _aux_pert_s = re.sub(REMOVE_SPACES_PATTERN, r"\1", perturbed_sample_text)
            _aux_ori_s = re.sub(REMOVE_SPACES_PATTERN, r"\1", sample_text)

            if _aux_pert_s == _aux_ori_s:
                # Even though the sample was 'perturbed', it did not change
                # any of its contents, so we don't count it as perturbed
                
                return sample_text, False
            else:
                return perturbed_sample_text, True

class WordPerturbation:
    def __init__(self,
                 perturbation_type: str,
                 proportion_words: float,
                 **kwargs):

        """
        Perturbation that alters the words of a sentence

        Args:
        ------------------------------------------------------
        proportion_words: float
            Number between 0 and 1 representing the % of
            words to be altered during the perturbation
        ------------------------------------------------------

        Returns:
        ------------------------------------------------------
        Transformation from the textflint module
        ------------------------------------------------------
        """
        assert perturbation_type in ALLOWED_WORD_LEVEL_PERTURBATIONS, f"'perturbation_type' must be on of the following: {', '.join(list(ALLOWED_WORD_LEVEL_PERTURBATIONS.keys()))}"
        assert 0 < proportion_words <= 1, "'proportion_words' parameter must be a number in (0, 1]"

        self.perturbation = ALLOWED_WORD_LEVEL_PERTURBATIONS[perturbation_type]
        self.proportion_words = proportion_words
        self.other_args = kwargs

    def apply(self, 
              sample: UTSample,
              text_field: str = "x") -> str:

        """
        Apply a word-level perturbation to a UTSample

        Args:
        -------------------------------------------------------
        sample: UTSample
            Universal Transformation sample from the textflint
            library. Usually the output of `text_to_utsample`

        text_field: str, default = "x"
            Name of the text field in the UTSample 
        -------------------------------------------------------

        Returns:
        -------------------------------------------------------
        perturbed_sample: str
            String representing the perturbed sample. If the
            sample couldn't be perturbed, it returns the 
            original sample
        
        perturbed: bool
            Whether the sample was perturbed or not. Mainly
            used to track the number of perturbed samples
            in the dataset
        -------------------------------------------------------
        """

        sample_text = sample.dump()[text_field]

        _aux_sample_text = re.sub(r'[^\w\s]', ' ', sample_text)
        n_words_to_change = max(int(len(_aux_sample_text.split()) * self.proportion_words), 1)
        perturbation = self.perturbation(
            trans_min = n_words_to_change,
            trans_max = n_words_to_change,
            trans_p = 1,
            **self.other_args
        )

        perturbed_sample_list = perturbation.transform(sample)
        if not perturbed_sample_list:
            # List is empty because perturbation failed: return original sample
            return sample_text
        else:
            perturbed_sample_text = perturbed_sample_list[0].dump()[text_field]
            return perturbed_sample_text
            _aux_pert_s = re.sub(REMOVE_SPACES_PATTERN, r"\1", perturbed_sample_text)
            _aux_ori_s = re.sub(REMOVE_SPACES_PATTERN, r"\1", sample_text)

            if _aux_pert_s == _aux_ori_s:
                # Even though the sample was 'perturbed', it did not change
                # any of its contents, so we don't count it as perturbed
                
                return sample_text, False
            else:
                return perturbed_sample_text, True

class OtherPerturbation:
    def __init__(self, 
                 perturbation_type: str, 
                 **kwargs):
        
        assert perturbation_type in ALLOWED_OTHER_PERTURBATIONS, f"'perturbation_type' must be on of the following: {', '.join(list(ALLOWED_OTHER_PERTURBATIONS.keys()))}"

        # Different from char and word-level perturbations: this time
        # we instantiate the perturbation type instead of referencing
        # for later isntantiation in the `apply` method. We do this
        # because in char and word-level perturbations we need the abs.
        # nº of characters to change, but this is not necesarry in
        # these types of perturbations

        self.perturbation = ALLOWED_OTHER_PERTURBATIONS[perturbation_type](**kwargs)

    def apply(self, 
              sample: UTSample,
              text_field: str = "x") -> str:
        
        sample_text = sample.dump()[text_field]

        perturbed_sample_list = self.perturbation.transform(sample)
        if not perturbed_sample_list:
            # List is empty because perturbation failed: return original sample
            return sample_text
        else:
            perturbed_sample_text = perturbed_sample_list[0].dump()[text_field]
            return perturbed_sample_text
            _aux_pert_s = re.sub(REMOVE_SPACES_PATTERN, r"\1", perturbed_sample_text)
            _aux_ori_s = re.sub(REMOVE_SPACES_PATTERN, r"\1", sample_text)

            if _aux_pert_s == _aux_ori_s:
                # Even though the sample was 'perturbed', it did not change
                # any of its contents, so we don't count it as perturbed
                
                return sample_text
            else:
                return perturbed_sample_text


class PerturbedDataset(Dataset):
    def __init__(self,
                 dataset: Dataset):

        """
        Hugginface NLP dataset that allows for alteration of 
        the input sentences

        Args:
        ------------------------------------------------------
        dataset: datasets.Dataset
            Hugginface dataset loaded from the `datasets`
            library, with text as inputs
        ------------------------------------------------------

        Returns:
        ------------------------------------------------------
        PerturbedDataset
        ------------------------------------------------------
        """

        self.dataset = dataset

    def __repr__(self) -> str:
        return self.dataset.__repr__()
    
    def __getitem__(self, index: int):
        return self.dataset[index]
    
    def __setitem__(self, index: int, item: dict):
        assert isinstance(item, dict), "Item to assign must be a dictionary"
        self.dataset[index] = item
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)

    def copy(self) -> PerturbedDataset:
        return self.__class__(dataset = self.dataset)
    
    def to_textflint_dataset(self, 
        x_field: str = "x",
        y_field: str = "y") -> TextflintDataset:

        textflint_dataset = TextflintDataset(task = "UT")
        for sample in self.dataset:
            textflint_sample = UTSample(data = {
                "x" : sample[x_field],
                "y" : str(sample[y_field])
            })
            textflint_dataset.append(textflint_sample)

        return textflint_dataset
    
    def to_tf_dataset(self, **kwargs):
        return self.dataset.to_tf_dataset(**kwargs)
    
    def map(self, function, **kwargs) -> PerturbedDataset:
        return self.__class__(self.dataset.map(function, **kwargs))

    def perturb(self,
        perturbation: CharacterPerturbation | WordPerturbation | OtherPerturbation,
        x_fields: str | list = "x"):

        n_perturbed_samples = 0

        if isinstance(x_fields, str):
            x_fields = [x_fields]

        # Function to apply textflint transformations (perturb the samples)
        def _map_perturbation(sample, x_fields, perturbation):

            _to_return = {k: None for k in x_fields}

            for x_field in x_fields:
                # Create UTSample
                textflint_sample = text_to_utsample(sample[x_field])

                # Apply perturbation
                perturbed_text_sample = perturbation.apply(textflint_sample)

                # Check if sample was actually perturbed

                _to_return[x_field] = perturbed_text_sample
            
            return _to_return
        
        #Kinda silly, but when mapping the function, if any model for the
        #perturbation to work needs to be downloaded, working with GPU gives
        #funky outputs (like the model is being downloaded) from different
        #processes? Let's just apply the transformation to the first sample
        #and then to the whole dataset

        void = _map_perturbation(self.dataset[0], x_fields, perturbation)

        #Apply the transformations (this is only possible via the map method due
        #to huggingface datasets being immutable and can only be changed by map)
        pert_dataset = PerturbedDataset(
            dataset = self.dataset.map(
            lambda x: _map_perturbation(x, x_fields, perturbation)
            ))
        
        #Check if samples were actually perturbed:
        for i, sample in enumerate(pert_dataset):
            perturbed = True
            for field in x_fields:
                sample_text = self.dataset[i][field]
                perturbed_sample_text = sample[field]

                _aux_pert_s = re.sub(REMOVE_SPACES_PATTERN, r"\1", perturbed_sample_text)
                _aux_ori_s = re.sub(REMOVE_SPACES_PATTERN, r"\1", sample_text)

                if _aux_pert_s == _aux_ori_s:
                    perturbed = False
                    break
            if perturbed:
                n_perturbed_samples += 1

        return pert_dataset, n_perturbed_samples / len(pert_dataset)

    def to_processed_nlp_dataset(
        self, 
        checkpoint: str, 
        x_fields: str | list = "x",
        y_fields: str | list = "y",
        return_tensors: str = "tf",
        batched: bool = True,
        shuffle: bool = True,
        batch_size: int = 32):

        if isinstance(x_fields, str):
            x_fields = [x_fields]

        if isinstance(y_fields, str):
            y_fields = [y_fields]

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        def _tok_map(sample):
            fields = []
            for x_field in x_fields:
                fields.append(sample[x_field])
            return tokenizer(*fields, truncation = True)
        
        tokenized_dataset = self.dataset.map(_tok_map, batched = batched)
        data_collator = DataCollatorWithPadding(tokenizer = tokenizer, return_tensors = return_tensors)

        return tokenized_dataset.to_tf_dataset(
            columns = ["attention_mask", "input_ids", "token_type_ids"],
            label_cols = y_fields,
            shuffle = shuffle,
            collate_fn = data_collator,
            batch_size = batch_size
        )