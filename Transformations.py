from __future__ import annotations

#TRANSFORMATIONS
from textflint.generation.transformation.UT.keyboard import Keyboard
from textflint.generation.transformation.UT.prejudice import Prejudice
from textflint.generation.transformation.UT.contraction import Contraction
from textflint.generation.transformation.UT.insert_adv import InsertAdv

#VALIDATORS
from textflint.generation.validator.sentence_encoding import SentenceEncoding

#DATASETS & SAMPLES
from textflint.input.dataset.dataset import Dataset as TextflintDataset
from textflint.input.component.sample.sa_sample import SASample
from textflint.input.component.sample.ut_sample import UTSample

from datasets import Dataset

#MISCELANEOUS
import numpy as np

ALLOWED_PERTURBATIONS = {
    "keyboard" : Keyboard,
    "prejudice" : Prejudice,
    "contraction" : Contraction,
    "insertadv" : InsertAdv
}

class Perturbation:
    def __init__(
        self, 
        perturbation: str, 
        perturbation_args: dict or None = None):

        assert perturbation in ALLOWED_PERTURBATIONS, "Perturbation must be one of the following: " +  ", ".join(ALLOWED_PERTURBATIONS.keys())

        self.perturbation_name = perturbation
        self.perturbation_args = perturbation_args

        if perturbation_args is None: #Default kwargs
            self.perturbation = ALLOWED_PERTURBATIONS[perturbation]()
        else:
            self.perturbation = ALLOWED_PERTURBATIONS[perturbation](**perturbation_args)

    def __repr__(self):
        cad = f"{self.perturbation_name} perturbation:\n\n"
        cad += "Args:"
        for arg, arg_value in self.perturbation_args.items():
            cad += f"\n\t{arg} = {arg_value}"

        return cad

class ExtendedDataset:
    def __init__(
        self, 
        dataset: Dataset, 
        perturbed: bool = False):

        self.dataset = dataset
        self.perturbed = perturbed

    def __repr__(self):
        return self.dataset.__repr__()
    
    def __getitem__(self, index: int):
        return self.dataset[index]
    
    def __setitem__(self, index: int, item: dict):
        assert isinstance(item, dict), "Item to assign must be a dictionary"
        self.dataset[index] = item
    
    def __len__(self):
        return len(self.dataset)

    def copy(self):
        return ExtendedDataset(dataset = self.dataset, perturbed = self.perturbed)

    def to_textflint_dataset(
        self, 
        x_field: str = "x",
        y_field: str = "label") -> TextflintDataset:

        textflint_dataset = TextflintDataset(task = "UT")
        for sample in self.dataset:
            textflint_sample = UTSample(data = {
                "x" : sample[x_field],
                "y" : str(sample[y_field])
            })
            textflint_dataset.append(textflint_sample)

        return textflint_dataset

    def map(self, function, **kwargs):
        return ExtendedDataset(self.dataset.map(function, **kwargs), perturbed = self.perturbed)

    def to_tf_dataset(self, **kwargs):
        return self.dataset.to_tf_dataset(**kwargs)

    def perturb(
        self,
        perturbation: Perturbation,
        x_fields: list,
        y_field: str,
        validation: bool = False,
        batched: bool = True,
        batch_size: str or int = "infer",
        inplace: bool = True) -> None or ExtendedDataset:

        if self.perturbed and inplace:
            print("Warning! Dataset is already perturbed and inplace is set to True, so the dataset will be again perturbed")

        if batched:
            raise Exception("For some god forsaken reason textflint does not work well with batched, so it is disabled for now")

        if batch_size == "infer":
            batch_size = int(len(self.dataset) * 0.02) #Batches of 2% the whole dataset

        #Function to apply textflint transformations (perturb the samples)
        def _pert_sample(sample, x_fields, perturbation):

            to_return = {k : None for k in x_fields}

            for x_field in x_fields:

                #Transformation
                textflint_sample = UTSample(data = {"x" : sample[x_field]})
                perturbed_sample = perturbation.perturbation.transform(textflint_sample)[0].dump()

                #Validation
                if validation:
                    ds1 = TextflintDataset()
                    ds1.append(textflint_sample, sample_id = 0)
                    ds2 = TextflintDataset()
                    ds2.append(perturbed_sample, sample_id = 0)

                    validation_score = SentenceEncoding(ds1, ds2, "x").score
                    to_return["validation_score"] = validation_score

                to_return[x_field] = perturbed_sample["x"]

            return to_return
        
        #Apply the transformations (this is only possible via the map method due
        #to huggingface datasets being immutable and can only be changed by map)
        ext_dataset = self.dataset.map(
            lambda x: _pert_sample(x, x_fields, perturbation), 
            batched = batched, 
            batch_size = batch_size
        )

        if not inplace:
            return ExtendedDataset(ext_dataset, perturbed = False)
        else:
            self.dataset = ext_dataset
            self.perturbed = True