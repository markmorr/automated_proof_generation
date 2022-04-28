# automated_proof_generation
Project to train a deep learning model to write proofs for graduate level machine learning problems.


## Full_pipeline.py
 is the most important script for our final approach, it contains all the code to read in the final curated dataset and to chain the outputs of the translation model to the proof generation model. ML_Flag is one variable to note--it signals whether we are attempting to generate proofs to solve the actual course problems, in which case we make use of the full MiniF2F dataset as our support set (rather than splitting it internally into query and support sets). This means we have to handle a few things differently, so the flag variable crops up a few times.

 ## datasets (within src)
 This dataset contains our pre-processed and curated problems from the MiniF2F dataset, as well as the Machine Learning course 4771 at Columbia. We didn't include the full dataset, nor the NaturalProofsWiki dataset-- please note that these datasets are available publicly online.

### Regular_expression_for_dataset_merging_and_curation.py
 is as the name suggests--a useful file for searching through the MiniF2F dataset and collecting the files we need, which was crucial for linking the lean statement and proof to its English expression.

### coq_gym folder
 represents some of our first steps, and is essentially defunct because we abandoned that approach early on in favor of leveraging pre-trained models. Please note that the vast majority of code in the folder coq_gym was downloaded from https://github.com/princeton-vl/CoqGym/tree/master/ASTactic and has only been adapted for our specific experiments.

### Loading_in_json.py
 was useful for curating the natural_proofs_wiki dataset, which we applied to GPT-3 in our first attempt at this problem. GPT-3 did poorly (the proofs were nonsense) but we kept this function as an important utility in case we decide to make use of this dataset in the future.
### Reorganize_dataset.p
y was useful for the natural_proofs_dataset, in particular performing pre-processing steps to handle the issue of unwanted newlines and escape characters, which were causing problems downstream.
### Converting_course_text
 is a simple application of a pdf-reader that we attempted to use to extract text from online textbook pdfs. The initial results were quite poor, because of images/figures/other complexities, and we moved on from a fine-tuning GPT-3 approach to a few-shot approach with Codex anyway, so it became irrelevant. We kept some of the code anyway because it was an interesting line of approach that we figured we might want to return to in the future.
