# DL_DeepBlue_a3_YDS
Yelp Dataset Challenge

### TODO:
###### Feature suggestion and dev procedure:
- Feel free to add features you think need to be added.
- If you start working on one of the features, add your initials.
- Dev in branches if your work might temporarily break the code.

###### Features TODO:
- Split data into training/validation and test sets to do all the model comparisons
  on the training/validation, and then run best models on test. Write test data
  and training/validation data to separate files.
- Write code to save trained models to file.
- Fix or exmplain why minibatch size greater than 1 breaks `linear_model()`
- Add confusion matrix
- Add/clarify training / test error messages in output
- Enable CUDA
- Automatically output log entries in `result_notes.md` log format
- Done (PV): Set seed for permutation and training/test set generation


###### Models TODO:
- **PV:** tfidf BOW
- concatenated words convolutional
- **PB:** character-level convolutional