# Open-Set Speech Language Identification and the CU MultiLang Dataset

#### Mustafa Eyceoz, Justin Lee, Siddarth Pittie


Run demo:
 - `python3 full_system.py`

Train TDNN with 5hrs, 4sec, 15 epochs
 - `python3 train_tdnn.py 5 4 15 0.8`

Test the tdnn-final-submission TDNN model
 - `python3 test_tdnn.py ./saved-models/tdnn-final-submission.pickle`

Train lda and plda layers using saved-tdnn-outputs
 - `python3 train_lda_plda.py ./saved-tdnn-outputs`
