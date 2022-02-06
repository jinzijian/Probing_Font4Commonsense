python main.py --epochs 30 --gpu 0 --models mbert_cnn --language_code zh --language_index 14 --mode character --alpha 1
python main.py --epochs 30 --gpu 0 --models mbert_base --language_code zh --language_index 14 --mode character --alpha 1
python main.py --epochs 30 --gpu 2 --models mroberta_base --language_code zh --language_index 14 --mode character --alpha 4
python main.py --epochs 30 --gpu 2 --models mroberta_cnn --language_code zh --language_index 14 --mode character --alpha 4