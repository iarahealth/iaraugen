# iaraugen
Data augmentation/generation utilities for Iara.

Can be used standalone (instructions below) or as part of other programs.

## Offline text augmentation
```
help: ./txt_aug.py -h

example usage:
./run_aug.py corpus_1br_10pt_15sept.tok --aug translate random --action delete --maxs 10 --lang en --translate_mode local --append --output out.txt
```

## Offline text generation
```
help: ./txt_gen.py -h

example usage:
./run_gen.py --input_file palavras.txt --context "radiologia médica" --num 2 --return_type "frases" --api_key "YOUR_OPENAI_API_KEY" --output query.txt
```
