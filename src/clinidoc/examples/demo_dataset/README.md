# Synthetic demo dataset

All notes are fake. Identifiers are invented (including the well-known sample SSN `078-05-1120`). Do not treat this folder as clinical data.

Planted defects:

- empty note (`n003`)
- patient `P1` in train and val (`n001` / `v001`) plus val timestamp before train
- exact duplicate discharge template (`n004` / `n005`)
- bad NER offset (`n006`)
- thin class `rare_fever` (`n007`) absent from val
- fake SSN in `n006`
