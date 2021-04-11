Please change to the directory `vae` first.

Training:
```
python vae.py
```

Evaluation:
```
python vae.py --eval --load_path checkpoints/best.pt 
```

---

Self-check before submitting (please ignore this if you are the TA):
```sh
python vae.py --grade # estimate the classification accuracy
python vae.py --eval --load_path checkpoints/best.pt # change this path
python grade.py
```