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
```
python vae.py --grade
python vae.py --eval --load_path checkpoints/best.pt 
python grade.py
```