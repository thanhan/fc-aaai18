# Fact-checking from crowds

Paper:
```
An Interpretable Joint Graphical Model for Fact-Checking from Crowds
Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)
```
Demo: http://fcweb.pythonanywhere.com/#

The Emergent dataset and the text features extraction for stance classification are from
```
Ferreira, W., & Vlachos, A. (2016). Emergent: a novel data-set for stance classification.
In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: Human language technologies (pp. 1163-1168).
```

To run offline/online experiments:
```
python run_experiment.py offline
```
or
```
python run_experiment.py online
```

Mturk directory contains stance labels collected by Amazon Mechanical Turk. The worker IDs have been mapped to another set of IDs. 
