# Evaluation for Machine Translation

## BLEU Score

### What is BLEU Score?

1. The BLEU Score *(Bilingual Evaluation Understudy)* evaluates the quality of machine-translated text by comparing a candidate texts translation to one or more reference translations. <br><br>

2.  The closer the BLEU score is to one, the better your model is. The closer to zero, the worse it is.

3.To get a BLEU score, the candidates and the references are usually based on an average of uni, bi, try or even four-gram precision.<br> 
<img src="./images/28. BLEU Score Calculation.png" width="40%"></img><br><br>

### Disadvantages of BLEU

1. BLEU doesn't consider the semantic meaning of words.
2. BLUE doesn't consider the sentence structure of the sentence.

## ROGUE Score

### What is ROGUE Score?

1. ROGUE stands for *Recall Oriented Understudy for Gisting Evaluation*, which is a mouthful. But let's you know right off the bat that it's more recall-oriented by default. This means that it's placing more importance on how much of the human created reference appears in the machine translation.
2. It was orignially developed to evaluate text-summarization models but works well for NMT as well.
3. ROGUE score calculates the precision and recall between the generated text and human-created text.
<img src="./images/29. ROUGE Score Calculation.png" width="40%"></img><br><br>

### Disadvantages of ROUGE

1. Low ROUGUE score doesn't necessarily mean that translation is bad. 
