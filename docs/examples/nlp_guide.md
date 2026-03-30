# Extended Experiment Analysis

To better understand model behavior, we compare all experiments across three axes: model architecture (RNN, LSTM, GRU), task (machine translation vs. text generation), and input representation (one-hot vs. pretrained embeddings). All results are taken from the final epoch after 5 training iterations.
## Machine Translation (Multi30k)

| Model | Encoding   | Val Loss | Perplexity |   BLEU    |
| :---- | :--------- | :------: | :--------: | :-------: |
| GRU   | One-Hot    | **1.16** |  **3.20**  | **89.81** |
| GRU   | Pretrained |   1.15   |    3.17    |   89.64   |
| LSTM  | One-Hot    |   1.27   |    3.56    |   89.17   |
| LSTM  | Pretrained |   1.28   |    3.61    |   89.08   |
| RNN   | One-Hot    |   1.31   |    3.72    |   88.95   |
| RNN   | Pretrained |   1.31   |    3.70    |   88.93   |

For machine translation, **BLEU score is the primary metric**, as it directly measures how closely generated translations match human references. Loss and perplexity are still useful for optimization insight, but BLEU provides the most interpretable measure of real-world translation quality.

Across all configurations, performance is consistently high, with BLEU scores near 89 for every model. The GRU achieves the strongest results, slightly outperforming both LSTM and the vanilla RNN in loss, perplexity, and BLEU. This suggests that the GRU strikes an effective balance between expressive power and training efficiency.

Interestingly, pretrained embeddings provide almost no improvement in this task. The differences between one-hot and pretrained variants are negligible, indicating that the model is able to learn effective word representations directly from the parallel translation data. This is expected in supervised settings like translation, where strong input-output alignment provides a rich learning signal.
## Text Generation (WikiText2)

| Model | Encoding   | Val Loss | Perplexity |
| :---- | :--------- | :------: | :--------: |
| GRU   | Pretrained | **4.77** | **120.39** |
| RNN   | Pretrained |   4.90   |   136.89   |
| LSTM  | Pretrained |   5.02   |   154.39   |
| RNN   | One-Hot    |   4.92   |   139.31   |
| GRU   | One-Hot    |   5.06   |   160.94   |
| LSTM  | One-Hot    |   5.07   |   162.01   |

For text generation, perplexity is the most meaningful metric, as it reflects how well the model predicts the next token in a sequence. BLEU is not particularly informative in this setting because there are many valid ways to continue a sentence, making exact n-gram overlap a poor measure of quality.

Unlike machine translation, performance varies more significantly across configurations. The GRU with pretrained embeddings achieves the lowest perplexity, indicating the best predictive performance overall. In general, pretrained embeddings lead to substantial improvements, reducing perplexity by a large margin compared to one-hot representations.

The vanilla RNN performs surprisingly well, even outperforming the LSTM in this setup. This likely reflects the relatively short training duration (5 epochs) and the nature of the dataset, where shorter-range dependencies dominate. More complex architectures like LSTM often require longer training to fully realize their advantages.
## Embedding Strategy Comparison

The effectiveness of the embedding strategy depends strongly on the task.

In machine translation, one-hot and pretrained embeddings perform nearly identically. This suggests that when sufficient supervised data is available, models can learn meaningful representations during training without needing external initialization.

In contrast, text generation benefits significantly from pretrained embeddings. Because the task lacks explicit supervision and must learn language structure from raw sequences, starting from semantically meaningful word vectors provides a major advantage. This results in both better performance and faster convergence.
## Runtime and Efficiency

Training time varies considerably across both model architecture and embedding strategy.

|Model|Task|One-Hot Train Time (s)|Pretrained Train Time (s)|
|:--|:--|:-:|:-:|
|GRU|MT|801.6|524.9|
|LSTM|MT|917.8|523.8|
|RNN|MT|609.8|489.8|
|GRU|Text|251.2|104.8|
|LSTM|Text|269.4|109.8|
|RNN|Text|189.0|102.6|

Pretrained embeddings consistently reduce training time, in some cases by nearly half. This is primarily due to the reduced dimensionality of the input representation compared to one-hot encoding, which significantly lowers the cost of matrix operations in the model.

Among the architectures, LSTMs are the most computationally expensive due to their more complex gating mechanisms, while RNNs are the fastest but least expressive. GRUs fall in between, offering a strong balance of efficiency and performance.
## Final Discussion

These experiments highlight how model choice, representation, and task interact in practical NLP systems.

For machine translation, all models perform well, and the differences between architectures and embedding strategies are relatively small. The GRU provides the best overall performance, but even the simplest RNN achieves competitive results. This suggests that the task is well-structured and benefits from strong supervision.

For text generation, the situation is more nuanced. Performance varies more widely, and both model architecture and embedding strategy play a larger role. Pretrained embeddings are especially important, and the GRU again emerges as a strong overall choice.

More broadly, the results reinforce an important idea in NLP: model performance depends not just on architecture, but on how well the approach matches the structure of the task. Choosing appropriate metrics is also critical—BLEU for translation and perplexity for language modeling—since different tasks require different ways of measuring success.