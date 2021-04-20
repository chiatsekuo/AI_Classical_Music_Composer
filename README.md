# AI_Classical_Music_Composer

This project aims to generate Classical Music using generative models – Bi-LSTM and CNN-GAN to compose Classical Music for some particular Classical Music genres and evaluate their performance respectively and collectively. Also, to further explore and strengthen the area of Artificial Intelligence in music composition.

The four musical eras being trained and tested are Baroque, Classical, Romantic, and Modernist eras.

Evaluation metrics: Pitch histogram, FID score, nearest neighbor, and survey.

## Conclusion

Among the four Classical Music eras, I found the Bi-LSTM model suitable for generating Baroque, Classical, and some Romantic Music owing to the design of the Bi-LSTM model and the nature of these musical styles. These types of music require strict rules and are less complexed than the Modernist music. The design of LSTMs works just well for doing these tasks. On the other hand, the GAN model seems to be more creative and stochastic in generating music. I therefore consider it suitable for generating Modernist Music.
