# AI_Classical_Music_Composer

This project aims to generate Classical Music using generative models – Bi-LSTM and CNN-GAN to compose Classical Music for some particular Classical Music genres and evaluate their performance respectively and collectively. Also, to further explore and strengthen the area of Artificial Intelligence in music composition.

- The **dataset** is obtained from https://github.com/bytedance/GiantMIDI-Piano.
- The **four musical eras** being trained and tested are *Baroque, Classical, Romantic, and Modernist eras*.
- **Evaluation metrics**: *Pitch histogram, FID score, nearest neighbor, and survey*.
- It is recommended to use the **https://musescore.org/en** software to play the midi files. It is a free music sheet editing tool that can visualize music in music sheet forms and also convert midi files to mp3 format.

## Demonstration
A video demonstrating the system can be accessed [here](https://1drv.ms/v/s!AhjIIVcsidoXalRgJqUKU2J2qr8?e=FGc3bp)

## Instruction

1.  Prepare your own midi files or use the provided midi files in the `midi_preprocess/midi/` folder.
2.  Convert the original midi files into trainable datasets. (Follow the steps in `midi_preprocess/midPreprocess.ipynb`)
3.  Copy your processed datasets from either `midi_preprocess/data_biLstm/` or `midi_preprocess/data_gan/` to either `Bi-LSTM/notes/` or `GAN/notes/`. Notice that some datasets are provided as examples. you may also use these for the later training processes.

### Bi-LSTM training & generation

Detailed instructions are provided in the `Bi-LSTM` folder.

### GAN training & generation

Detailed instructions are provided in the `GAN` folder.

## Conclusion

Among the four Classical Music eras, I found the Bi-LSTM model suitable for generating Baroque, Classical, and some Romantic Music owing to the design of the Bi-LSTM model and the nature of these musical styles. These types of music require strict rules and are less complexed than the Modernist music. The design of LSTMs works just well for doing these tasks. On the other hand, the GAN model seems to be more creative and stochastic in generating music. I therefore consider it suitable for generating Modernist Music.
