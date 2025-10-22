# Create prompt–completion pairs with OpenAI API
This repository processes cleaned song lyrics and sends them to the OpenAI API to generate concise lyric summaries, which are used to construct prompt–completion pairs for future model fine-tuning.

The dataset comprises 10,681 song lyrics. Summaries were generated using the GPT-5-Nano model at a total cost of $3.44.

Example of prompt–completion pairs:

```json
{
  "prompt": "Write a rap song in year 2009's Eminem style. The topic is about:...",
  "completion": "title \n\n lyrics"
}
```



The lyrics file is [here](https://drive.google.com/file/d/1PqADJhbqqTgyEAXKltgPe6Q0x_zQGLqA/view?usp=drive_link) .
The prompt–completion pairs file is [here](https://drive.google.com/file/d/1Y8ADxBk0wa-lU00q42TmYFyEg4qqC90K/view?usp=drive_link) .
