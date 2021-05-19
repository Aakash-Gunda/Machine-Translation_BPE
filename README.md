# Machine-Translation_BPE
In this notebook, we aim build a deep neural network that tries to enhance the language translation process by introducing a sub-word tokenization algorithm called byte pair encoding. Final pipeline will accept English text as input and return the French translation.

# Setup
This project requires GPU acceleration to run efficiently. You can use GPU from either your personal system's GPU or GPU provided by Google Colab.

# Technologies used:
   #### TensorFlow: 
   TensorFlow is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

  #### Collaboratory: 
  Free Jupyter notebook environment that requires no setup and runs entirely in the cloud. It Provides GPU which is required for this project.

  #### Keras: 
  Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation.
  
# Neural Machine translation:

Neural Machine translation is the use of neural network models to learn a statistical model for machine translation. The key benefit to the approach is that a single system can be trained directly on source and target text. Multilayer Perceptron neural network models can be used for machine translation, although the models are limited by a fixed-length input sequence where the output must be the same length. These early models have been greatly improved upon recently using recurrent neural networks organized into an encoder-decoder architecture that allow for variable length input and output sequences. Key to the encoder-decoder architecture is the ability of the model to encode the source text into an internal fixed-length representation called the context vector. Interestingly, once encoded, different decoding systems could be used, in principle, to translate the context into different languages. The encoder-decoder recurrent neural network architecture with attention is currently the state-of-the-art on some benchmark problems for machine translation.

For a neural network to predict text data, it must first be converted into data that it can understand. The input data for a neural network must be numbers since it is a sequence of multiplication and addition operations. Hence, we won’t use text data as input, instead will convert the text into sequences of integers using tokenization. Now after converting text data into a sequence of integers, we need to pad the sequence of integers with zeros to make the length of each sentence equal which is known as padding. After padding, we use Recurrent Neural Network models to convent English data to French data. The output of the RNN model is again ids which we will convert back to text using the tokenizer function. In this project we are using word tokenization from the Keras library and a subword tokenization method called  byte pair encoding and comparing the accuracies of both tokenization methods by implementing them using an RNN model.

