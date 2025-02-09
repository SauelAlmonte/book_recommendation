# Book Recommendation System

## Overview
The **Book Recommendation System** is an intelligent system built using Python, OpenAI, LangChain, and Gradio. It provides personalized book recommendations based on semantic content, allowing users to discover books similar to their preferences. The system uses a large language model (LLM) for text classification, sentiment analysis, and semantic search, making the recommendations more meaningful and relevant.

## Purpose of this Project
The main reason for this project is to illustrate the application of large language models (LLMs) and semantic search techniques in building a personalized book recommendation system. By leveraging OpenAI's models, LangChain, and Gradio, this project demonstrates how to develop an interactive and intelligent recommendation engine, which shows how modern machine learning frameworks and APIs can be used to create practical applications in natural language processing. The system helps users find books based on content similarity, sentiment, and personal preferences.
## Technologies Used
- **Python**: The primary programming language used for the system.
- **OpenAI API**: Utilized for text classification and natural language understanding.
- **LangChain**: A framework used to manage text splitting and vector databases.
- **Gradio**: Provides an interactive web interface for users to interact with the system.
- **GitHub**: For version control and collaboration.

## Dependencies
- **kagglehub**: For accessing Kaggle datasets.
- **pandas**: For data manipulation and analysis.
- **seaborn**: For statistical data visualization (requires `matplotlib` for plotting).
- **matplotlib**: Visualization library for creating static, animated, and interactive plots.
- **python-dotenv**: Used for loading environment variables, including the OpenAI API key.
- **langchain-community**: The main framework for creating LLM-based applications.
- **langchain-openai**: Specifically works with OpenAI models.
- **langchain-chroma**: Used for working with a Chroma database.
- **transformers**: A powerful library from Hugging Face for working with large open-source LLMs.
- **gradio**: Framework for building interactive dashboards that can interface with the system.
- **notebook**: Jupyter notebook support for working with notebooks.
- **ipywidgets**: Interactive HTML widgets for working with notebooks.
- **torch**: PyTorch library for deep learning.
- **tensorflow**: TensorFlow library for machine learning and deep learning models.
- **flax**: A flexible machine learning framework built on top of JAX.
- **tf-keras**: Keras API, running on top of TensorFlow for deep learning.


## Features
- **Book Recommendations**: Based on book descriptions and user preferences.
- **Semantic Search**: Uses vector search to find similar books based on their content.
- **Zero-Shot Text Classification**: Classifies book descriptions to enhance recommendation accuracy.
- **Sentiment Analysis**: Analyzes emotions in book descriptions to improve recommendation relevance.
- **Interactive UI**: Gradio provides a user-friendly web interface for easy interaction with the system.

## Setup

### Requirements
- Python 3.x
- OpenAI API key (Sign up on OpenAI to get your API key)
- LangChain library
- Gradio library

## License
- This project is licensed under the MIT License - see the LICENSE file for details.