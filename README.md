# Technology Stockbot Chat
The purpose of this project is to create a chatbot that converses about technology stocks using a various number of tools. The project is split into four stages with indepth explanation of each stage. 

Stage 1: Development of a Rule-Based and Similarity-Based Chatbot:

- Implemented rules and incorporated a file with pre-defined Q/A pairs in an appropriate format (e.g., plain text or CSV).
- Maintained a conversation log and produced design documentation.
- While using Python and AIML files as foundational resources, these files were expanded and customized according to the design specification.
- The similarity-driven component was designed using the bag-of-words model, tf/idf, and cosine similarity.

Stage 2: Integration of an Image Classification Component:

- Updated previous stage's programs, files, and documentation.
- Introduced a new feature in the Python program to allow the chatbot to call upon a pre-trained convolutional neural network when presented with image-related queries.
- The neural network's architecture and dataset were personally chosen, with potential object classifications dependent on the chatbot's domain.
- Employed the Keras library to develop the convolutional neural network and recommended tools like Scikit-learn, Keras, and Tensorflow datasets (tfds) APIs for data management and preprocessing.
- Ensured the model could be saved post-training for later classifications during chatbot interactions.

Stage 3: Inclusion of a Basic Logical Knowledgebase Component:

- Updated all programs, files, and documentation from Stage 2, while enhancing the AIML file and introducing a sample knowledgebase file in CSV format.
- Developed a basic first order logic (FOL) knowledgebase and inference engine using the NLTK library.
- The system was designed to interpret user inputs following patterns like “I know that ... is ...” and “Check that ... is ...”.
- Created a CSV file containing initial statements related to the chatbot's theme, structured in NLTK’s FOL syntax.
- Integrated logic to handle user assertions and queries, managing potential contradictions and utilizing NLTK’s resolution algorithm.

Stage 4: Addition of a Cloud-Based Multi-Lingual Component:

- Enhanced and updated the systems from Stage 3.
- Incorporated an automatic multi-lingual feature, enabling the chatbot to detect, translate, and respond in the user's language.
- Utilized NLP services provided by Microsoft Azure cloud to achieve the multilingual capability.
- Demonstrated the chatbot's ability through conversation logs in multiple non-English languages, adjusting AIML/CSV files as necessary to optimize the multi-lingual functionality.
- Ensured the chatbot's potential support for numerous languages offered by Azure AI services.
