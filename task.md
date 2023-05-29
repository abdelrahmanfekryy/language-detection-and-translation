![](RackMultipart20230525-1-lcx4cn_html_593c34941b3bb62.jpg)

##

## Intro

A blogger usually publishes short news articles, memes, and images with captions on his page, he wants to increase the number of blog readers from other countries by publishing the text in his posts in both Arabic and English languages.

## Task

Develop a complete ML solution that receives a sentence in a request body through API endpoints:

- API-1: Detect sentence language
- API-2 Translate the sentence to the other language

Sample scrapped data are attached with the email, these represent the use cases which the server will usually be used for. Feel free to use the data as you see fit (training, fine-tunning, testing, neglect it).

#### API-1: Model-1: Language detection

- Model has to be light weight and fast (less than 0.5sec in prediction)
- Model architecture, backbone, and implementation are up to your choice.
- Support as many languages as you see fit but at minimum Arabic, English.
- API Response is just the language of the sentence

#### API-2: Model-2: Translation model

- Model architecture, backbone, and implementation are up to your choice.
- Supports translating from Arabic to English, and from English to Arabic, through any kind of implementation which can produce such results.
- The input to the API is just the sentence.
- The API return the sentence in both languages Arabic and English

## Evaluation Criteria:

#### Minimum Requirements:

- Include libraries and pre-requisite requirements for running the code
- Code runs without syntax errors or exceptions.
- Code includes a server which runs on local host.
- Local python server can accept requests, and includes at least 2 APIs endpoints for language detection & translation.
- Choose the metric to measure your model performance with very brief comment why did you choose it. Model has to perform better than baseline model with random output, anything better is acceptable.
- Server accepts the following test cases and responds with similar results (doesn't have to be exactly the same):

| **Sentence** | **API-1 (Language)** | **API-2 (Translation)** |
| --- | --- | --- |
| انا لا اشعر بالعطش | Arabic | I'm not thirsty |
| This castle is amazing | English | هذه القلعة مدهشة |

#### Bonus points

- Code is readable and understandable with comments, good naming conventions, …
- Code is structructed logically and uses OOP concepts where it's needed
- APIs can handle edge cases through checks and error handling. Ex:
  - Sentence: "42"
  - Sentence: "?!!!!!!?
- Language detection model can classify more than the minimum languages. Extra points for non-Latin languages
- Translation model responds with "Not-supported" if the input sentence is not Arabic or English.

## Submission

Your submission could be submitted though any of the following

- ZIP file all your code and model files and resend it through mail
- ZIP file all your code and model files and upload it to a drive (google drive, onedrive,…), and send the link through email (make sure link is shared)
- Upload the code and all files to a github repo and send the repo link in an email
- Finish all the code in a co-lab notebook and send the link through email

##

## Duration

Available duration for the task is 3 days starting from the email receiving time.