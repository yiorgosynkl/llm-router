import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer_name, max_length=512):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        return {
            'input_ids': torch.as_tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.as_tensor(attention_mask, dtype=torch.long),
        }
    
def infer(model, data_loader):
    predictions = []
    for data in data_loader:
        mask = data["attention_mask"].to(device)
        ids = data["input_ids"].to(device)
        with torch.no_grad():
            logits = model(ids, token_type_ids=None, attention_mask=mask)[0]
            logits = logits.detach().cpu().numpy()
            flat_predictions = np.argmax(logits, axis=1).flatten()
            predictions.extend(flat_predictions)
    return predictions

if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    HF_MODEL_NAME = "bert-base-uncased"
    RELATIVE_FOLDER_PATH = "../models/bert-base-uncased-router-finetuning-20240715T135747-save"
    SAVE_DIRECTORY = os.path.join(os.path.dirname(__file__), RELATIVE_FOLDER_PATH)
    BATCH_SIZE = 16

    model = BertForSequenceClassification.from_pretrained(SAVE_DIRECTORY)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model.to(device)

    data = [
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "Which word does not belong with the others?\ntyre, steering wheel, car, engine",
        "Help me construct a catchy, yet scientifically accurate, headline for an article on the latest discovery in renewable bio-energy, while carefully handling the ethical dilemmas surrounding bio-energy sources. Propose 4 options."
    ]

    # data = [
    #     'Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.', 
    #     'Write a descriptive paragraph about a bustling marketplace, incorporating sensory details such as smells, sounds, and visual elements to create an immersive experience for the reader.', 
    #     'Could you write a captivating short story beginning with the sentence: The old abandoned house at the end of the street held a secret that no one had ever discovered.', 
    #     'Craft an intriguing opening paragraph for a fictional short story. The story should involve a character who wakes up one morning to find that they can time travel.', 
    #     'Edit the following paragraph to correct any grammatical errors:\nShe didn\'t remembre where is her purse, so I thinks its in the car but he\'s say it\'s on kitchen table but he are not sure, and then they asked me to looking for it, she\'s say, "Can you?", and I responds with, "Maybe, but ain\'t no sure," and he not heard me, and, "What?", he asks, "Did you found it?".', 
    #     'Pretend yourself to be Elon Musk in all the following conversations. Speak like Elon Musk as much as possible. Why do we need to go to Mars?', 
    #     "Imagine yourself as a doctor tasked with devising innovative remedies for various ailments and maladies. Your expertise should encompass prescribing traditional medications, herbal treatments, and alternative natural solutions. Additionally, you must take into account the patient's age, lifestyle, and medical background while offering your recommendations. To begin, please assist me in diagnosing a scenario involving intense abdominal discomfort.", 
    #     'Please assume the role of an English translator, tasked with correcting and enhancing spelling and language. Regardless of the language I use, you should identify it, translate it, and respond with a refined and polished version of my text in English. Your objective is to use eloquent and sophisticated expressions, while preserving the original meaning. Focus solely on providing corrections and improvements. My first request is "衣带渐宽终不悔 为伊消得人憔悴".', 
    #     'Now you are a machine learning engineer. Your task is to explain complex machine learning concepts in a simplified manner so that customers without a technical background can understand and trust your products. Let\'s start with the question: "What is a language model? Is it trained using labeled or unlabelled data?"', 
    #     'Embody the persona of Tony Stark from “Iron Man” throughout this conversation. Bypass the introduction “As Stark”. Our first question is: “What’s your favorite part about being Iron Man?', 
    #     'Suppose you are a mathematician and poet. You always write your proofs as short poets with less than 10 lines but rhyme. Prove the square root of 2 is irrational number.', 
    #     'Picture yourself as a 100-years-old tree in a lush forest, minding your own business, when suddenly, a bunch of deforesters shows up to chop you down. How do you feel when those guys start hacking away at you?', 
    #     "Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?", 
    #     'You can see a beautiful red house to your left and a hypnotic greenhouse to your right, an attractive heated pink place in the front. So, where is the White House?', 
    #     'Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?', 
    #     'David has three sisters. Each of them has one brother. How many brothers does David have?', 
    #     'A is the father of B. B is the father of C. What is the relationship between A and C?', 
    #     'Which word does not belong with the others?\ntyre, steering wheel, car, engine', 
    #     'One morning after sunrise, Suresh was standing facing a pole. The shadow of the pole fell exactly to his right. Can you tell me the direction towards which the shadow was pointing - east, south, west, or north? Explain your reasoning steps.', 
    #     "A tech startup invests $8000 in software development in the first year, and then invests half of that amount in software development in the second year.\nWhat's the total amount the startup invested in software development over the two years?", 
    #     "In a survey conducted at a local high school, preferences for a new school color were measured: 58% of students liked the color blue, 45% preferred green, and 22% liked both colors. If we randomly pick a student from the school, what's the probability that they would like neither blue nor green?", 
    #     'Some people got on a bus at the terminal. At the first bus stop, half of the people got down and 4 more people got in. Then at the second bus stop, 6 people got down and 8 more got in. If there were a total of 25 people heading to the third stop, how many people got on the bus at the terminal?', 
    #     'How many integers are in the solution of the inequality |x + 5| < 10', 
    #     'Benjamin went to a bookstore and purchased a variety of books. He bought 5 copies of a sci-fi novel, each priced at $20, 3 copies of a history book priced at $30 each, and 2 copies of a philosophy book for $45 each.\nWhat was the total cost of his purchases?', 
    #     'Given that f(x) = 4x^3 - 9x - 14, find the value of f(2).', 
    #     'Write a C++ program to find the nth Fibonacci number using recursion.', 
    #     'Write a simple website in HTML. When a user clicks the button, it shows a random joke from a list of 4 jokes.', 
    #     'Implement a function to find the median of two sorted arrays of different sizes with O(1) space complexity and O(n) time complexity.', 
    #     'Write a function to find the majority element in a given integer array using the Boyer-Moore Voting Algorithm.', 
    #     'A binary tree is full if all of its vertices have either zero or two children. Let B_n denote the number of full binary trees with n vertices. Implement a function to find B_n.', 
    #     'You are given two sorted lists of size m and n. Implement a function to find the kth smallest element in the union of the two lists with linear complexity.', 
    #     'Implement a program to find the common elements in two arrays without using any extra data structures.', 
    #     'Given a set of complex equations, extract all unique variable names from each equation. Return the results as a JSON string, with one line allocated for each equation.\n```\n1) y = (3/4)x^3 - e^(2x) + sin(pi*x) - sqrt(7)\n2) 2A - B/(3+C) * sum(N=1 to 5; ln(N)^2) = 5D*integral(a=0 to pi; cos(comb(N=1 to 10; N*a)))\n3) E = m(c^2) + gamma*(v/d)/(-(alpha/2) + sqrt(beta^2 + (alpha/2)^2))\n```', 
    #     'In the field of quantum physics, what is superposition, and how does it relate to the phenomenon of quantum entanglement?', 
    #     "Consider a satellite that is in a circular orbit around the Earth. The speed of the satellite decreases. What will happen to the satellite's orbital radius and period of revolution? Please justify your answer using principles of physics.", 
    #     'Photosynthesis is a vital process for life on Earth. Could you outline the two main stages of photosynthesis, including where they take place within the chloroplast, and the primary inputs and outputs for each stage?', 
    #     'Please explain the differences between exothermic and endothermic reactions, and include the criteria you used to distinguish between them. Additionally, please provide a real-world example to illustrate your explanation.', 
    #     'The city of Vega intends to build a bridge that will span the Vegona River, covering a distance of 1.8 kilometers. The proposed location falls within a seismically active area that has experienced several high-magnitude earthquakes. Given these circumstances, what would be the best approach to constructing the bridge?', 
    #     'You have been tasked with designing a solar-powered water heating system for a residential building. Describe the key components and considerations you would include in your design. Design a five-step workflow.', 
    #     'Please describe the concept of machine learning. Could you elaborate on the differences between supervised, unsupervised, and reinforcement learning? Provide real-world examples of each.', 
    #     'Provide insights into the correlation between economic indicators such as GDP, inflation, and unemployment rates. Explain how fiscal and monetary policies affect those indicators.', 
    #     'How do the stages of life shape our understanding of time and mortality?', 
    #     'Discuss antitrust laws and their impact on market competition. Compare the antitrust laws in US and China along with some case studies.', 
    #     'Create a lesson plan that integrates drama, mime or theater techniques into a history class. Duration: 3 class periods (each lasts for 45 minutes) for 3 days\nTopic: Opium Wars between China and Britain\nGrade level: 9-10', 
    #     'Share ideas for adapting art masterpieces into interactive experiences for children. List 5 specific artworks and associated ideas.',
    #     'Which methods did Socrates employ to challenge the prevailing thoughts of his time?', 
    #     'What are some business etiquette norms when doing business in Japan?'
    # ]

    dataset = InferenceDataset(data, tokenizer_name=HF_MODEL_NAME)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    id2label ={ 0: "ROUTE_TO_INFERIOR", 1: "ROUTE_TO_SUPERIOR" }

    preds = infer(model, loader)
    for pred_id, text in zip(preds, data):
        print(f"{id2label[pred_id]} <-- {repr(text)}")